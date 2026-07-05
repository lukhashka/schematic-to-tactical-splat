import torch
import json
import numpy as np

# ВИПРАВЛЕНО: раніше `import splat_generator_core` виконувався безумовно на
# рівні модуля. Якщо розширення ще не скомпільоване, це кидало ImportError
# одразу при імпорті synthetic_generator — а оскільки main.py тепер імпортує
# synthetic_generator на рівні модуля (щоб не робити sys.path-хаки на кожен
# запит), відсутність .so файлу валила б увесь FastAPI-сервер при старті.
# Тепер відсутність розширення просто вимикає AnalyticalSplatGenerator із
# чітким повідомленням при спробі використання, а не при імпорті.
try:
    import splat_generator_core
    CPP_CORE_AVAILABLE = True
except ImportError:
    splat_generator_core = None
    CPP_CORE_AVAILABLE = False


class AnalyticalSplatGenerator:
    def __init__(self, layout_path: str):
        if not CPP_CORE_AVAILABLE:
            raise RuntimeError(
                "splat_generator_core недоступний (C++ розширення не скомпільоване). "
                "Зберіть його (напр. `pip install -e .` у теці з setup.py) перед генерацією сцени."
            )

        with open(layout_path, "r", encoding="utf-8") as f:
            self.layout = json.load(f)

        self.materials = self.layout["materials"]
        self.density = self.layout["metadata"]["point_density_per_meter"]

        # ВИПРАВЛЕНО: обчислення кутів стін було втрачено при переході на
        # C++ ядро (використовувалось для AO-затемнення біля кутів). Тепер
        # рахуємо їх тут же, як і в старій Python-версії, і передаємо в C++
        # плоским масивом [x0,z0,x1,z1,...].
        corners = []
        for w in self.layout["walls"]:
            if w.get("type", "wall") == "wall":
                corners.append(w["start"])
                corners.append(w["end"])
        unique_corners = np.unique(np.array(corners), axis=0) if corners else np.empty((0, 2))
        self.corners_flat = [float(v) for pair in unique_corners for v in pair]

    def build_scene_fast(self):
        wall_color = self.materials.get("wall", [0.6, 0.6, 0.6])
        floor_color = self.materials.get("floor", [0.88, 0.9, 0.92])

        cxx_walls = []
        for w in self.layout["walls"]:
            w_type = 1 if w.get("type", "wall") == "door" else 0
            normal = w.get("normal", [0.0, 0.0])
            cxx_walls.append(splat_generator_core.Wall(
                float(w["start"][0]), float(w["start"][1]),
                float(w["end"][0]), float(w["end"][1]),
                float(w.get("height", 3.0)),
                float(normal[0]), float(normal[1]),
                int(w_type)
            ))

        cxx_floors = []
        for f in self.layout["floors"]:
            cxx_floors.append(splat_generator_core.Floor(
                float(f["min_x"]), float(f["max_x"]),
                float(f["min_z"]), float(f["max_z"]),
                float(f["y"])
            ))

        # C++ ядро тепер повертає ПЛОСКІ масиви (xyz: 3/точку, scale: 3/точку,
        # rotation: 4/точку, opacity: 1/точку, rgb: 3/точку) — набагато швидше
        # за попередню версію, яка пакувала кожну точку в окремий std::vector.
        xyz_flat, scale_flat, rotation_flat, opacity_flat, rgb_flat = splat_generator_core.generate_scene(
            cxx_walls, cxx_floors, float(self.density), wall_color, floor_color, self.corners_flat
        )

        xyz = torch.tensor(xyz_flat, dtype=torch.float32).reshape(-1, 3)
        scale = torch.tensor(scale_flat, dtype=torch.float32).reshape(-1, 3)
        rotation = torch.tensor(rotation_flat, dtype=torch.float32).reshape(-1, 4)
        opacity = torch.tensor(opacity_flat, dtype=torch.float32)
        rgb = torch.tensor(rgb_flat, dtype=torch.float32).reshape(-1, 3)

        return xyz, scale, rotation, opacity, rgb

    def save_splat(self, output_path: str):
        xyz, scale, rotation, opacity, rgb = self.build_scene_fast()
        torch.save({
            "xyz": xyz, "scale": scale, "rotation": rotation, "opacity": opacity, "rgb": rgb
        }, output_path)
        print(f"✨ [C++ Monolith] Сцену згенеровано успішно! Разом точок: {xyz.shape[0]:,}")