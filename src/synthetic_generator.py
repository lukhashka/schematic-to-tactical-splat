import torch
import json
import numpy as np

# Зберігаємо безпечний імпорт розширення, щоб не валити FastAPI при старті[cite: 5]
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
                "Зберіть його перед генерацією сцени."
            )

        with open(layout_path, "r", encoding="utf-8") as f:
            self.layout = json.load(f)

        self.materials = self.layout["materials"]
        self.density = self.layout["metadata"]["point_density_per_meter"]

        # Розрахунок кутів стін для Ambient Occlusion[cite: 5]
        corners = []
        for w in self.layout["walls"]:
            if w.get("type", "wall") == "wall":
                corners.append(w["start"])
                corners.append(w["end"])
        unique_corners = np.unique(np.array(corners), axis=0) if corners else np.empty((0, 2))
        self.corners_flat = [float(v) for pair in unique_corners for v in pair]

    def _parse_cxx_objects(self):
        """
        ВИПРАВЛЕНО: раніше цей парсинг (стіни/отвори/підлоги/дірки → C++
        об'єкти) був написаний лише один раз, всередині build_scene_fast().
        Тепер, коли з'явився build_collision_mesh() (Layer 2), обидва методи
        мають працювати з ОДНАКОВИМИ walls/floors/holes — інакше колізійна
        сітка й візуальна хмара точок можуть розійтись, якщо хтось поправить
        парсинг в одному місці й забуде про інше. Тому парсинг тепер один,
        спільний.
        """
        cxx_walls = []
        for w in self.layout["walls"]:
            normal = w.get("normal", [0.0, 0.0])
            openings = []

            # Зворотна сумісність: якщо редактор прислав готовий відрізок дверей
            if w.get("type", "wall") == "door":
                dx = w["end"][0] - w["start"][0]
                dz = w["end"][1] - w["start"][1]
                length = float((dx**2 + dz**2)**0.5)
                openings.append(splat_generator_core.Opening(0.0, length, 0.0, 2.1))

            # Майбутній розширений формат: якщо стіна монолітна, але має масив вбудованих вікон/отворів
            if "openings" in w:
                for op in w["openings"]:
                    v_start = op.get("vert_start", 0.0)
                    v_end = op.get("vert_end", 2.1)

                    openings.append(splat_generator_core.Opening(
                        float(op["horiz_start"]), float(op["horiz_end"]),
                        float(v_start), float(v_end)
                    ))

            cxx_walls.append(splat_generator_core.Wall(
                float(w["start"][0]), float(w["start"][1]),
                float(w["end"][0]), float(w["end"][1]),
                float(w.get("height", 3.0)),
                float(normal[0]), float(normal[1]),
                openings
            ))

        cxx_floors = []
        for f in self.layout["floors"]:
            cxx_floors.append(splat_generator_core.Floor(
                float(f["min_x"]), float(f["max_x"]),
                float(f["min_z"]), float(f["max_z"]),
                float(f["y"])
            ))

        cxx_holes = []
        if "holes" in self.layout:
            for h in self.layout["holes"]:
                cxx_holes.append(splat_generator_core.DestructionHole(
                    float(h["x"]), float(h["y"]), float(h["z"]), float(h["radius"])
                ))

        return cxx_walls, cxx_floors, cxx_holes

    def build_scene_fast(self):
        wall_color = self.materials.get("wall", [0.6, 0.6, 0.6])
        floor_color = self.materials.get("floor", [0.88, 0.9, 0.92])

        cxx_walls, cxx_floors, cxx_holes = self._parse_cxx_objects()

        # Виклик оновленого C++ ядра з підтримкою 3D руйнувань та прорізів[cite: 6]
        xyz_flat, scale_flat, rotation_flat, opacity_flat, rgb_flat = splat_generator_core.generate_scene(
            cxx_walls, cxx_floors, cxx_holes, float(self.density), wall_color, floor_color, self.corners_flat
        )

        # Пакування плоских масивів у PyTorch тензори[cite: 5]
        xyz = torch.tensor(xyz_flat, dtype=torch.float32).reshape(-1, 3)
        scale = torch.tensor(scale_flat, dtype=torch.float32).reshape(-1, 3)
        rotation = torch.tensor(rotation_flat, dtype=torch.float32).reshape(-1, 4)
        opacity = torch.tensor(opacity_flat, dtype=torch.float32)
        rgb = torch.tensor(rgb_flat, dtype=torch.float32).reshape(-1, 3)

        return xyz, scale, rotation, opacity, rgb

    def build_collision_mesh(self, cell_size: float = 0.2):
        """
        Layer 2 — точна колізійна/симуляційна геометрія (box'и стін + плити
        підлог, з вирізаними отворами/дірками) для THREE.Raycaster +
        three-mesh-bvh на клієнті. Повністю НЕЗАЛЕЖНА від point_density_per_meter
        і від точкової хмари — саме це прибирає "зубчастість"/"протікання"
        старого LOS, побудованого проти хмари точок.
        """
        cxx_walls, cxx_floors, cxx_holes = self._parse_cxx_objects()

        vertices_flat, indices_flat = splat_generator_core.generate_collision_mesh(
            cxx_walls, cxx_floors, cxx_holes, float(cell_size)
        )
        return vertices_flat, indices_flat

    def save_splat(self, output_path: str):
        xyz, scale, rotation, opacity, rgb = self.build_scene_fast()
        torch.save({
            "xyz": xyz, "scale": scale, "rotation": rotation, "opacity": opacity, "rgb": rgb
        }, output_path)
        print(f"✨ [C++ Monolith] Сцену згенеровано успішно! Разом точок: {xyz.shape[0]:,}")