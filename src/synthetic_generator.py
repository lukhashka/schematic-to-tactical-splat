import torch
import json
import splat_generator_core

class AnalyticalSplatGenerator:
    def __init__(self, layout_path: str):
        with open(layout_path, "r", encoding="utf-8") as f:
            self.layout = json.load(f)
        
        self.materials = self.layout["materials"]
        self.density = self.layout["metadata"]["point_density_per_meter"]

    def build_scene_fast(self):
        wall_color = self.materials.get("wall", [0.6, 0.6, 0.6])
        floor_color = self.materials.get("floor", [0.88, 0.9, 0.92])
        
        # Парсимо стіни (явне приведення до float)
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
            
        # Парсимо підлогу (явне приведення до float)
        cxx_floors = []
        for f in self.layout["floors"]:
            cxx_floors.append(splat_generator_core.Floor(
                float(f["min_x"]), float(f["max_x"]),
                float(f["min_z"]), float(f["max_z"]),
                float(f["y"])
            ))
            
        # Викликаємо монолітну C++ функцію
        xyz, scale, rotation, opacity, rgb = splat_generator_core.generate_scene(
            cxx_walls, cxx_floors, float(self.density), wall_color, floor_color
        )

        # Перетворюємо масиви на тензори з гарантованим типом float32
        return (
            torch.tensor(xyz, dtype=torch.float32),
            torch.tensor(scale, dtype=torch.float32),
            torch.tensor(rotation, dtype=torch.float32),
            torch.tensor(opacity, dtype=torch.float32),
            torch.tensor(rgb, dtype=torch.float32)
        )

    def save_splat(self, output_path: str):
        xyz, scale, rotation, opacity, rgb = self.build_scene_fast()
        torch.save({
            "xyz": xyz, "scale": scale, "rotation": rotation, "opacity": opacity, "rgb": rgb
        }, output_path)
        print(f"✨ [C++ Monolith] Сцену згенеровано успішно! Разом точок: {xyz.shape[0]:,}")