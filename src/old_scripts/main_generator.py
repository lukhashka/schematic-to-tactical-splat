import os
import json
import numpy as np
import torch

class SyntheticSceneGenerator:
    def __init__(self, layout_path: str):
        with open(layout_path, "r") as f:
            self.layout = json.load(f)
        
        self.meta = self.layout["metadata"]
        self.materials = self.layout["materials"]
        self.density = self.meta["point_density_per_meter"]
        self.height = self.meta["global_height"]
        
        self.points_xyz = []
        self.points_rgb = []

    def _add_point(self, x: float, y: float, z: float, color_type: str):
        """Додає точку в загальний масив із легким шумом для стабільності 3DGS"""
        noise = np.random.normal(0, 0.005, 3) # мікро-шум, щоб Гаусіани не злипалися в нуль
        self.points_xyz.append([x + noise[0], y + noise[1], z + noise[2]])
        self.points_rgb.append(self.materials.get(color_type, [0.5, 0.5, 0.5]))

    def generate_walls(self):
        """Дискретизує лінії стін у 3D хмару точок"""
        for wall in self.layout["walls"]:
            start = np.array(wall["start"])
            end = np.array(wall["end"])
            w_type = wall["type"]
            
            # Рахуємо довжину стіни та кількість кроків по горизонталі
            length = np.linalg.norm(end - start)
            steps_h = int(max(2, length * self.density))
            steps_v = int(max(2, self.height * self.density))
            
            # Генеруємо сітку точок для стіни
            for h_step in range(steps_h):
                alpha = h_step / (steps_h - 1)
                curr_xz = start + alpha * (end - start)
                
                for v_step in range(steps_v):
                    curr_y = (v_step / (steps_v - 1)) * self.height
                    self._add_point(curr_xz[0], curr_y, curr_xz[1], w_type)

    def generate_floors(self):
        """Заповнює площину підлоги точками"""
        for floor in self.layout["floors"]:
            steps_x = int(max(2, (floor["max_x"] - floor["min_x"]) * self.density))
            steps_z = int(max(2, (floor["max_z"] - floor["min_z"]) * self.density))
            
            for i in range(steps_x):
                x = floor["min_x"] + (i / (steps_x - 1)) * (floor["max_x"] - floor["min_x"])
                for j in range(steps_z):
                    z = floor["min_z"] + (j / (steps_z - 1)) * (floor["max_z"] - floor["min_z"])
                    self._add_point(x, floor["y"], z, "floor")

    def save(self, output_path: str):
        self.generate_walls()
        self.generate_floors()
        
        xyz_tensor = torch.tensor(self.points_xyz, dtype=torch.float32)
        rgb_tensor = torch.tensor(self.points_rgb, dtype=torch.float32)
        
        data = {
            "xyz": xyz_tensor,
            "rgb": rgb_tensor
        }
        
        torch.save(data, output_path)
        print(f"========================================================")
        print(f" ✅ СИНТЕТИЧНУ ХМАРУ СТВОРЕНО")
        print(f"========================================================")
        print(f" 📂 Збережено в: {output_path}")
        print(f" 🔢 Кількість згенерованих точок: {xyz_tensor.shape[0]:,}")
        print(f"========================================================")

if __name__ == "__main__":
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Тепер шлях буде збиратися ідеально
    LAYOUT = os.path.join(PROJECT_DIR, "data", "layout.json")
    OUTPUT = os.path.join(PROJECT_DIR, "data", "cloud.pt") # Підміняємо вхідний файл для тренера
    
    generator = SyntheticSceneGenerator(LAYOUT)
    generator.save(OUTPUT)