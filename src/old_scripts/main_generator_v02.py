import os
import json
import numpy as np
import torch
from pathlib import Path

class AnalyticalSplatGenerator:
    def __init__(self, layout_path: str):
        with open(layout_path, "r", encoding="utf-8") as f:
            self.layout = json.load(f)
        
        self.meta = self.layout["metadata"]
        self.materials = self.layout["materials"]
        self.density = self.meta["point_density_per_meter"]
        self.height = self.meta["global_height"]
        
        # Списки для збору параметрів 3DGS
        self.xyz = []
        self.scale = []
        self.rotation = []
        self.opacity = []
        self.rgb = []

    def _add_wall_gaussian(self, x: float, y: float, z: float, scale_h: float, scale_v: float, angle_rad: float, color_rgb: list):
        """Додає один плоский Гаусіан стіни, повернутий під потрібним кутом"""
        self.xyz.append([x, y, z])
        
        # Масштаб: розтягуємо по горизонталі та вертикалі, робимо ультра-тонким по товщині (0.002м)
        # Множимо на 1.2, щоб сусідні Гаусіани заходили один за одного й перекривали дірки
        self.scale.append([scale_h * 1.2, scale_v * 1.2, 0.002])
        
        # Розрахунок кватерніона обертання навколо вертикальної осі Y [W, X, Y, Z]
        qw = float(np.cos(angle_rad / 2))
        qy = float(np.sin(angle_rad / 2))
        self.rotation.append([qw, 0.0, qy, 0.0])
        
        self.opacity.append(1.0)  # Залізобетонна непрозорість
        self.rgb.append(color_rgb)

    def _add_floor_gaussian(self, x: float, y: float, z: float, scale_x: float, scale_z: float, color_rgb: list):
        """Додає один плоский горизонтальний Гаусіан підлоги"""
        self.xyz.append([x, y, z])
        self.scale.append([scale_x * 1.2, 0.002, scale_z * 1.2]) # тонкий по вертикалі
        self.rotation.append([1.0, 0.0, 0.0, 0.0]) # без повороту
        self.opacity.append(1.0)
        self.rgb.append(color_rgb)

    def build_scene(self):
        # 1. Генеруємо стіни
        step_h = 1.0 / self.density  # крок між точками по горизонталі (напр. 0.05м)
        step_v = 1.0 / self.density  # крок по вертикалі

        for wall in self.layout["walls"]:
            start = np.array(wall["start"])
            end = np.array(wall["end"])
            color = self.materials.get(wall["type"], [0.5, 0.5, 0.5])
            
            vec = end - start
            length = np.linalg.norm(vec)
            angle = math.atan2(-vec[1], vec[0]) # Кут повороту стіни в просторі
            
            dir_vector = vec / length
            num_steps_h = int(max(2, length * self.density))
            num_steps_v = int(max(2, self.height * self.density))
            
            for h in range(num_steps_h):
                alpha = h / (num_steps_h - 1)
                curr_xz = start + alpha * vec
                
                for v in range(num_steps_v):
                    curr_y = (v / (num_steps_v - 1)) * self.height
                    
                    # Додаємо плоский Гаусіан розміром з наш крок дискретизації
                    self._add_wall_gaussian(
                        x=curr_xz[0], y=curr_y, z=curr_xz[1],
                        scale_h=step_h, scale_v=step_v,
                        angle_rad=angle, color_rgb=color
                    )

        # 2. Генеруємо підлогу
        for floor in self.layout["floors"]:
            color = self.materials.get("floor", [0.3, 0.25, 0.2])
            len_x = floor["max_x"] - floor["min_x"]
            len_z = floor["max_z"] - floor["min_z"]
            
            num_x = int(max(2, len_x * self.density))
            num_z = int(max(2, len_z * self.density))
            
            for i in range(num_x):
                x = floor["min_x"] + (i / (num_x - 1)) * len_x
                for j in range(num_z):
                    z = floor["min_z"] + (j / (num_z - 1)) * len_z
                    
                    self._add_floor_gaussian(
                        x=x, y=floor["y"], z=z,
                        scale_x=step_h, scale_z=step_v, color_rgb=color
                    )

    def save_splat(self, output_path: str):
        import math
        self.build_scene()
        
        # Пакуємо все в тензори структури 3DGS
        data = {
            "xyz":      torch.tensor(self.xyz, dtype=torch.float32),
            "scale":    torch.tensor(self.scale, dtype=torch.float32),
            "rotation": torch.tensor(self.rotation, dtype=torch.float32),
            "opacity":  torch.tensor(self.opacity, dtype=torch.float32),
            "rgb":      torch.tensor(self.rgb, dtype=torch.float32)
        }
        
        torch.save(data, output_path)
        print(f"========================================================")
        print(f" ✅ АНАЛІТИЧНИЙ МОНОЛІТНИЙ СПЛАТ СТВОРЕНО (БЕЗ ДІРОК)")
        print(f"========================================================")
        print(f" 💾 Файл збережено в: {output_path}")
        print(f" 🔢 Всього Гаусіанів: {len(self.xyz):,}")
        print(f"========================================================")

if __name__ == "__main__":
    import math
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Тепер шлях буде збиратися ідеально
    LAYOUT = os.path.join(PROJECT_DIR, "data", "layout.json")
    OUTPUT_SPLAT = os.path.join(PROJECT_DIR, "data", "cloud.pt") # Підміняємо вхідний файл для тренера
    
    generator = AnalyticalSplatGenerator(str(LAYOUT))
    generator.save_splat(str(OUTPUT_SPLAT))