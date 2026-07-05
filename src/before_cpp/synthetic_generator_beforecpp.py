import os
import json
import math
import numpy as np
import torch
from pathlib import Path

class AnalyticalSplatGenerator:
    def __init__(self, layout_path: str):
        with open(layout_path, "r", encoding="utf-8") as f:
            self.layout = json.load(f)
        
        self.materials = self.layout["materials"]
        self.density = self.layout["metadata"]["point_density_per_meter"]
        
        self.corners = []
        for wall in self.layout["walls"]:
            if wall.get("type", "wall") == "wall":
                self.corners.append(wall["start"])
                self.corners.append(wall["end"])
        self.corners = np.unique(np.array(self.corners), axis=0) if self.corners else np.array([])

        self.xyz, self.scale, self.rotation, self.opacity, self.rgb = [], [], [], [], []

    def _calculate_lighting_and_ao(self, x: float, y: float, z: float, angle_rad: float, base_color: list, wall_height: float, is_floor: bool = False) -> list:
        ao_factor = 1.0
        shadow_zone = 0.3
        
        if y < shadow_zone:
            ao_factor *= 0.45 + 0.55 * (y / shadow_zone)
        elif (wall_height - y) < shadow_zone:
            ao_factor *= 0.45 + 0.55 * ((wall_height - y) / shadow_zone)

        if not is_floor and self.corners.size > 0:
            curr_p = np.array([x, z])
            for corner in self.corners:
                if np.linalg.norm(curr_p - corner) < 0.3:
                    ao_factor *= 0.5 + 0.5 * (np.linalg.norm(curr_p - corner) / 0.3)

        ao_factor = max(0.2, ao_factor)
        sun_angle = math.pi / 4
        
        if is_floor:
            sun_factor = 1.05
        else:
            sun_factor = 1.0 + 0.12 * math.cos(angle_rad - sun_angle)

        return [
            max(0.0, min(1.0, base_color[0] * ao_factor * sun_factor)),
            max(0.0, min(1.0, base_color[1] * ao_factor * sun_factor)),
            max(0.0, min(1.0, base_color[2] * ao_factor * sun_factor))
        ]

    def build_scene(self):
        step_h = 1.0 / self.density
        step_v = 1.0 / self.density

        # 1. ГЕНЕРАЦІЯ СТІН ІЗ ЧІТКИМИ ВНУТРІШНІМИ НОРМАЛЯМИ КІМНАТ
        for wall in self.layout["walls"]:
            start = np.array(wall["start"])
            end = np.array(wall["end"])
            w_type = wall.get("type", "wall")
            room_height = wall.get("height", 3.0)
            color = self.materials.get(w_type, [0.6, 0.6, 0.6])
            
            # Зчитуємо готову орієнтацію стіни з фронтенду
            # Якщо раптом немає (старий json), ставимо за замовчуванням 0
            normal = wall.get("normal", [0.0, 0.0])
            normal_x = normal[0]
            normal_z = normal[1]
            
            vec = end - start
            length = np.linalg.norm(vec)
            if length < 1e-4: continue
            
            angle = math.atan2(-vec[1], vec[0])
            
            # Зсуваємо стіну СТРОГО всередину її власної кімнати на 4 см
            offset_dist = 0.04
            
            num_steps_h = int(max(2, length * self.density))
            door_height = 2.1 

            y_min, y_max = (door_height, room_height) if w_type == "door" else (0.0, room_height)
            if y_min >= y_max: continue
            
            num_steps_v = int(max(2, (y_max - y_min) * self.density))

            for h in range(num_steps_h):
                alpha = h / (num_steps_h - 1)
                curr_xz = start + alpha * vec
                
                # Застосовуємо абсолютно правильний внутрішній зсув
                shifted_x = curr_xz[0] + normal_x * offset_dist
                shifted_z = curr_xz[1] + normal_z * offset_dist
                
                for v in range(num_steps_v):
                    curr_y = y_min + (v / (num_steps_v - 1)) * (y_max - y_min)
                    
                    self.xyz.append([shifted_x, curr_y, shifted_z])
                    self.scale.append([step_h * 1.3, step_v * 1.3, 0.001])
                    self.rotation.append([float(math.cos(angle / 2)), 0.0, float(math.sin(angle / 2)), 0.0])
                    self.opacity.append(1.0)
                    self.rgb.append(self._calculate_lighting_and_ao(shifted_x, curr_y, shifted_z, angle, color, room_height, is_floor=False))

        # 2. ЗАЛІЗОБЕТОННА ГЕНЕРАЦІЯ ПІДЛОГИ
        for floor in self.layout["floors"]:
            color = self.materials.get("floor", [0.88, 0.9, 0.92])
            len_x = floor["max_x"] - floor["min_x"]
            len_z = floor["max_z"] - floor["min_z"]
            
            num_x = int(max(2, len_x * self.density))
            num_z = int(max(2, len_z * self.density))
            
            for i in range(num_x):
                x = floor["min_x"] + (i / (num_x - 1)) * len_x
                for j in range(num_z):
                    z = floor["min_z"] + (j / (num_z - 1)) * len_z
                    
                    self.xyz.append([x, floor["y"], z])
                    # Увага: передаємо однаковий крок по X та Z, а товщину ставимо рівно в 0.0!
                    # Більше ніяких 0.002, які плутають шейдер
                    self.scale.append([step_h * 1.3, 0.0, step_v * 1.3])
                    self.rotation.append([1.0, 0.0, 0.0, 0.0]) # Нульовий кватерніон
                    self.opacity.append(1.0)
                    self.rgb.append(self._calculate_lighting_and_ao(x, floor["y"], z, 0.0, color, 3.0, is_floor=True))

    def save_splat(self, output_path: str):
        self.build_scene()
        torch.save({
            "xyz": torch.tensor(self.xyz, dtype=torch.float32),
            "scale": torch.tensor(self.scale, dtype=torch.float32),
            "rotation": torch.tensor(self.rotation, dtype=torch.float32),
            "opacity": torch.tensor(self.opacity, dtype=torch.float32),
            "rgb": torch.tensor(self.rgb, dtype=torch.float32)
        }, output_path)
        print(f"✨ Сцену згенеровано! Точок: {len(self.xyz):,}")