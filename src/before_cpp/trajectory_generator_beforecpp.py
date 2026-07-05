import os
import math
import struct
import numpy as np
import torch
from PIL import Image

# Функція конвертації матриці обертання в кватерніон (потрібно для COLMAP images.bin)
# ВИПРАВЛЕНО: повна чотирьохгілкова стабільна реалізація (Shepperd's method).
# Стара версія мала лише гілку tr > 0 і тихо повертала identity-кватерніон
# для будь-якої матриці з trace <= 0, що псує позу кадру в COLMAP.
def rmat_to_qvec(R):
    Rxx, Ryx, Rzx = R[0,0], R[1,0], R[2,0]
    Rxy, Ryy, Rzy = R[0,1], R[1,1], R[2,1]
    Rxz, Ryz, Rzz = R[0,2], R[1,2], R[2,2]
    tr = Rxx + Ryy + Rzz

    if tr > 0:
        am = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * am
        qx = (Rzy - Ryz) / am
        qy = (Rxz - Rzx) / am
        qz = (Ryx - Rxy) / am
    elif (Rxx > Ryy) and (Rxx > Rzz):
        am = math.sqrt(1.0 + Rxx - Ryy - Rzz) * 2
        qw = (Rzy - Ryz) / am
        qx = 0.25 * am
        qy = (Rxy + Ryx) / am
        qz = (Rxz + Rzx) / am
    elif Ryy > Rzz:
        am = math.sqrt(1.0 + Ryy - Rxx - Rzz) * 2
        qw = (Rxz - Rzx) / am
        qx = (Rxy + Ryx) / am
        qy = 0.25 * am
        qz = (Ryz + Rzy) / am
    else:
        am = math.sqrt(1.0 + Rzz - Rxx - Ryy) * 2
        qw = (Ryx - Rxy) / am
        qx = (Rxz + Rzx) / am
        qy = (Ryz + Rzy) / am
        qz = 0.25 * am

    qvec = np.array([qw, qx, qy, qz])
    return qvec / np.linalg.norm(qvec)

class SyntheticOrbitRenderer:
    def __init__(self, cloud_path: str, output_dir: str):
        self.output_dir = output_dir
        self.sparse_dir = os.path.join(output_dir, "sparse", "0")
        self.images_dir = os.path.join(output_dir, "images")
        
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Завантажуємо нашу згенеровану хмару точок
        data = torch.load(cloud_path, map_location="cpu", weights_only=False)
        self.xyz = data["xyz"].numpy()
        self.rgb = data["rgb"].numpy()

        # Параметри віртуальної камери (Pinhole)
        self.W, self.H = 800, 600
        self.fx = self.fy = 600.0
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

        # Обчислюємо географічний центр сцени
        self.center = self.xyz.mean(axis=0)
        self.center[1] = 1.5  # Фіксуємо фокус на середній висоті стін (3м / 2)

    def render_point_cloud(self, R, T):
        """ Швидкий рендеринг хмари точок на основі Painter's Algorithm """
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8) + 30 # темне тло
        
        # Трансформація точок у простір камери: X_c = R * X_w + T
        xyz_cam = (self.xyz @ R.T) + T
        
        # Відсікаємо точки, що знаходяться позаду камери
        mask = xyz_cam[:, 2] > 0.1
        if not np.any(mask):
            return img
            
        xyz_cam = xyz_cam[mask]
        rgb_filtered = self.rgb[mask]

        # Проєкція на екран камери
        u = ((xyz_cam[:, 0] * self.fx) / xyz_cam[:, 2] + self.cx).astype(np.int32)
        v = ((xyz_cam[:, 1] * self.fy) / xyz_cam[:, 2] + self.cy).astype(np.int32)

        # Фільтрація меж екрану
        valid = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        if not np.any(valid):
            return img

        u, v = u[valid], v[valid]
        z_depth = xyz_cam[valid, 2]
        rgb_valid = (rgb_filtered[valid] * 255).astype(np.uint8)

        # Сортування точок від далеких до близьких (Художній алгоритм)
        sort_idx = np.argsort(-z_depth)
        
        # Малюємо точки на кадрі
        img[v[sort_idx], u[sort_idx]] = rgb_valid[sort_idx]
        return img

    def generate_orbit(self, num_frames=120):
        radius = 14.0  # Радіус обльоту комплексу
        height = 5.0   # Висота польоту дрона

        cameras_bin_data = []
        images_bin_data = []

        print(f"🎬 Генерація {num_frames} синтетичних кадрів орбіти...")

        for i in range(num_frames):
            angle = (i / num_frames) * 2 * math.pi
            
            # Позиція камери у світових координатах
            cam_x = self.center[0] + radius * math.cos(angle)
            cam_z = self.center[2] + radius * math.sin(angle)
            cam_y = height
            cam_pos = np.array([cam_x, cam_y, cam_z])

            # Розрахунок матриці орієнтації Look-At (камера дивиться чітко в центр)
            forward = self.center - cam_pos
            forward /= np.linalg.norm(forward)
            
            right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
            right /= np.linalg.norm(right)
            
            up = np.cross(forward, right)

            # Світова матриця в матрицю камери (World-to-Camera)
            R = np.vstack([right, up, forward])
            T = -R @ cam_pos

            # Рендеримо штучне зображення
            frame = self.render_point_cloud(R, T)
            img_name = f"frame_{i:04d}.jpg"
            Image.fromarray(frame).save(os.path.join(self.images_dir, img_name))

            # Зберігаємо дані для генерації фейкового COLMAP
            qvec = rmat_to_qvec(R)
            images_bin_data.append((i + 1, qvec, T, 1, img_name))

        # Запис FAKE cameras.bin (Модель 1 - PINHOLE)
        with open(os.path.join(self.sparse_dir, "cameras.bin"), "wb") as f:
            f.write(struct.pack("<Q", 1)) # 1 камера в системі
            f.write(struct.pack("<iiQQ", 1, 1, self.W, self.H)) # id=1, model=1 (PINHOLE), W, H
            f.write(struct.pack("<dddd", self.fx, self.fy, self.cx, self.cy))

        # Запис FAKE images.bin
        with open(os.path.join(self.sparse_dir, "images.bin"), "wb") as f:
            f.write(struct.pack("<Q", len(images_bin_data)))
            for img_id, qvec, tvec, cam_id, name in images_bin_data:
                f.write(struct.pack("<i", img_id))
                f.write(struct.pack("<dddd", *qvec))
                f.write(struct.pack("<ddd", *tvec))
                f.write(struct.pack("<i", cam_id))
                f.write(name.encode('utf-8') + b"\x00")
                f.write(struct.pack("<Q", 0)) # 0 2D точок (фічі не потрібні для навчання)

        print("✨ Пайплайн 'Schematic-to-Tactical-Splat' успішно згенерував Fake-COLMAP структуру!")

if __name__ == "__main__":
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CLOUD_PT = os.path.join(PROJECT_DIR, "data", "cloud.pt")
    COLMAP_OUT = os.path.join(PROJECT_DIR, "data", "colmap_output")
    
    renderer = SyntheticOrbitRenderer(CLOUD_PT, COLMAP_OUT)
    renderer.generate_orbit(num_frames=120)