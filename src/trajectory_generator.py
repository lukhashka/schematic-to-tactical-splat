import os
import math
import struct
import numpy as np
import torch
from PIL import Image

def rmat_to_qvec(R):
    """ Повна стабільна реалізація методу Шепперда """
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

class TacticalOrbitRenderer:
    def __init__(self, cloud_path: str, output_dir: str):
        self.output_dir = output_dir
        self.sparse_dir = os.path.join(output_dir, "sparse", "0")
        self.images_dir = os.path.join(output_dir, "images")
        
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        data = torch.load(cloud_path, map_location="cpu", weights_only=False)
        self.xyz = data["xyz"].numpy()
        self.rgb = data["rgb"].numpy()

        self.W, self.H = 800, 600
        self.fx = self.fy = 600.0
        self.cx, self.cy = self.W / 2.0, self.H / 2.0

        self.center = self.xyz.mean(axis=0)
        
        # Визначаємо радіуси сцени для побудови кращої траєкторії
        self.min_bounds = self.xyz.min(axis=0)
        self.max_bounds = self.xyz.max(axis=0)

    def render_point_cloud_advanced(self, R, T):
        """ Рендеринг з антиаліасингом: суперсемплінг розмиття меж точок """
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8) + 25 
        
        xyz_cam = (self.xyz @ R.T) + T
        mask = xyz_cam[:, 2] > 0.1
        if not np.any(mask): return img
            
        xyz_cam = xyz_cam[mask]
        rgb_filtered = self.rgb[mask]

        u = ((xyz_cam[:, 0] * self.fx) / xyz_cam[:, 2] + self.cx).astype(np.int32)
        v = ((xyz_cam[:, 1] * self.fy) / xyz_cam[:, 2] + self.cy).astype(np.int32)

        # Буфер з невеликим запасом на краї під сплейтинг
        pad = 3
        valid = (u >= pad) & (u < self.W - pad) & (v >= pad) & (v < self.H - pad)
        if not np.any(valid): return img

        u, v = u[valid], v[valid]
        z_depth = xyz_cam[valid, 2]
        rgb_valid = (rgb_filtered[valid] * 255).astype(np.uint8)

        # Сортуємо (алгоритм художника)
        sort_idx = np.argsort(-z_depth)
        u_s, v_s, rgb_s = u[sort_idx], v[sort_idx], rgb_valid[sort_idx]

        # Малюємо базові центри
        img[v_s, u_s] = rgb_s

        # М'який антиаліасинг: векторизоване розмиття суміжних пікселів для знищення муару
        # Змішуємо поточний колір із сусідніми з коефіцієнтом прозорості (імітуємо радіус гауссіана на екрані)
        for du in [-1, 1]:
            img[v_s, u_s + du] = (img[v_s, u_s + du].astype(uint16) * 4 + rgb_s.astype(uint16) * 3) // 7
        for dv in [-1, 1]:
            img[v_s + dv, u_s] = (img[v_s + dv, u_s].astype(uint16) * 4 + rgb_s.astype(uint16) * 3) // 7

        return img

    def generate_complex_trajectory(self, num_frames=150):
        """ Траєкторія «Спіраль + Зазирання»: дрон змінює висоту і кут тангажу, щоб бачити внутрішній простір кімнат """
        cameras_bin_data = []
        images_bin_data = []

        print(f"🎬 Генерація {num_frames} тактичних кадрів з оглядом кімнат...")

        for i in range(num_frames):
            # 1. Складна просторова траєкторія (зміна радіусу та висоти)
            angle = (i / num_frames) * 2 * math.pi
            
            # Динамічний радіус: то ближче (щоб зазирнути вглиб), то далі
            current_radius = 11.0 + 3.0 * math.sin(2 * angle)
            
            # Динамічна висота: дрон піднімається високо (до 8м) для ракурсу «зверху-вниз»
            # і опускається нижче (до 2.5м), щоб зазирнути крізь двері
            cam_y = 4.5 + 3.0 * math.cos(angle)
            
            cam_x = self.center[0] + current_radius * math.cos(angle)
            cam_z = self.center[2] + current_radius * math.sin(angle)
            cam_pos = np.array([cam_x, cam_y, cam_z])

            # 2. Динамічна точка фокусу (камера не просто дивиться в одну точку, а сканує сцену)
            look_target = self.center.copy()
            look_target[0] += 1.5 * math.sin(3 * angle)
            look_target[2] += 1.5 * math.cos(3 * angle)
            
            # Коли дрон високо, він фокусується нижче, щоб бачити підлогу всередині кімнат
            if cam_y > 5.0:
                look_target[1] = 0.2 
            else:
                look_target[1] = 1.3

            # Розрахунок матриці орієнтації камери
            forward = look_target - cam_pos
            forward /= np.linalg.norm(forward)
            
            # Стабільний вектор горизонту з урахуванням нахилів
            right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
            right /= np.linalg.norm(right)
            
            up = np.cross(forward, right)

            R = np.vstack([right, up, forward])
            T = -R @ cam_pos

            # Рендер
            frame = self.render_point_cloud_advanced(R, T)
            img_name = f"frame_{i:04d}.jpg"
            Image.fromarray(frame).save(os.path.join(self.images_dir, img_name))

            qvec = rmat_to_qvec(R)
            images_bin_data.append((i + 1, qvec, T, 1, img_name))

        # Запис COLMAP структур
        with open(os.path.join(self.sparse_dir, "cameras.bin"), "wb") as f:
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<iiQQ", 1, 1, self.W, self.H))
            f.write(struct.pack("<dddd", self.fx, self.fy, self.cx, self.cy))

        with open(os.path.join(self.sparse_dir, "images.bin"), "wb") as f:
            f.write(struct.pack("<Q", len(images_bin_data)))
            for img_id, qvec, tvec, cam_id, name in images_bin_data:
                f.write(struct.pack("<i", img_id))
                f.write(struct.pack("<dddd", *qvec))
                f.write(struct.pack("<ddd", *tvec))
                f.write(struct.pack("<i", cam_id))
                f.write(name.encode('utf-8') + b"\x00")
                f.write(struct.pack("<Q", 0))

        print("✨ Роботу завершено. Нові кадри згенеровано зі складною 3D-динамікою польоту!")

if __name__ == "__main__":
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CLOUD_PT = os.path.join(PROJECT_DIR, "data", "cloud_splat.pt")
    COLMAP_OUT = os.path.join(PROJECT_DIR, "data", "colmap_output")
    
    # Використовуємо uint16 для безпечних розрахунків блендингу кольорів
    uint16 = np.uint16
    
    renderer = TacticalOrbitRenderer(CLOUD_PT, COLMAP_OUT)
    renderer.generate_complex_trajectory(num_frames=150)