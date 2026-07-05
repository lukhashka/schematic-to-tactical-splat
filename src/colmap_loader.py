import os
import struct
import math
import random
import time
import numpy as np
import torch
from torch import Tensor
from PIL import Image
from typing import List, NamedTuple, Optional

# ═══════════════════════════════════════════════════════════════════
#  COLMAP BINARY READERS
# ═══════════════════════════════════════════════════════════════════

class ColmapCamera(NamedTuple):
    id:         int
    model:      str
    width:      int
    height:     int
    params:     np.ndarray


class ColmapImage(NamedTuple):
    id:          int
    qvec:        np.ndarray
    tvec:        np.ndarray
    camera_id:   int
    name:        str


CAMERA_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE":        4,
    "SIMPLE_RADIAL":  4,
    "RADIAL":         5,
    "OPENCV":         8,
    "FULL_OPENCV":    12,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
}

CAMERA_MODEL_IDS = {
    0:  "SIMPLE_PINHOLE",
    1:  "PINHOLE",
    2:  "SIMPLE_RADIAL",
    3:  "RADIAL",
    4:  "OPENCV",
    5:  "FULL_OPENCV",
    6:  "SIMPLE_RADIAL_FISHEYE",
    7:  "RADIAL_FISHEYE",
}


def read_cameras_binary(path: str) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id    = struct.unpack("<i", f.read(4))[0]
            model_id  = struct.unpack("<i", f.read(4))[0]
            width     = struct.unpack("<Q", f.read(8))[0]
            height    = struct.unpack("<Q", f.read(8))[0]

            model_name  = CAMERA_MODEL_IDS.get(model_id, "PINHOLE")
            num_params  = CAMERA_MODEL_PARAMS.get(model_name, 4)
            params      = np.frombuffer(f.read(8 * num_params), dtype=np.float64)

            cameras[cam_id] = ColmapCamera(
                id=cam_id, model=model_name,
                width=int(width), height=int(height),
                params=params,
            )
    return cameras


def read_images_binary(path: str) -> dict:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id  = struct.unpack("<i", f.read(4))[0]
            qvec      = np.frombuffer(f.read(32), dtype=np.float64)
            tvec      = np.frombuffer(f.read(24), dtype=np.float64)
            camera_id = struct.unpack("<i", f.read(4))[0]

            name = b""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name += char
            name = name.decode("utf-8")

            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2d)

            images[image_id] = ColmapImage(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
            )
    return images


# ═══════════════════════════════════════════════════════════════════
#  CAMERA POSE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════

def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),          1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),          2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])


def build_view_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3,  3] = t
    return view


def extract_intrinsics(cam: ColmapCamera):
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    elif cam.model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])


# ═══════════════════════════════════════════════════════════════════
#  CAMERA DATACLASS
# ═══════════════════════════════════════════════════════════════════

class Camera:
    def __init__(
        self,
        image_path: str,
        R: np.ndarray,
        T: np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        width: int, height: int,
        device: torch.device,
        downscale: float = 1.0,
    ):
        if downscale != 1.0:
            width  = int(width  * downscale)
            height = int(height * downscale)
            fx *= downscale;  fy *= downscale
            cx *= downscale;  cy *= downscale

        self.R = R
        self.T = T
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.image_width  = width
        self.image_height = height
        self.image_path   = image_path
        self.device       = device

        self.FoVx = 2 * math.atan(width  / (2 * fx))
        self.FoVy = 2 * math.atan(height / (2 * fy))

        self.view_matrix = torch.tensor(
            build_view_matrix(R, T), dtype=torch.float32, device=device
        )
        self.gt_image = self._load_image(image_path, width, height, device)

    def _load_image(self, path: str, W: int, H: int, device: torch.device) -> Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((W, H), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr, device=device).permute(2, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  COLMAP DATA LOADER
# ═══════════════════════════════════════════════════════════════════

class ColmapDataLoader:
    def __init__(self, colmap_root: str, device: torch.device):
        self.colmap_root = colmap_root
        self.device      = device

        # Підтримка гнучких шляхів для репозиторію
        sparse_candidates = [
            os.path.join(colmap_root, "sparse", "0"),
            os.path.join(colmap_root, "sparse"),
        ]
        self.sparse_dir = None
        for path in sparse_candidates:
            if os.path.isdir(path):
                self.sparse_dir = path
                break

        if self.sparse_dir is None:
            raise FileNotFoundError(f"Не знайдено sparse/ в {colmap_root}")

        # Додано підтримку як 'dense/images', так і просто 'images' з нашої орбіти
        images_candidates = [
            os.path.join(colmap_root, "dense", "images"),
            os.path.join(colmap_root, "images"),
        ]
        self.images_dir = None
        for path in images_candidates:
            if os.path.isdir(path):
                self.images_dir = path
                break
                
        if self.images_dir is None:
            raise FileNotFoundError(f"Не знайдено папки зображень в {colmap_root}")

    def load(self, downscale: float = 1.0, max_cameras: Optional[int] = None) -> List[Camera]:
        cameras_bin = os.path.join(self.sparse_dir, "cameras.bin")
        images_bin  = os.path.join(self.sparse_dir, "images.bin")

        colmap_cameras = read_cameras_binary(cameras_bin)
        colmap_images = read_images_binary(images_bin)

        sorted_images = sorted(colmap_images.values(), key=lambda x: x.name)
        if max_cameras is not None:
            sorted_images = sorted_images[:max_cameras]

        cameras = []
        for col_img in sorted_images:
            img_path = os.path.join(self.images_dir, col_img.name)
            if not os.path.exists(img_path):
                continue

            col_cam = colmap_cameras[col_img.camera_id]
            fx, fy, cx, cy = extract_intrinsics(col_cam)
            R = qvec_to_rotation_matrix(col_img.qvec)
            T = col_img.tvec.astype(np.float32)

            cam = Camera(
                image_path=img_path, R=R, T=T,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=col_cam.width, height=col_cam.height,
                device=self.device, downscale=downscale,
            )
            cameras.append(cam)

        print(f"  Завантажено {len(cameras)} камер, розмір: {cameras[0].image_width}×{cameras[0].image_height}")
        return cameras


# ═══════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════

def train_with_real_data(
    colmap_root:    str,
    pt_path:        str,
    output_path:    str,
    num_iterations: int   = 7000,
    downscale:      float = 1.0,  # Для нашої синтетики 800x600 залишаємо 1.0
    use_cuda:       bool  = False,
    max_points:     int   = 100000,
):
    from gs_trainer import (
        GaussianModel, PureTorchRasterizer,
        render_with_cuda_rasterizer,
        photometric_loss, GaussianDensifier, create_optimizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  3DGS Training: Пайплайн 'Schematic-to-Tactical-Splat'\n{'='*60}\n")

    # 1. Завантаження камер
    loader  = ColmapDataLoader(colmap_root, device)
    cameras = loader.load(downscale=downscale)

    H, W = cameras[0].image_height, cameras[0].image_width
    fx, fy, cx, cy = cameras[0].fx, cameras[0].fy, cameras[0].cx, cameras[0].cy

    # 2. Завантаження хмари точок
    if os.path.exists(pt_path):
        data = torch.load(pt_path, map_location=device, weights_only=False)
        if data["xyz"].shape[0] > max_points:
            idx = torch.randperm(data["xyz"].shape[0])[:max_points]
            data["xyz"] = data["xyz"][idx]
            data["rgb"] = data["rgb"][idx]
        print(f"  Завантажено {data['xyz'].shape[0]:,} точок")
    else:
        raise FileNotFoundError(f"Не знайдено початкову хмару точок: {pt_path}")

    # 3. Нормалізація сцени (Центрування хмари)
    xyz = data["xyz"].to(device)
    xyz_mean = xyz.mean(dim=0)
    xyz_std  = xyz.std().item()
    data["xyz"] = (xyz - xyz_mean) / xyz_std

    # ✨ ФІКС: Математично правильна нормалізація камер ОДИН РАЗ до циклу
    xyz_mean_np = xyz_mean.cpu().numpy()
    for camera in cameras:
        t_norm = (camera.T + camera.R @ xyz_mean_np) / xyz_std
        camera.T = t_norm.astype(np.float32)
        camera.view_matrix = torch.tensor(
            build_view_matrix(camera.R, camera.T),
            dtype=torch.float32, device=device
        )

    # 4. Ініціалізація моделі
    model     = GaussianModel(data["xyz"], data["rgb"]).to(device)
    optimizer = create_optimizer(model)
    densifier = GaussianDensifier(model, grad_threshold=0.0002)

    rasterizer = PureTorchRasterizer(H, W, fx, fy, cx, cy)
    bg_color   = torch.zeros(3, device=device)

    print(f"\n  Старт тренування | Ітерацій: {num_iterations} | Гаусіанів: {model._xyz.shape[0]:,}\n")
    losses = []

    for iteration in range(1, num_iterations + 1):
        t0 = time.time()
        
        # Випадковий вибір камери — тепер її матриця стабільна і не псується
        camera = random.choice(cameras)

        optimizer.zero_grad()
        params = model.get_params()

        if use_cuda:
            out = render_with_cuda_rasterizer(dict(params), camera, bg_color)
            rendered = out["render"]
            gt       = camera.gt_image
        else:
            rendered = rasterizer.render(params, camera.view_matrix, bg_color)
            rendered = rendered.permute(2, 0, 1)
            gt       = camera.gt_image

        loss = photometric_loss(rendered, gt)
        loss.backward()
        
        # Обмеження розростання Гаусіанів для стабільності пам'яті 8GB
        densifier.update_stats()
        optimizer.step()

        losses.append(loss.item())
        elapsed = (time.time() - t0) * 1000

        # Денсифікація кожні 100 кроків (ліміт затиснуто до 300к для уникнення OOM)
        if iteration % 100 == 0 and iteration < int(num_iterations * 0.75):
            densifier.densify_and_prune(optimizer, max_gaussians=300000)

        if iteration % 50 == 0 or iteration == 1:
            print(f"  [{iteration:>5}/{num_iterations}] Loss: {loss.item():.4f} | "
                  f"Гаусіанів: {model._xyz.shape[0]:,} | Камера: {os.path.basename(camera.image_path)} | {elapsed:.0f}ms")

    # Збереження
    final = model.get_params()
    torch.save({k: v.detach().cpu() for k, v in final.items() if k != "cov3d"}, output_path)
    print(f"\n  ✅ Навчання завершено! Модель збережено в: {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap",     default="data/colmap_output")
    parser.add_argument("--input",      default="data/cloud.pt")
    parser.add_argument("--output",     default="data/cloud_splat.pt")
    parser.add_argument("--iterations", type=int,   default=7000)
    parser.add_argument("--cuda",       action="store_true")
    args = parser.parse_args()

    # Залізобетонна робота зі шляхами через pathlib
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    
    train_with_real_data(
        colmap_root    = str(PROJECT_DIR / args.colmap),
        pt_path        = str(PROJECT_DIR / args.input),
        output_path    = str(PROJECT_DIR / args.output),
        num_iterations = args.iterations,
        use_cuda       = args.cuda,
    )