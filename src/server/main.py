import os
import time
import json
from pathlib import Path
import torch
from fastapi import FastAPI, Response, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sys

app = FastAPI(title="AeroSplat-GIS Tactical Server v0.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SPLAT_PATH = BASE_DIR / "data" / "cloud_splat.pt"
LAYOUT_PATH = SPLAT_PATH.parent / "layout.json"
WEB_DIR = BASE_DIR / "web"

# ВИПРАВЛЕНО: імпорт генератора піднято на рівень модуля замість того, щоб
# робити sys.path.append() + import всередині кожного POST-запиту. Раніше це
# було крихко під uvicorn --reload (можливе неузгоджене кешування модуля)
# і просто витрачало час на кожен виклик.
src_dir = str(BASE_DIR / "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from synthetic_generator import AnalyticalSplatGenerator

# ── 1. ЕНДПОІНТ ЧИТАННЯ (Тільки 3DGS) ───────────────────────────────
@app.get("/api/v1/spatial-chunk")
async def get_spatial_chunk():
    start_time = time.time()

    if not SPLAT_PATH.exists():
        return Response(content=f"Error: Splat file not found", status_code=404)
        
    data = torch.load(str(SPLAT_PATH), map_location="cpu", weights_only=False)
    xyz = data["xyz"].float()
    scale = data["scale"].float()
    rotation = data["rotation"].float()
    opacity = data["opacity"].float().unsqueeze(-1)
    rgb = data["rgb"].float()
    count = xyz.shape[0]

    # Пакуємо: 14 float32
    flat_data = torch.cat([xyz, scale, rotation, opacity, rgb], dim=-1)
    binary_buffer = flat_data.numpy().tobytes()

    processing_time = round((time.time() - start_time) * 1000, 1)

    headers = {
        "X-Processing-Time-Ms": str(processing_time),
        "X-Gaussians-Count": str(count),
        "Access-Control-Expose-Headers": "X-Processing-Time-Ms, X-Gaussians-Count"
    }

    return Response(content=binary_buffer, media_type="application/octet-stream", headers=headers)


# ── 2. ЕНДПОІНТ ЧИТАННЯ СХЕМИ ────────────────────────────────────────
# ВИПРАВЛЕНО: раніше не було способу віддати редактору останню збережену
# схему, тож перезавантаження сторінки завжди скидало все до одної кімнати.
@app.get("/api/v1/layout")
async def get_layout():
    if not LAYOUT_PATH.exists():
        return Response(content=json.dumps({"error": "No saved layout yet"}), status_code=404, media_type="application/json")

    with open(str(LAYOUT_PATH), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


# ── 3. ЕНДПОІНТ ЗАПИСУ (Винесений наружу, СИНХРОННИЙ!) ───────────────
@app.post("/api/v1/layout")
def save_and_generate_layout(payload: dict = Body(...)):
    """
    Приймає схему з веб-конструктора, запускає генератор в окремому потоці.
    """
    try:
        # Зберігаємо схему
        with open(str(LAYOUT_PATH), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            
        print(f"📥 Нову схему збережено. Запускаю генерацію...")

        # Перезбірка сцени (генератор вже імпортований на рівні модуля)
        generator = AnalyticalSplatGenerator(str(LAYOUT_PATH))
        generator.save_splat(str(SPLAT_PATH))
        
        return {"status": "success", "message": "Layout compiled and 3DGS model updated analytically."}
        
    except Exception as e:
        print(f"❌ Помилка автогенерації: {str(e)}")
        return Response(content=json.dumps({"error": str(e)}), status_code=500, media_type="application/json")


# Монтування статики
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)