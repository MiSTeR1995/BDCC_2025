# video_preprocessor.py
# coding: utf-8
from __future__ import annotations
import os, cv2, torch, numpy as np
from typing import Optional, Tuple, Sequence
from ultralytics import YOLO

# ── ленивая инициализация YOLO ────────────────────────────────────
_YOLO: Optional[YOLO] = None
def _lazy_yolo(weights_path: str) -> YOLO:
    global _YOLO
    if _YOLO is None:
        _YOLO = YOLO(weights_path)
    return _YOLO

# ── утилиты ───────────────────────────────────────────────────────
def select_uniform_frames(frames: Sequence[int], N: int) -> list[int]:
    N = int(N)
    if N <= 0 or len(frames) <= N:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, num=N, dtype=int)
    return [frames[i] for i in idx]

def _to_pixel_values(image_rgb: np.ndarray, image_processor, device: str) -> Optional[torch.Tensor]:
    if image_rgb is None or image_rgb.size == 0 or image_rgb.ndim != 3:
        return None
    inputs = image_processor(images=image_rgb, return_tensors="pt")
    pv = inputs["pixel_values"]
    return pv.to(device) if isinstance(pv, torch.Tensor) else pv

# ── основной экстрактор тела ──────────────────────────────────────
@torch.no_grad()
def get_body_pixel_values(
    video_path: str,
    segment_length: int,
    image_processor,                # CLIPProcessor или AutoImageProcessor
    *,
    device: str = "cuda",
    yolo_weights: str = "modalities/video/checkpoints/body/best_YOLO.pt",
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Возвращает: (video_name, body_pixel_values [T,3,H,W] | None)
    Логика: на выбранных кадрах ищем самое крупное тело через YOLO; если не нашли — берём весь кадр.
    """
    model = _lazy_yolo(yolo_weights)

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    need = set(select_uniform_frames(range(total_frames), segment_length))

    batches = []
    t = 0
    while True:
        ok, im0 = cap.read()
        if not ok:
            break
        if t in need:
            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # детект без трекинга (меньше предупреждений и зависимостей)
            results = model.predict(
                im_rgb, imgsz=640, conf=0.15, iou=0.5,
                device=0 if str(device).startswith("cuda") else None,
                verbose=False
            )

            pv = None
            if results and len(results[0].boxes):
                # берём самое крупное тело (по площади bbox)
                boxes = results[0].boxes
                largest = max(
                    boxes,
                    key=lambda b: float((b.xyxy[0, 2] - b.xyxy[0, 0]) * (b.xyxy[0, 3] - b.xyxy[0, 1]))
                )
                x1, y1, x2, y2 = largest.xyxy[0].int().cpu().tolist()
                x1 = max(x1, 0); y1 = max(y1, 0)
                x2 = min(x2, im_rgb.shape[1]); y2 = min(y2, im_rgb.shape[0])
                if x2 > x1 and y2 > y1:
                    roi = im_rgb[y1:y2, x1:x2]
                    pv = _to_pixel_values(roi, image_processor, device)

            # фолбэк — весь кадр, если тел не нашли или roi пустой
            if pv is None:
                pv = _to_pixel_values(im_rgb, image_processor, device)

            if pv is not None:
                batches.append(pv)  # [1,3,H,W]
        t += 1

    cap.release()
    body_tensor = torch.cat(batches, dim=0) if batches else None  # [T,3,H,W] или None
    return video_name, body_tensor

# ── совместимая обёртка под старое имя ────────────────────────────
@torch.no_grad()
def get_metadata(
    video_path: str,
    segment_length: int,
    image_processor,
    device: str = "cuda",
    yolo_weights: str = "./src/data_loading/best_YOLO.pt",
) -> Tuple[str, Optional[torch.Tensor]]:
    return get_body_pixel_values(
        video_path=video_path,
        segment_length=segment_length,
        image_processor=image_processor,
        device=device,
        yolo_weights=yolo_weights,
    )
