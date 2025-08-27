# video_preprocessor.py
# coding: utf-8
"""
WSM: извлечение body-ROI из видео.
- Детектор: YOLO (ultralytics) для тела.
- Связка с лицом (MediaPipe Face): если найдены лица, ищем body-box, содержащий центр лица.
- Если лиц нет: берём самое крупное тело.
- Фолбэк: весь кадр.
Возвращаем батч pixel_values [T,3,H,W] под поданный image_processor (CLIPProcessor или AutoImageProcessor).
"""

from __future__ import annotations

import os
import cv2
import torch
import numpy as np
from typing import Optional, Tuple, Sequence
from ultralytics import YOLO
import mediapipe as mp


# ───────── Mediapipe Face (для привязки лицо→тело) ───────── #
mp_face_detection = mp.solutions.face_detection
_face_det = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.6
)

# ───────── YOLO (ленивая инициализация) ───────── #
_YOLO_MODEL: Optional[YOLO] = None


def _lazy_yolo(weights_path: str) -> YOLO:
    """Ленивая загрузка модели YOLO один раз на процесс."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(weights_path)
    return _YOLO_MODEL


# ───────── утилиты ───────── #
def select_uniform_frames(frames: Sequence[int], N: int) -> list[int]:
    """Возвращает N равномерно распределённых индексов из frames."""
    N = int(N)
    if N <= 0 or len(frames) <= N:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, num=N, dtype=int)
    return [frames[i] for i in idx]


def _to_pixel_values(image_rgb: np.ndarray, image_processor, device: str) -> Optional[torch.Tensor]:
    """
    RGB (H,W,3) uint8 → processor(...)->pixel_values [1,3,H,W] на нужном девайсе.
    image_processor: CLIPProcessor или AutoImageProcessor.
    """
    if image_rgb is None or image_rgb.size == 0 or image_rgb.ndim != 3:
        return None
    inputs = image_processor(images=image_rgb, return_tensors="pt")
    pv = inputs["pixel_values"]
    if isinstance(pv, torch.Tensor):
        pv = pv.to(device)
    return pv  # [1,3,H,W]


# ───────── основной экстрактор ───────── #
@torch.no_grad()
def get_body_pixel_values(
    video_path: str,
    segment_length: int,
    image_processor,            # CLIPProcessor или AutoImageProcessor
    *,
    device: str = "cuda",
    yolo_weights: str = "./src/data_loading/best_YOLO.pt",
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Возвращает:
        video_name (str),
        body_tensor [T,3,H,W] или None
    Логика:
      1) Детектим лица (MP). Для каждого лица ищем body-box, содержащий центр лица.
      2) Если лиц нет, берём самое крупное тело из YOLO.
      3) Если тел нет — фолбэк: весь кадр.
    """
    model = _lazy_yolo(yolo_weights)

    # Сброс трекера YOLO между видео (если есть)
    if hasattr(model.predictor, "trackers") and model.predictor.trackers:
        try:
            model.predictor.trackers[0].reset()
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    need_frames = select_uniform_frames(list(range(total_frames)), int(segment_length))

    body_batches = []
    t = 0
    while True:
        ret, im0 = cap.read()
        if not ret:
            break
        if t in need_frames:
            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # YOLO тела
            body_results = model.track(
                im_rgb,
                persist=True,
                imgsz=640,
                conf=0.01,
                iou=0.5,
                augment=False,
                device=0 if str(device).startswith("cuda") else None,
                verbose=False,
            )

            pv = None

            # 1) есть лица — подберём body-box, содержащий центр лица
            face_res = _face_det.process(im_rgb)
            if face_res and face_res.detections:
                h, w = im_rgb.shape[:2]

                def _bbox_from_det(det):
                    bb = det.location_data.relative_bounding_box
                    x1 = max(int(bb.xmin * w), 0)
                    y1 = max(int(bb.ymin * h), 0)
                    x2 = min(int((bb.xmin + bb.width) * w), w)
                    y2 = min(int((bb.ymin + bb.height) * h), h)
                    return x1, y1, x2, y2

                # берём первое валидное совпадение лицо→тело
                for det in face_res.detections:
                    x1, y1, x2, y2 = _bbox_from_det(det)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    body_bbox = None

                    if body_results and len(body_results[0].boxes):
                        for box in body_results[0].boxes:
                            bx = box.xyxy.int().cpu().numpy()[0]  # [x1,y1,x2,y2]
                            if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]:
                                body_bbox = bx
                                break

                    if body_bbox is not None:
                        bx = body_bbox
                        roi = im_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                        if roi.size:
                            pv = _to_pixel_values(roi, image_processor, device)
                            break  # достаточно одного совпадения

            # 2) лиц нет — возьмём самое крупное тело
            if pv is None and body_results and len(body_results[0].boxes):
                largest = max(
                    body_results[0].boxes,
                    key=lambda b: (b.xyxy[0, 2] - b.xyxy[0, 0]) * (b.xyxy[0, 3] - b.xyxy[0, 1]),
                )
                bx = largest.xyxy.int().cpu().numpy()[0]
                roi = im_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                if roi.size:
                    pv = _to_pixel_values(roi, image_processor, device)

            # 3) фолбэк — весь кадр
            if pv is None:
                pv = _to_pixel_values(im_rgb, image_processor, device)

            if pv is not None:
                body_batches.append(pv)

        t += 1

    cap.release()

    body_tensor = torch.cat(body_batches, dim=0) if body_batches else None  # [T,3,H,W]
    return video_name, body_tensor


# ───────── тонкая совместимость со старым кодом ───────── #
@torch.no_grad()
def get_metadata(
    video_path: str,
    segment_length: int,
    image_processor,
    device: str = "cuda",
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Совместимая обёртка под старое имя.
    Возвращает: (video_name, body_pixel_values [T,3,H,W] | None)
    """
    return get_body_pixel_values(
        video_path=video_path,
        segment_length=segment_length,
        image_processor=image_processor,
        device=device,
    )
