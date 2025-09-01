# coding: utf-8
from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
import torch, torch.nn as nn
from transformers import (
    CLIPModel, CLIPProcessor,
    ViTModel, AutoImageProcessor
)


def _ensure_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    d = (device or "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")


class ClipVideoExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda", output_mode: str = "seq"):
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode  # "seq" | "pooled"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  = CLIPProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"clipv:{self.model_name}"

    @torch.no_grad()
    def extract(self, *, pixel_values: torch.Tensor | None = None,
                      face_tensor: torch.Tensor | None = None, **_) -> Dict[str, torch.Tensor]:
        if pixel_values is None and face_tensor is not None:
            # fallback: сырые кадры -> один батч через процессор
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)
            imgs_cpu = [img.cpu() for img in face_tensor]
            pixel_values = self.proc(images=imgs_cpu, return_tensors="pt")["pixel_values"].to(self.device)

        if pixel_values is None:
            # пусто — вернём корректной формы нулевой тензор
            if self.output_mode == "seq":
                # оценим размер скрытого пространства
                D = self.model.visual_projection.out_features
                return {"embedding": torch.empty((0, D), device=self.device)}
            else:
                D = self.model.visual_projection.out_features
                return {"embedding": torch.empty((0, D), device=self.device)}

        pv = pixel_values.to(self.device)  # [T,3,H,W]
        if self.output_mode == "pooled":
            emb = self.model.get_image_features(pixel_values=pv)   # [T,D]
            return {"embedding": emb}

        # seq-mode: берём скрытые состояния vision_model и убираем CLS-агрегацию
        # out.last_hidden_state: [T, L, D], где L = 1 + N_patches (включая CLS)
        out = self.model.vision_model(pixel_values=pv, output_hidden_states=False, return_dict=True)
        seq = out.last_hidden_state  # [T, L, D]
        # выбрасываем CLS-токен (позиция 0) — оставляем только патчи
        if seq.size(1) > 1:
            seq = seq[:, 1:, :]  # [T, L-1, D]
        # склеиваем по времени и патчам → единая последовательность
        seq = seq.flatten(0, 1).contiguous()  # [T*(L-1), D]
        return {"embedding": seq}


class VitVideoExtractor:
    """Видео → ViT → последовательность патч-токенов, конкатенированная по времени."""
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = "cuda"):
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  = AutoImageProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"vitv:{self.model_name}:seq"

    @torch.no_grad()
    def extract(self, *, pixel_values: torch.Tensor | None = None, **_) -> Dict[str, torch.Tensor]:
        if pixel_values is None:
            return {"embedding": torch.empty((0, self.model.config.hidden_size), device=self.device)}
        pv = pixel_values.to(self.device)  # [T,3,H,W]
        # ViTModel ожидает уже нормализованные тензоры; предполагаем, что нам подали
        # выход AutoImageProcessor (см. video_preprocessor body-ROI → processor → pixel_values)
        out = self.model(pixel_values=pv, output_hidden_states=False, return_dict=True)
        seq = out.last_hidden_state  # [T, L, D] (включая CLS)
        if seq.size(1) > 1:
            seq = seq[:, 1:, :]  # выброс CLS
        seq = seq.flatten(0, 1).contiguous()  # [T*(L-1), D]
        return {"embedding": seq}


def build_extractors_from_config(cfg) -> Dict[str, Any]:
    device = cfg.device
    sample_rate = int(getattr(cfg, "sample_rate", 48000))

    # Для данных-требований: всегда последовательности
    output_mode = "seq"

    ex: Dict[str, Any] = {}

    vid_model: str = cfg.video_extractor
    if isinstance(vid_model, str) and vid_model.lower() != "off":
        v = vid_model.lower()
        if "clip" in v:
            ex["body"] = ClipVideoExtractor(model_name=vid_model, device=device, output_mode=output_mode)
        elif "vit" in v:
            ex["body"] = VitVideoExtractor(model_name=vid_model, device=device)
        else:
            raise ValueError(f"Video extractor '{vid_model}' не поддерживается (ожидается CLIP/VIT).")

    # audio/text/behavior отключаем для этой задачи
    return ex
