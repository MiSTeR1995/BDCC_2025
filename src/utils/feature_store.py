# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple

import torch


# ---------------------------
# 1) Ключ кэша (fingerprint + препроцесс)
# ---------------------------

@dataclass(frozen=True)
class CacheKey:
    """
    Паспорт для конкретной модальности/экстрактора/препроцесса.
    Меняешь любой важный параметр — ключ меняется → файл считается заново.
    """
    mod: str                     # "face"|"audio"|"text"|"behavior"
    extractor_fp: str            # extractor.fingerprint(), напр. "clapa:laion/clap-htsat-fused:pooled"
    avg: str                     # average_features: "mean"|"mean_std"|"raw"
    frames: int                  # counter_need_frames (для видео/face)
    img: int                     # image_size (для видео/face)
    text_col: Optional[str]      # имя текстовой колонки (для behavior)
    pre_v: str = "v1"            # версия логики препроцесса (get_metadata и т.п.)

    def short_id(self) -> str:
        """
        Читаемый идентификатор для путей/папок.
        Пример: face__clipv-openai-clip-vit-base-patch32__frames30_img224_avg-mean_std_pv-v1
        """
        def _sanitize(s: str) -> str:
            bad = r'\/:*?"<>|'
            for ch in bad:
                s = s.replace(ch, '-')
            s = s.replace(' ', '-')       # пробелы → '-'
            while '--' in s:
                s = s.replace('--', '-')  # схлопываем
            return s

        parts = [self.mod, self.extractor_fp]
        if self.mod == "face":
            parts.append(f"frames{self.frames}")
            parts.append(f"img{self.img}")
        if self.mod == "behavior" and self.text_col:
            parts.append(f"col-{self.text_col}")
        parts.append(f"avg-{self.avg}")
        parts.append(f"pv-{self.pre_v}")

        human = '__'.join(_sanitize(str(p)) for p in parts if p is not None)
        # Windows любит лимиты по длине путей — чуть подрежем
        return human[:144]


def build_cache_key(mod: str, extractor: Any, cfg: Any) -> CacheKey:
    """
    Сборка CacheKey из конфига и экстрактора.
    extractor должен иметь .fingerprint(); если нет — используем имя класса как fallback.
    Параметры препроцесса учитываем ТОЛЬКО там, где они релевантны модальности.
    """
    # fingerprint экстрактора или безопасный фолбэк
    fp_fn = getattr(extractor, "fingerprint", None)
    extractor_fp = fp_fn() if callable(fp_fn) else type(extractor).__name__

    # общие поля
    avg_raw = getattr(cfg, "average_features", "mean_std")
    avg = str(avg_raw).strip().lower()  # "mean"|"mean_std"|"raw"
    pre_v = str(getattr(cfg, "preprocess_version", "v1"))

    # модально-зависимые поля
    if mod == "face":
        frames = int(getattr(cfg, "counter_need_frames", 30))
        img    = int(getattr(cfg, "image_size", 224))
        text_col = None
    elif mod == "behavior":
        frames = 0
        img    = 0
        tc = getattr(cfg, "text_description_column", None)
        text_col = str(tc) if (tc is not None and not isinstance(tc, str)) else tc
    else:  # audio, text, прочее
        frames = 0
        img    = 0
        text_col = None

    return CacheKey(
        mod=mod,
        extractor_fp=extractor_fp,
        avg=avg,
        frames=frames,
        img=img,
        text_col=text_col,
        pre_v=pre_v,
    )


# ---------------------------
# 2) Фичи: упаковка/распаковка на диск
# ---------------------------

def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def _atomic_save_pt(obj: Any, path: str):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _atomic_save_pickle(obj: Any, path: str):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class FeatureStore:
    """
    Дисковый кэш по модальностям.
    Хранит:
      - meta: список сэмплов (без признаков)
      - feats: {sample_name -> dict(core_key -> Tensor)|None} c заголовком header=CacheKey
    API:
      - load_meta/save_meta
      - load_modality_store/save_modality_store
      - get_store (ленивый in-memory кэш)
    """
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._stores_mem: Dict[Tuple[str, str, str, int, int, str, str], Dict[str, Optional[dict]]] = {}
        # mem_key: (dataset, split, mod, seed, subset, avg, short_id)

    # ------ пути
    def _base_dir(self, dataset: str, split: str) -> str:
        return os.path.join(self.root, dataset, split)

    def meta_path(self, dataset: str, split: str, seed: int, subset: int) -> str:
        base = self._base_dir(dataset, split)
        _safe_makedirs(base)
        return os.path.join(base, f"meta_seed{seed}_subset{subset}.pickle")

    def feats_path(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> str:
        base = self._base_dir(dataset, split)
        mod_dir = os.path.join(base, key.mod, key.short_id())  # читабельная подпапка
        _safe_makedirs(mod_dir)
        fname = f"feats_seed{seed}_subset{subset}_avg-{key.avg}.pt"
        return os.path.join(mod_dir, fname)

    # ------ meta
    def load_meta(self, dataset: str, split: str, seed: int, subset: int) -> list[dict]:
        p = self.meta_path(dataset, split, seed, subset)
        if not os.path.exists(p):
            return []
        with open(p, "rb") as f:
            return pickle.load(f)

    def save_meta(self, dataset: str, split: str, seed: int, subset: int, meta: list[dict]):
        p = self.meta_path(dataset, split, seed, subset)
        _atomic_save_pickle(meta, p)

    # ------ модальность
    def load_modality_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> Tuple[Dict[str, Optional[dict]], Optional[CacheKey]]:
        p = self.feats_path(dataset, split, key, seed, subset)
        if not os.path.exists(p):
            return {}, None
        obj = torch.load(p, map_location="cpu")
        data = obj.get("data", {}) if isinstance(obj, dict) else obj  # обратная совместимость
        header = obj.get("header", None)
        if isinstance(header, dict):
            header = CacheKey(**header)
        return data, header

    def save_modality_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int, store: Dict[str, Optional[dict]]):
        p = self.feats_path(dataset, split, key, seed, subset)
        payload = {"header": asdict(key), "data": store}
        _atomic_save_pt(payload, p)

    # ------ ленивый in-memory доступ для __getitem__
    def get_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> Dict[str, Optional[dict]]:
        mem_key = (dataset, split, key.mod, seed, subset, key.avg, key.short_id())
        if mem_key in self._stores_mem:
            return self._stores_mem[mem_key]
        store, _ = self.load_modality_store(dataset, split, key, seed, subset)
        self._stores_mem[mem_key] = store
        return store


# ---------------------------
# 3) Вспомогалки
# ---------------------------

def need_full_reextract(cfg: Any, mod: str, old_header: Optional[CacheKey], new_key: CacheKey) -> bool:
    """
    Если ключи отличаются — пересчитываем заново текущий таргет-файл.
    Плюс поддержка ручного форса через конфиг.
    Ожидается, что cfg.overwrite_modality_cache: bool, cfg.force_reextract: list[str]
    """
    if getattr(cfg, "overwrite_modality_cache", False):
        return True
    force_list = set(cfg.force_reextract)
    if mod in force_list:
        return True
    return (old_header is None) or (old_header != new_key)


def merge_missing(store: Dict[str, Optional[dict]], sample_names: list[str]) -> list[str]:
    """Список имён, которых нет в store — их надо доизвлечь."""
    return [s for s in sample_names if s not in store]
