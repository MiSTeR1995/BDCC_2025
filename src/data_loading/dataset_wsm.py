from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from .video_preprocessor import get_body_pixel_values
from src.utils.feature_store import FeatureStore, build_cache_key, need_full_reextract, merge_missing


class WSMBodyDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        config,
        split: str,
        modality_processors: Dict[str, Any],        # должен содержать 'body'
        modality_feature_extractors: Dict[str, Any],# должен содержать 'body'
        dataset_name: str = "wsm",
        device: str = "cuda",
    ) -> None:
        super().__init__()

        # базовые поля
        self.csv_path   = csv_path
        self.video_dir  = video_dir
        self.config     = config
        self.split      = split
        self.dataset_name = dataset_name
        self.device     = device

        # параметры извлечения
        self.segment_length   = int(getattr(config, "segment_length", 16))
        self.subset_size      = int(getattr(config, "subset_size", 0) or 0)
        self.average_features = str(getattr(config, "average_features", "raw"))  # 'raw'|'mean'|'mean_std'
        self.yolo_weights     = str(getattr(config, "src/data_loading/best_YOLO.pt"))

        # процессоры/экстракторы (только body)
        self.proc = modality_processors.get("body", None)
        self.extr = modality_feature_extractors.get("body", None)
        if self.proc is None:
            raise ValueError("Нужен image processor для 'body' (CLIPProcessor/AutoImageProcessor).")
        if self.extr is None:
            raise ValueError("Нужен extractor для 'body' (CLIP/VIT).")

        # кэш
        self.save_prepared_data = bool(getattr(config, "save_prepared_data", True))
        self.save_feature_path  = str(getattr(config, "save_feature_path", "features_cache"))
        self.store = FeatureStore(self.save_feature_path)

        # CSV
        df = pd.read_csv(self.csv_path)
        if "video_name" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV должен содержать столбцы 'video_name' и 'label'.")
        if self.subset_size > 0:
            df = df.head(self.subset_size)
            logging.info(f"[WSMBodyDataset] subset_size={self.subset_size}, rows={len(df)}")
        self.df = df
        self.video_names = sorted(self.df["video_name"].astype(str).unique())

        self.corpus = self._detect_corpus(self.csv_path, self.video_dir)

        # meta (пути + лейбл)
        self.meta: List[Dict[str, Any]] = []
        if self.save_prepared_data:
            self.meta = self.store.load_meta(
                self.dataset_name, self.split, getattr(self.config, "random_seed", 0), self.subset_size
            )
        if not self.meta:
            self._build_meta_only()
            if self.save_prepared_data:
                self.store.save_meta(
                    self.dataset_name, self.split, getattr(self.config, "random_seed", 0),
                    self.subset_size, self.meta
                )

        # подготовим/дозаполним кэш фич
        self._prepare_body_cache()

    # ───────────────────── helpers ───────────────────── #

    @staticmethod
    def _detect_corpus(csv_path: str, video_dir: str) -> str:
        def f(s: str) -> Optional[str]:
            s = (s or "").lower()
            if "parkinson" in s: return "parkinson"
            if "depress"   in s: return "depression"
            return None
        return f(csv_path) or f(video_dir) or "unknown"

    def _map_label(self, raw: int) -> int:
        raw = int(raw)
        if self.corpus == "depression":
            return 1 if raw == 1 else 0
        if self.corpus == "parkinson":
            return 2 if raw == 1 else 0
        return 0


    def _find_file(self, base_dir: str, base_filename: str) -> Optional[str]:
        target = os.path.splitext(base_filename)[0]
        for root, _, files in os.walk(base_dir):
            for file in files:
                if os.path.splitext(file)[0] == target:
                    return os.path.join(root, file)
        return None

    def _build_meta_only(self) -> None:
        self.meta = []
        for name in tqdm(self.video_names, desc="Indexing WSM"):
            vpath = self._find_file(self.video_dir, name)
            if vpath is None:
                logging.warning(f"❌ Видео не найдено: {name}")
                continue
            raw = self.df.loc[self.df["video_name"] == name, "label"].values[0]
            class_id = self._map_label(int(raw))
            self.meta.append({
                "sample_name": name,
                "video_path": vpath,
                "label": int(class_id),
            })

    # ─────────────────── feature caching ─────────────────── #

    def _prepare_body_cache(self) -> None:
        if not self.meta:
            return
        sample_names = [m["sample_name"] for m in self.meta]

        mod = "body"
        ex = self.extr

        key = build_cache_key(mod, ex, self.config)
        store, header = self.store.load_modality_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size
        )
        if need_full_reextract(self.config, mod, header, key):
            store = {}

        missing = merge_missing(store, sample_names)
        if not missing:
            return

        for name in tqdm(missing, desc=f"Extracting {mod}"):
            try:
                vpath = self._find_file(self.video_dir, name)
                if not vpath:
                    store[name] = None
                    continue

                # ROI тела → pixel_values [T,3,H,W]
                _, body_pv = get_body_pixel_values(
                    video_path=vpath,
                    segment_length=self.segment_length,
                    image_processor=self.proc,
                    device=self.device,
                    yolo_weights=self.yolo_weights,
                )

                feats = ex.extract(pixel_values=body_pv) if body_pv is not None else None
                feats = self._aggregate(feats, self.average_features) if feats is not None else None
                store[name] = feats

            except Exception as e:
                logging.warning(f"{mod} extract error {name}: {e}")
                store[name] = None

        self.store.save_modality_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size, store
        )
        torch.cuda.empty_cache()

    def _aggregate(self, feats: Any, average: str) -> Optional[dict]:
        """
        feats: {'embedding': Tensor [T,D] или [D]}
        'raw'  -> {'seq': [T,D]}
        'mean' -> {'mean': [D]}
        'mean_std' -> {'mean':[D],'std':[D]}
        """
        if not isinstance(feats, dict):
            raise TypeError(f"Expected dict with key 'embedding', got {type(feats)}")
        emb = feats.get("embedding", None)
        if emb is None or not isinstance(emb, torch.Tensor):
            raise TypeError(f"Features dict must contain 'embedding' Tensor, got keys {list(feats.keys())}")

        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1,D]

        if average == "mean_std":
            return {"mean": emb.mean(dim=0), "std": emb.std(dim=0, unbiased=False)}
        elif average == "mean":
            return {"mean": emb.mean(dim=0)}
        else:  # 'raw'
            return {"seq": emb}

    # ───────────────────── dataset API ───────────────────── #

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.meta[idx]
        name = base["sample_name"]

        # достаем из кэша
        features = {}
        key = build_cache_key("body", self.extr, self.config)
        cache = self.store.get_store(
            self.dataset_name, self.split, key, getattr(self.config, "random_seed", 0), self.subset_size
        )
        features["body"] = cache.get(name, None)

        return {
            "sample_name": name,
            "video_path": base["video_path"],
            "label": torch.tensor(base["label"], dtype=torch.long),
            "features": features,
        }
