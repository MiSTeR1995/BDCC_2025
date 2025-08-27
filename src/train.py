# coding: utf-8
from __future__ import annotations

import os, logging
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lion_pytorch import Lion

from .utils.schedulers import SmartScheduler
from .utils.logger_setup import color_metric, color_split
from .utils.measures import mf1, uar, acc_func, ccc, mf1_ah, uar_ah
from .utils.losses import MultiTaskLossWithNaN
from .models.models import MultiModalFusionModel

# ─────────────────────────────── utils ────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def transform_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    "хитрый" пост-процесс для эмоций под мультиклассовую метрику:
    первый логит — "нейтраль/фон", остальные шесть — реальные классы.
    """
    threshold1 = 1 - 1/7
    threshold2 = 1/7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result

def process_predictions(pred_emo: torch.Tensor, true_emo: torch.Tensor):
    """
    Преобразование предсказаний эмоций под mF1/mUAR:
    softmax -> transform_matrix -> binarизация таргетов (отбрасываем "0-й класс").
    """
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).cpu().detach().numpy()
    pred_emo = transform_matrix(pred_emo).tolist()
    true_emo = true_emo.cpu().detach().numpy()
    true_emo = np.where(true_emo > 0, 1, 0)[:, 1:].tolist()
    return pred_emo, true_emo

def drop_domains_in_batch(batch: dict, cfg):
    """Заглушка под старые флаги — сейчас ничего не дропаем (абляций нет)."""
    return batch

def _first_nonempty_batch(loader: DataLoader) -> dict:
    """Возвращает первый не-None батч (с учётом того, что collate_fn может вернуть None)."""
    it = iter(loader)
    while True:
        b = next(it)  # если датасет пуст, пусть падает громко
        if b is not None:
            return b

def _infer_modal_dims_from_batch(batch: dict[str, Any]) -> dict[str, int]:
    """Смотрим на batch['features'][mod] -> [B, D] и собираем {mod: D}."""
    dims = {}
    for mod, x in batch["features"].items():
        if isinstance(x, torch.Tensor):
            dims[mod] = int(x.shape[-1])
            # dims[mod] = int(x.shape[1]/2)
    if not dims:
        raise ValueError("Не удалось вывести размерности модальностей из батча.")
    return dims

# ─────────────────────────── evaluation ────────────────────────────
@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   cfg) -> Dict[str, float]:
    """Собирает метрики на всём лоадере (emotion/personality/AH)."""
    model.eval()
    emo_preds, emo_tgts = [], []
    pkl_preds, pkl_tgts = [], []
    ah_preds, ah_tgts = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = drop_domains_in_batch(batch, cfg)
        out = model(batch)

        # Emotion
        logits_e = out.get("emotion_logits")
        if logits_e is not None:
            y_e = batch["labels"]["emotion"]
            valid_e = ~torch.isnan(y_e).all(dim=1)
            if valid_e.any():
                p, t = process_predictions(logits_e[valid_e], y_e[valid_e])
                emo_preds.extend(p)
                emo_tgts.extend(t)

        # Personality
        preds_p = out.get("personality_scores")
        if preds_p is not None:
            y_p = batch["labels"]["personality"]
            valid_p = ~torch.isnan(y_p).all(dim=1)
            if valid_p.any():
                pkl_preds.append(preds_p[valid_p].detach().cpu().numpy())
                pkl_tgts.append(y_p[valid_p].detach().cpu().numpy())

        # AH (binary, logits->[B,2])
        logits_ah = out.get("ah_logits")
        if logits_ah is not None and "ah" in batch["labels"]:
            y_ah = batch["labels"]["ah"]
            # допускаем float c NaN или long без NaN
            valid_ah = ~(torch.isnan(y_ah) if y_ah.dtype.is_floating_point else torch.zeros_like(y_ah, dtype=torch.bool))
            if valid_ah.any():
                pred = logits_ah[valid_ah].argmax(dim=1).cpu().numpy()
                tgt = (y_ah[valid_ah].long() if y_ah.dtype != torch.long else y_ah[valid_ah]).cpu().numpy()
                ah_preds.append(pred)
                ah_tgts.append(tgt)

    metrics: Dict[str, float] = {}
    if emo_tgts:
        tgt, prd = np.asarray(emo_tgts), np.asarray(emo_preds)
        metrics["mF1"] = float(mf1(tgt, prd))
        metrics["mUAR"] = float(uar(tgt, prd))
    if pkl_tgts:
        tgt, prd = np.vstack(pkl_tgts), np.vstack(pkl_preds)
        metrics["ACC"] = float(acc_func(tgt, prd))
        metrics["CCC"] = float(ccc(tgt, prd))
    if ah_tgts:
        tgt = np.concatenate(ah_tgts, axis=0)
        prd = np.concatenate(ah_preds, axis=0)
        metrics["MF1_AH"] = float(mf1_ah(tgt, prd))
        metrics["UAR_AH"] = float(uar_ah(tgt, prd))
    return metrics


def log_and_aggregate_split(name: str,
                            loaders: dict[str, DataLoader],
                            model: torch.nn.Module,
                            device: torch.device,
                            cfg) -> dict[str, float]:
    """
    Универсальная функция логирования и подсчёта агрегатов для dev/test.
    Теперь учитывает AH и даёт отдельный mean_ah + интегральный mean_all.
    """
    logging.info(f"—— {name} metrics ——")
    all_metrics: dict[str, float] = {}

    for ds_name, loader in loaders.items():
        m = evaluate_epoch(model, loader, device, cfg)
        all_metrics.update({f"{k}_{ds_name}": v for k, v in m.items()})
        msg = " · ".join(color_metric(k, v) for k, v in m.items())
        logging.info(f"[{color_split(name)}:{ds_name}] {msg}")

    mf1s = [v for k, v in all_metrics.items() if k.startswith("mF1_")]
    uars = [v for k, v in all_metrics.items() if k.startswith("mUAR_")]
    accs = [v for k, v in all_metrics.items() if k.startswith("ACC_")]
    cccs = [v for k, v in all_metrics.items() if k.startswith("CCC_")]
    f1_ahs = [v for k, v in all_metrics.items() if k.startswith("MF1_AH_")]
    uar_ahs = [v for k, v in all_metrics.items() if k.startswith("UAR_AH_")]

    if mf1s and uars:
        all_metrics["mean_emo"] = float(np.mean(mf1s + uars))
    if accs and cccs:
        all_metrics["mean_pkl"] = float(np.mean(accs + cccs))
    if f1_ahs and uar_ahs:
        all_metrics["mean_ah"] = float(np.mean(f1_ahs + uar_ahs))

    # общий агрегат по всем доступным подпоказателям
    buckets = []
    for k in ("mean_emo", "mean_pkl", "mean_ah"):
        if k in all_metrics:
            buckets.append(all_metrics[k])
    if buckets:
        all_metrics["mean_all"] = float(np.mean(buckets))

    # компактное резюме
    if any(k in all_metrics for k in ("mean_emo", "mean_pkl", "mean_ah", "mean_all")):
        summary_parts = []
        for k in ("mean_emo", "mean_pkl", "mean_ah", "mean_all"):
            if k in all_metrics:
                summary_parts.append(color_metric(k, all_metrics[k]))
        logging.info(f"{name} Summary | " + " ".join(summary_parts))

    return all_metrics


# ────────────────────────── основной train() ──────────────────────────
def train(cfg,
          mm_loader: DataLoader,
          dev_loaders: dict[str, DataLoader] | None = None,
          test_loaders: dict[str, DataLoader] | None = None):
    """
    Чистый мультитаск на трёх задачах: emotion, personality, ah.
    """
    seed_everything(cfg.random_seed)
    device = cfg.device

    # Смотрим на реальный первый батч и фиксируем входные размерности модальностей
    sample_batch = _first_nonempty_batch(mm_loader)
    modality_input_dim = _infer_modal_dims_from_batch(sample_batch)

    # ─── Модель ───────────────────────────────────────────────────────
    model = MultiModalFusionModel(
        modality_input_dim=modality_input_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_transformer_heads,
        out_dim = cfg.out_features,
        dropout=cfg.dropout,
        emo_out_dim=7,
        pkl_out_dim=5,
        ah_out_dim=2,
        device=device,
        add_similarity=getattr(cfg, "add_similarity", True),
    ).to(device)

    # ─── Оптимизатор ──────────────────────────────────────────────────
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"⛔ Неизвестный оптимизатор: {cfg.optimizer}")
    logging.info(f"⚙️ Оптимизатор: {cfg.optimizer}, learning rate: {cfg.lr}")

    # ─── Scheduler ────────────────────────────────────────────────────
    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch
    )

    # ─── Лосс (вес AH добавлен) ───────────────────────────────────────
    criterion = MultiTaskLossWithNaN(
        weight_emotion=getattr(cfg, "weight_emotion", 1.0),
        weight_personality=getattr(cfg, "weight_personality", getattr(cfg, "weight_pers", 1.0)),
        weight_ah=getattr(cfg, "weight_ah", 1.0),  # <-- добавили вес для AH
        emo_weights=(torch.FloatTensor(
            [5.890161, 7.534918, 11.228363, 27.722221,
             1.3049748, 5.6189237, 26.639517]).to(device)
                     if getattr(cfg, "flag_emo_weight", False) else None),
        personality_loss_type=getattr(cfg, "personality_loss_type", getattr(cfg, "pers_loss_type", "ccc")),
        emotion_loss_type=getattr(cfg, "emotion_loss_type", "BCE"),
    )

    best_dev, best_test = {}, {}
    best_score  = -float("inf")
    patience_counter = 0

    # ── 1. Эпохи ──────────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        logging.info(f"═══ EPOCH {epoch + 1}/{cfg.num_epochs} ═══")
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds_emo, total_targets_emo = [], []
        total_preds_per, total_targets_per = [], []
        total_preds_ah,  total_targets_ah  = [], []

        for batch in tqdm(mm_loader):
            if batch is None:
                continue

            batch = drop_domains_in_batch(batch, cfg)

            # лейблы
            emo_labels = batch["labels"]["emotion"].to(device)
            per_labels = batch["labels"]["personality"].to(device)
            ah_labels  = batch["labels"].get("ah", None)
            if ah_labels is not None:
                ah_labels = ah_labels.to(device)

            # валидные маски
            valid_emo = ~torch.isnan(emo_labels).all(dim=1)
            valid_per = ~torch.isnan(per_labels).all(dim=1)
            if ah_labels is None:
                valid_ah = None
            else:
                valid_ah = ~(torch.isnan(ah_labels) if ah_labels.dtype.is_floating_point
                             else torch.zeros_like(ah_labels, dtype=torch.bool))

            # форвард
            outputs = model(batch)

            # лосс: скармливаем все три задачи и маски
            loss_inputs = {
                "emotion": emo_labels,
                "personality": per_labels,
                "valid_emo": valid_emo,
                "valid_per": valid_per,
            }
            if ah_labels is not None:
                loss_inputs["ah"] = ah_labels
                loss_inputs["valid_ah"] = valid_ah

            loss = criterion(outputs, loss_inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_level=True)

            bs = emo_labels.shape[0]
            total_loss += loss.item() * bs
            total_samples += bs

            # --- train сбор предсказаний для логов (опционально) ---
            if outputs.get('emotion_logits') is not None and valid_emo.any():
                preds_emo, targets_emo = process_predictions(
                    outputs['emotion_logits'][valid_emo],
                    emo_labels[valid_emo]
                )
                total_preds_emo.extend(preds_emo)
                total_targets_emo.extend(targets_emo)

            if outputs.get('personality_scores') is not None and valid_per.any():
                preds_per = outputs['personality_scores'][valid_per]
                targets_per = per_labels[valid_per]
                total_preds_per.extend(preds_per.detach().cpu().numpy().tolist())
                total_targets_per.extend(targets_per.detach().cpu().numpy().tolist())

            if ah_labels is not None and outputs.get('ah_logits') is not None and valid_ah.any():
                pred_ah = outputs['ah_logits'][valid_ah].argmax(dim=1).cpu().numpy().tolist()
                tgt_ah  = (ah_labels[valid_ah].long() if ah_labels.dtype != torch.long else ah_labels[valid_ah]).cpu().numpy().tolist()
                total_preds_ah.extend(pred_ah)
                total_targets_ah.extend(tgt_ah)

        # --- train метрики ---
        train_loss = total_loss / max(1, total_samples)

        # эмоции
        if total_targets_emo:
            mF1_train = mf1(np.asarray(total_targets_emo), np.asarray(total_preds_emo))
            mUAR_train = uar(np.asarray(total_targets_emo), np.asarray(total_preds_emo))
            mean_emo_train = np.mean([mF1_train, mUAR_train])
        else:
            mF1_train = mUAR_train = mean_emo_train = float('nan')

        # персоналити
        if total_targets_per:
            t_per = np.asarray(total_targets_per)
            p_per = np.asarray(total_preds_per)
            acc_train = acc_func(t_per, p_per)
            # CCC можно не считать на каждом батче, но пусть будет для симметрии:
            ccc_vals = []
            for i in range(t_per.shape[1]):
                mask = ~np.isnan(t_per[:, i])
                if mask.sum() == 0: continue
                ccc_vals.append(ccc(t_per[mask, i], p_per[mask, i]))
            ccc_train = float(np.mean(ccc_vals)) if ccc_vals else float('nan')
            mean_pkl_train = np.nanmean([acc_train, ccc_train])
        else:
            acc_train = ccc_train = mean_pkl_train = float('nan')

        # AH
        if total_targets_ah:
            mf1_ah_train = mf1_ah(np.asarray(total_targets_ah), np.asarray(total_preds_ah))
            uar_ah_train = uar_ah(np.asarray(total_targets_ah), np.asarray(total_preds_ah))
            mean_ah_train = np.mean([mf1_ah_train, uar_ah_train])
        else:
            mf1_ah_train = uar_ah_train = mean_ah_train = float('nan')

        # красивый лог
        parts = [
            f"Loss={train_loss:.4f}",
            f"EMO: UAR={mUAR_train:.4f} MF1={mF1_train:.4f} MEAN={mean_emo_train:.4f}",
            f"PKL: ACC={acc_train:.4f} CCC={ccc_train:.4f} MEAN={mean_pkl_train:.4f}",
            f"AH:  UAR={uar_ah_train:.4f} MF1={mf1_ah_train:.4f} MEAN={mean_ah_train:.4f}",
        ]
        logging.info(f"[{color_split('TRAIN')}] " + " | ".join(parts))

        # ── Evaluation ──
        cur_dev = log_and_aggregate_split("Dev", dev_loaders, model, device, cfg) if dev_loaders else {}
        cur_test = log_and_aggregate_split("Test", test_loaders, model, device, cfg) if test_loaders else {}

        # основная метрика для ранней остановки:
        # предпочитаем mean_all (emo+pkl+ah), если её можно посчитать,
        # иначе — mean_emo или mean_pkl (что доступно).
        cur_eval = cur_dev if getattr(cfg, "early_stop_on", "dev") == "dev" else cur_test

        metric_val = None
        for key in ("mean_all", "mean_emo", "mean_pkl", "mean_ah"):
            if key in cur_eval:
                metric_val = cur_eval[key]
                break
        if metric_val is None:
            metric_val = -float("inf")

        scheduler.step(metric_val)

        improved = metric_val > best_score
        if improved:
            best_score  = metric_val
            best_dev = cur_dev
            best_test = cur_test
            patience_counter = 0

            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

            def fmt(x): return f"{x:.4f}" if x is not None else "NA"
            ckpt_name = f"best_ep{epoch + 1}_all{fmt(cur_eval.get('mean_all'))}_emo{fmt(cur_eval.get('mean_emo'))}_pkl{fmt(cur_eval.get('mean_pkl'))}_ah{fmt(cur_eval.get('mean_ah'))}.pt"
            ckpt_path = Path(cfg.checkpoint_dir) / ckpt_name
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"✔ Best model saved: {ckpt_path.name}")
        else:
            patience_counter += 1
            logging.warning(f"No improvement — patience {patience_counter}/{cfg.max_patience}")
            if patience_counter >= cfg.max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    return best_dev, best_test
