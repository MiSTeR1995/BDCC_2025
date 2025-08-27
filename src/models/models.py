# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)

class AdapterFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.layernorm(x + self.adapter(x))


class GuideBank(nn.Module):
    def __init__(self, out_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(out_dim, hidden_dim))

    def forward(self):
        return self.embeddings

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

class SemanticGuideBank(nn.Module):
    def __init__(self, class_names: List[str], hidden_dim: int, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.class_names = class_names
        self.hidden_dim = hidden_dim

        # Загружаем CLIP
        self.device = _ensure_device(device)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(clip_model_name)

        with torch.no_grad():
            inputs = self.proc(text=[f"a photo of a {c}" for c in class_names], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_embs = self.model.get_text_features(**inputs)

            if hidden_dim != 512:
                proj = nn.Linear(512, hidden_dim).to(self.device)
                text_embs = proj(text_embs)

            self.embeddings = nn.Parameter(text_embs)

    def forward(self):
        return self.embeddings

class DynamicAdjacencyLayer(nn.Module):
    def __init__(self, hidden_dim, temperature=1.0, learnable_temp=True):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=learnable_temp)

    def forward(self, h):
        # h: [B, N, H]
        sim = F.cosine_similarity(h.unsqueeze(2), h.unsqueeze(1), dim=-1)  # [B, N, N]
        adj = torch.softmax(sim / self.temperature, dim=-1)
        return adj  # [B, N, N]

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [B, T, D]
        weights = self.attention(x)  # [B, T, 1]
        return torch.sum(weights * x, dim=1)  # [B, D]


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.1, alpha=0.2):
        super().__init__()
        out_dim = out_dim or in_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, h, adj):
        # h: [B, N, D], adj: [B, N, N]
        B, N, _ = h.size()
        Wh = self.W(h)  # [B,N,D']
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B,N,N,D']
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B,N,N,D']
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)     # [B,N,N,2D']
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B,N,N]
        neg_inf = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, neg_inf)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)  # [B,N,D']
        return h_prime


class MultiModalFusionModel(nn.Module):
    """
    Совмещает модальности: face, audio, text, behavior.
    Три задачи: emotion (7 логитов), personality (5 скоров [0..1]), ah (2 логита).
    """
    def __init__(
        self,
        modality_input_dim: Dict[str, int] = {
            "face": 512,      # 1024 + 1024
            "audio": 512,     # 512 + 512
            "text": 768,      # 768 + 768
            "behavior": 512   # 768 + 768
        },
        hidden_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 256,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        active_tasks=("emotion", "personality", "ah"),
        device: str = "cpu",
        add_similarity: bool = True,  # cosine с GuideBank, можно выключить
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.active_tasks = tuple(active_tasks)
        self.modalities = dict(modality_input_dim)
        self.add_similarity = add_similarity
        self.out_dim = out_dim

        # Проекторы по модальностям: [B,*,Din] -> [B,H]
        self.modality_projectors = nn.ModuleDict({
            mod: nn.Sequential(
                Projector(in_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )
            for mod, in_dim in self.modalities.items()
        })

        self.graph_attns = nn.ModuleDict()
        self.prediction_projectors = nn.ModuleDict()
        self.cross_attns = nn.ModuleDict()
        self.guide_banks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        task_dims = {"emotion": emo_out_dim, "personality": pkl_out_dim, "ah": ah_out_dim}

        for task in self.active_tasks:
            out_dim = task_dims[task]

            # logits-проектор -> обратно в hidden
            self.predictors[task] = nn.Sequential(
                Projector(hidden_dim, out_dim, dropout=dropout),
                AdapterFusion(out_dim, dropout=dropout),
            )
            self.prediction_projectors[task] = nn.Sequential(
                Projector(out_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )

            self.graph_attns[task] = GraphAttentionLayer(hidden_dim, dropout=dropout)
            self.cross_attns[task] = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
            )
            # self.guide_banks[task] = GuideBank(out_dim, self.out_dim)
            self.guide_banks[task] = GuideBank(out_dim, hidden_dim)

            if task == "personality":
                self.heads[task] = nn.Sequential(
                    Projector(hidden_dim, self.out_dim, dropout=dropout),
                    nn.Linear(self.out_dim, out_dim),
                    nn.Sigmoid(),  # [0..1]
                )
            else:
                self.heads[task] = nn.Sequential(Projector(hidden_dim, self.out_dim, dropout=dropout),
                                                nn.Linear(self.out_dim, out_dim))

        self.graph_attns["features"] = GraphAttentionLayer(hidden_dim, dropout=dropout)

    @staticmethod
    def _temporal_pool(x: torch.Tensor | None) -> torch.Tensor | None:
        """Принимает [B,D] или [B,T,D] -> [B,D]"""
        if x is None:
            return None
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unexpected feature shape {tuple(x.shape)}")

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        # print(batch["features"].keys())
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        # приведение всех модальностей к [B,H]
        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        # [B, N, H]
        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            # per-mod logits -> обратно в hidden -> граф-агрегация
            task_logits_per_mod = self.predictors[task](x_stack)                 # [B,N,C]
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)  # [B,N,H]

            adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)               # [B,N,H]

            query = ctx_preds   # [B,N,H]
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)   # [B,N,H]
            task_repr = task_repr.mean(dim=1, keepdim=True) # [B,1,H]
            task_repr = task_repr.squeeze(1)   # [B,H]

            logits = self.heads[task](task_repr)                                  # [B,C] | scores

            if self.add_similarity:
                guides = self.guide_banks[task]()                                 # [C,H]
                sim = F.cosine_similarity(
                    task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                )
                if task == "personality":
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs

class MultiModalFusionModel_v2(nn.Module):
    """
    Совмещает модальности: face, audio, text, behavior.
    Три задачи: emotion (7 логитов), personality (5 скоров [0..1]), ah (2 логита).
    """
    def __init__(
        self,
        modality_input_dim: Dict[str, int] = {
            "face": 512,      # 1024 + 1024
            "audio": 512,     # 512 + 512
            "text": 768,      # 768 + 768
            "behavior": 512   # 768 + 768
        },
        hidden_dim: int = 256,
        num_heads: int = 8,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        active_tasks=("emotion", "personality", "ah"),
        device: str = "cpu",
        add_similarity: bool = True,  # cosine с GuideBank, можно выключить
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.active_tasks = tuple(active_tasks)
        self.modalities = dict(modality_input_dim)
        self.add_similarity = add_similarity

        # Проекторы по модальностям: [B,*,Din] -> [B,H]
        self.modality_projectors = nn.ModuleDict({
            mod: nn.Sequential(
                Projector(in_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )
            for mod, in_dim in self.modalities.items()
        })

        self.graph_attns = nn.ModuleDict()
        self.prediction_projectors = nn.ModuleDict()
        self.cross_attns = nn.ModuleDict()
        self.guide_banks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        task_dims = {"emotion": emo_out_dim, "personality": pkl_out_dim, "ah": ah_out_dim}

        for task in self.active_tasks:
            out_dim = task_dims[task]

            # logits-проектор -> обратно в hidden
            self.predictors[task] = nn.Sequential(
                Projector(hidden_dim, out_dim, dropout=dropout),
                AdapterFusion(out_dim, dropout=dropout),
            )
            self.prediction_projectors[task] = nn.Sequential(
                Projector(out_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )

            self.graph_attns[task] = GraphAttentionLayer(hidden_dim, dropout=dropout)
            self.cross_attns[task] = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
            )
            self.guide_banks[task] = GuideBank(out_dim, hidden_dim)

            if task == "personality":
                self.heads[task] = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),
                    nn.Sigmoid(),  # [0..1]
                )
            else:
                self.heads[task] = nn.Linear(hidden_dim, out_dim)

        self.graph_attns["features"] = GraphAttentionLayer(hidden_dim, dropout=dropout)

    @staticmethod
    def _temporal_pool(x: torch.Tensor | None) -> torch.Tensor | None:
        """Принимает [B,D] или [B,T,D] -> [B,D]"""
        if x is None:
            return None
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unexpected feature shape {tuple(x.shape)}")

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        # print(batch["features"].keys())
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        # приведение всех модальностей к [B,H]
        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        # [B, N, H]
        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            # per-mod logits -> обратно в hidden -> граф-агрегация
            task_logits_per_mod = self.predictors[task](x_stack)                 # [B,N,C]
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)  # [B,N,H]

            adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)               # [B,N,H]

            query = ctx_preds   # [B,N,H]
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)   # [B,N,H]
            task_repr = task_repr.mean(dim=1, keepdim=True) # [B,1,H]
            task_repr = task_repr.squeeze(1)   # [B,H]

            logits = self.heads[task](task_repr)                                  # [B,C] | scores

            if self.add_similarity:
                guides = self.guide_banks[task]()                                 # [C,H]
                sim = F.cosine_similarity(
                    task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                )
                if task == "personality":
                    # logits = (logits + torch.sigmoid(sim)) / 2.0
                    logits = logits
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs

class MultiModalFusionModel_v3(nn.Module):
    """
    Совмещает модальности: face, audio, text, behavior.
    Три задачи: emotion (7 логитов), personality (5 скоров [0..1]), ah (2 логита).
    """
    def __init__(
        self,
        modality_input_dim: Dict[str, int] = {
            "face": 512,      # 1024 + 1024
            "audio": 512,     # 512 + 512
            "text": 768,      # 768 + 768
            "behavior": 512   # 768 + 768
        },
        hidden_dim: int = 256,
        num_heads: int = 8,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        active_tasks=("emotion", "personality", "ah"),
        device: str = "cpu",
        add_similarity: bool = True,  # cosine с GuideBank, можно выключить
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.active_tasks = tuple(active_tasks)
        self.modalities = dict(modality_input_dim)
        self.add_similarity = add_similarity

        # Проекторы по модальностям: [B,*,Din] -> [B,H]
        self.modality_projectors = nn.ModuleDict({
            mod: nn.Sequential(
                Projector(in_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )
            for mod, in_dim in self.modalities.items()
        })

        self.graph_attns = nn.ModuleDict()
        self.prediction_projectors = nn.ModuleDict()
        self.cross_attns = nn.ModuleDict()
        self.guide_banks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        task_dims = {"emotion": emo_out_dim, "personality": pkl_out_dim, "ah": ah_out_dim}

        for task in self.active_tasks:
            out_dim = task_dims[task]

            # logits-проектор -> обратно в hidden
            self.predictors[task] = nn.Sequential(
                Projector(hidden_dim, out_dim, dropout=dropout),
                AdapterFusion(out_dim, dropout=dropout),
            )
            self.prediction_projectors[task] = nn.Sequential(
                Projector(out_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )

            self.graph_attns[task] = GraphAttentionLayer(hidden_dim, dropout=dropout)
            self.cross_attns[task] = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
            )
            self.guide_banks[task] = GuideBank(out_dim, hidden_dim)

            if task == "personality":
                self.heads[task] = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),
                    nn.Sigmoid(),  # [0..1]
                )
            else:
                self.heads[task] = nn.Linear(hidden_dim, out_dim)

        self.graph_attns["features"] = GraphAttentionLayer(hidden_dim, dropout=dropout)

    @staticmethod
    def _temporal_pool(x: torch.Tensor | None) -> torch.Tensor | None:
        """Принимает [B,D] или [B,T,D] -> [B,D]"""
        if x is None:
            return None
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unexpected feature shape {tuple(x.shape)}")

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        # print(batch["features"].keys())
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        # приведение всех модальностей к [B,H]
        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        # [B, N, H]
        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            # per-mod logits -> обратно в hidden -> граф-агрегация
            task_logits_per_mod = self.predictors[task](x_stack)                 # [B,N,C]
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)  # [B,N,H]

            adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)               # [B,N,H]

            query = ctx_preds   # [B,N,H]
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)   # [B,N,H]
            task_repr = task_repr.mean(dim=1, keepdim=True) # [B,1,H]
            task_repr = task_repr.squeeze(1)   # [B,H]

            logits = self.heads[task](task_repr)                                  # [B,C] | scores

            if self.add_similarity:
                guides = self.guide_banks[task]()                                 # [C,H]
                # sim = F.cosine_similarity(
                #     task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                # )
                attn = torch.matmul(task_repr, guides.T)  # [B, C]
                attn = F.softmax(attn, dim=-1)
                if task == "personality":
                    logits = (logits + torch.sigmoid(attn)) / 2.0
                else:
                    logits = (logits + attn) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs

# Только замена guide bank + добавление CLIP
class MultiModalFusionModel_v1_SemanticGuide(MultiModalFusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class_names = {
            "emotion": ["happy", "sad", "angry", "surprised", "disgusted", "fearful", "neutral"],
            "personality": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
            "ah": ["normal", "abnormal"]
        }

        # Пересоздаём guide banks с семантикой
        for task in self.active_tasks:
            if task in class_names:
                self.guide_banks[task] = SemanticGuideBank(
                    class_names[task], self.hidden_dim,
                    clip_model_name="openai/clip-vit-base-patch32",
                    device=self.device
                )

class MultiModalFusionModel_v2_DynamicGraph(MultiModalFusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adj_layer = DynamicAdjacencyLayer(self.hidden_dim)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)  # [B, N, H]
        B, N, H = x_stack.shape

        # Динамическая adjacency
        adj = self.adj_layer(x_stack)  # [B, N, N]
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            task_logits_per_mod = self.predictors[task](x_stack)
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)

            # Динамический граф для предиктов
            adj_t = self.adj_layer(preds_stack)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)

            query = ctx_preds
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)
            task_repr = task_repr.mean(dim=1).squeeze(1)

            logits = self.heads[task](task_repr)

            if self.add_similarity:
                guides = self.guide_banks[task]()
                sim = F.cosine_similarity(
                    task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                )
                if task == "personality":
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs


class MultiModalFusionModel_v3_TemporalAttn(MultiModalFusionModel_v2_DynamicGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_pool = TemporalAttentionPool(self.hidden_dim)

    @staticmethod
    def _temporal_pool(x):  # можно удалить статикметод
        raise NotImplementedError("Use self.temporal_pool instead")

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = tensor.to(self.device)
            if x.dim() == 3:
                x = self.temporal_pool(x)  # [B,T,D] -> [B,D]
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        # [B, N, H]
        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            # per-mod logits -> обратно в hidden -> граф-агрегация
            task_logits_per_mod = self.predictors[task](x_stack)                 # [B,N,C]
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)  # [B,N,H]

            adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)               # [B,N,H]

            query = ctx_preds   # [B,N,H]
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)   # [B,N,H]
            task_repr = task_repr.mean(dim=1, keepdim=True) # [B,1,H]
            task_repr = task_repr.squeeze(1)   # [B,H]

            logits = self.heads[task](task_repr)                                  # [B,C] | scores

            if self.add_similarity:
                guides = self.guide_banks[task]()                                 # [C,H]
                sim = F.cosine_similarity(
                    task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                )                                                                 # [B,C]
                if task == "personality":
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs

class ModalityGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, x):
        # x: [B, H]
        return self.gate(x)  # [B, 1]


class MultiModalFusionModel_v4_GatedFusion(MultiModalFusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Гейт для каждой модальности
        self.gates = nn.ModuleDict({
            mod: ModalityGate(self.hidden_dim) for mod in self.modalities
        })

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_proj = self.modality_projectors[mod](x)  # [B, H]

            # Применяем гейт
            gate_weight = self.gates[mod](x_proj)  # [B, 1]
            x_mods[mod] = x_proj * gate_weight  # масштабируем

            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        # Остальной путь — как в базовой модели
        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            task_logits_per_mod = self.predictors[task](x_stack)
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)

            adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)

            query = ctx_preds
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)
            task_repr = task_repr.mean(dim=1).squeeze(1)

            logits = self.heads[task](task_repr)

            if self.add_similarity:
                guides = self.guide_banks[task]()
                sim = F.cosine_similarity(
                    task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                )
                if task == "personality":
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs


class MultiModalFusionModel_v5_TaskBalancing(MultiModalFusionModel):
    def __init__(self, *args, learn_task_weights=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.learn_task_weights = learn_task_weights
        if learn_task_weights:
            # Логарифм дисперсии (как в GradNorm, Uncertainty Weighting)
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(0.0)) for task in self.active_tasks
            })

    def get_task_loss_weights(self):
        if self.learn_task_weights:
            # exp(-log_var) — вес, обратный дисперсии
            # + log_var — добавляется в loss
            return {task: torch.exp(-self.log_vars[task]) for task in self.active_tasks}
        else:
            return {task: torch.tensor(1.0).to(self.device) for task in self.active_tasks}

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        # Тот же forward
        return super().forward(batch)

class PseudoLabelGuideBank(nn.Module):
    """
    GuideBank, обновляющийся по псевдолейблам (soft-labels из логитов модели).
    Для регрессии (personality): каждый выход — независимая регрессия.
    Для классификации: softmax -> soft-labels.
    """
    def __init__(self, num_classes: int, hidden_dim: int, momentum: float = 0.95):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.momentum = momentum

        # Инициализация guide-эмбеддингов
        self.register_buffer(
            "guides", torch.randn(num_classes, hidden_dim)
        )
        nn.init.xavier_uniform_(self.guides)
        self.is_initialized = False

    def forward(self):
        return self.guides  # [C, H]

    @torch.no_grad()
    def update_from_logits(self, preds_stack: torch.Tensor, logits: torch.Tensor):
        """
        Обновляет guide-эмбеддинги на основе:
        - preds_stack: [B, N, H] — предиктивные эмбеддинги
        - logits: [B, C] — выходы головы

        Использует softmax(logits) как soft-labels.
        """
        B, N, H = preds_stack.shape

        # Soft-labels: [B, C]
        if logits.size(-1) > 1:
            soft_labels = F.softmax(logits, dim=-1)  # классификация
        else:
            # Для случая C=1 (не используется у нас), но на всякий
            soft_labels = torch.sigmoid(logits).clamp(min=1e-6, max=1-1e-6)
            soft_labels = torch.cat([1 - soft_labels, soft_labels], dim=-1)  # [B,2]

        # Усредняем preds_stack по модальностям: [B, N, H] -> [B, H]
        pred_repr = preds_stack.mean(dim=1)  # [B, H]
        pred_repr = F.normalize(pred_repr, dim=-1)  # нормализуем

        # Накопление: вклад каждого примера в каждый класс
        class_sum = torch.matmul(soft_labels.T, pred_repr)  # [C, H]
        class_count = soft_labels.sum(dim=0, keepdim=True).T.clamp(min=1)  # [C, 1]
        avg_embs = class_sum / class_count  # [C, H]

        # EMA update
        if not self.is_initialized:
            self.guides = avg_embs
            self.is_initialized = True
        else:
            self.guides = self.momentum * self.guides + (1 - self.momentum) * avg_embs


class MultiModalFusionModel_v6_PseudoGuide(nn.Module):
    """
    Мультимодальная модель с guide-эмбеддингами, обновляемыми по псевдолейблам.
    Учитывает, что personality — регрессия.
    """
    def __init__(
        self,
        modality_input_dim: Dict[str, int] = {
            "face": 512,
            "audio": 512,
            "text": 768,
            "behavior": 512
        },
        hidden_dim: int = 256,
        num_heads: int = 8,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        active_tasks=("emotion", "personality", "ah"),
        device: str = "cuda",
        add_similarity: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = _ensure_device(device)
        self.active_tasks = tuple(active_tasks)
        self.modalities = dict(modality_input_dim)
        self.add_similarity = add_similarity

        # Проекторы модальностей
        self.modality_projectors = nn.ModuleDict({
            mod: nn.Sequential(
                Projector(in_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )
            for mod, in_dim in self.modalities.items()
        })

        self.graph_attns = nn.ModuleDict()
        self.prediction_projectors = nn.ModuleDict()
        self.cross_attns = nn.ModuleDict()
        self.guide_banks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        task_dims = {"emotion": emo_out_dim, "personality": pkl_out_dim, "ah": ah_out_dim}

        for task in self.active_tasks:
            out_dim = task_dims[task]

            self.predictors[task] = nn.Sequential(
                Projector(hidden_dim, out_dim, dropout=dropout),
                AdapterFusion(out_dim, dropout=dropout),
            )
            self.prediction_projectors[task] = nn.Sequential(
                Projector(out_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            )

            self.graph_attns[task] = GraphAttentionLayer(hidden_dim, dropout=dropout)
            self.cross_attns[task] = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
            )
            self.guide_banks[task] = PseudoLabelGuideBank(out_dim, hidden_dim)

            if task == "personality":
                self.heads[task] = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),
                    nn.Sigmoid()  # [0..1]
                )
            else:
                self.heads[task] = nn.Linear(hidden_dim, out_dim)

        self.graph_attns["features"] = GraphAttentionLayer(hidden_dim, dropout=dropout)

    @staticmethod
    def _temporal_pool(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unexpected feature shape {tuple(x.shape)}")

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []

        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)

        if not x_mods:
            raise ValueError("No valid modalities provided")

        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)  # [B, N, H]
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            task_logits_per_mod = self.predictors[task](x_stack)                 # [B,N,C]
            preds_stack = self.prediction_projectors[task](task_logits_per_mod)  # [B,N,H]

            adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
            ctx_preds = self.graph_attns[task](preds_stack, adj_t)               # [B,N,H]

            query = ctx_preds
            task_repr, _ = self.cross_attns[task](query, ctx_mods, ctx_mods)     # [B,N,H]
            task_repr = task_repr.mean(dim=1)                                    # [B,H]

            logits = self.heads[task](task_repr)                                 # [B,C]

            # --- ОБНОВЛЕНИЕ GUIDE BANK (только при обучении) ---
            if self.add_similarity and self.training:
                self.guide_banks[task].update_from_logits(preds_stack, logits)

            # --- ИСПОЛЬЗОВАНИЕ GUIDE BANK ---
            if self.add_similarity:
                guides = self.guide_banks[task]()  # [C, H]
                sim = F.cosine_similarity(
                    task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1
                )  # [B, C]

                if task == "personality":
                    # Для регрессии: просто усредняем
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits

        return outputs
