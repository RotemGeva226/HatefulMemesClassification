from dataclasses import dataclass
import torch
import torch.nn as nn

try:
    from transformers import BlipModel, Blip2Model, AutoModel
except Exception:
    BlipModel = None
    Blip2Model = None
    AutoModel = None

@dataclass
class ModelConfig:
    backbone_name: str
    freeze_backbone: bool = False
    hidden_dim: int = 768
    dropout: float = 0.1

class VisionLanguageBinaryClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        name_lower = cfg.backbone_name.lower()
        self.feature_dim = cfg.hidden_dim
        self.backbone = None

        if Blip2Model is not None and ("blip2" in name_lower):
            try:
                self.backbone = Blip2Model.from_pretrained(cfg.backbone_name)
                self.feature_dim = getattr(self.backbone.vision_model, "hidden_size", cfg.hidden_dim)
            except Exception:
                pass

        if self.backbone is None and BlipModel is not None:
            try:
                self.backbone = BlipModel.from_pretrained(cfg.backbone_name)
                self.feature_dim = getattr(self.backbone.vision_model.config, "hidden_size", cfg.hidden_dim)
            except Exception:
                pass

        if self.backbone is None and AutoModel is not None:
            self.backbone = AutoModel.from_pretrained(cfg.backbone_name)
            self.feature_dim = getattr(self.backbone.config, "hidden_size", cfg.hidden_dim)

        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(self.feature_dim, 1))

        if cfg.freeze_backbone and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        assert self.backbone is not None, "Backbone not initialized; install transformers and check model name."
        outputs = self.backbone(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **kwargs)
        pooled = None
        if hasattr(outputs, "vision_model_output") and hasattr(outputs.vision_model_output, "pooler_output"):
            pooled = outputs.vision_model_output.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            for v in getattr(outputs, "__dict__", {}).values():
                if torch.is_tensor(v) and v.dim() >= 2:
                    pooled = v.mean(dim=1); break
        if pooled is None:
            raise RuntimeError("Customize forward() pooling per your notebook.")
        logits = self.head(pooled)
        return logits

def build_model(backbone_name: str, freeze_backbone: bool, hidden_dim: int, dropout: float):
    mcfg = ModelConfig(backbone_name, freeze_backbone, hidden_dim, dropout)
    return VisionLanguageBinaryClassifier(mcfg)
