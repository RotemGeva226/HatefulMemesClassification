from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from torch import nn

class LLaVA(nn.Module):
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        super().__init__()
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')

        # Freeze LLaVA model if needed
        for param in self.model.parameters():
            param.requires_grad = False

        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        ).to(self.device)

    def forward(self, images, prompts):
        # Preprocess inputs
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get last hidden state from LLaVA encoder
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1)

        # Classify (large negative logit: high confidence in class 0, around 0: uncertain)
        logits = self.classifier(pooled.float())
        return logits.float()

    def generate_embeddings(self, images, prompts):
        """No CLS token available.
        Mean pooling is more robust
        Pooling captures both modalities"""
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get last hidden state from LLaVA encoder
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1)
            normalized_emb = F.normalize(pooled, p=2, dim=1)
            return normalized_emb

if __name__ == "__main__":
    model = LLaVA()
    print(model)