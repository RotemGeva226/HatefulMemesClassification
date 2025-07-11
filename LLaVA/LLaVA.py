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
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ).to(self.device)

    def forward(self, images, prompts):
        # Preprocess inputs
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get last hidden state from LLaVA encoder
        with torch.no_grad():
            outputs = self.model.vision_tower(pixel_values=inputs["pixel_values"]) # Extract image embeddings
            image_embeds = outputs.last_hidden_state.to(self.device)  # shape: [B, N, D]

        # Mean pooling -> to get fixed-size representation
        pooled = image_embeds.mean(dim=1)  # shape: [B, D]
        pooled = pooled.float()

        # Classify (large negative logit: high confidence in class 0, around 0: uncertain)
        logits = self.classifier(pooled)
        logits = logits.float()
        return logits

    def forward(self, images, prompts):
        # Preprocess inputs
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get last hidden state from LLaVA encoder
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.max(dim=1)[0]

        # Classify (large negative logit: high confidence in class 0, around 0: uncertain)
        logits = self.classifier(pooled.float())
        return logits.float()

    def generate_embeddings(self, images, prompts):
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get last hidden state from LLaVA encoder
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.max(dim=1)[0]
            return pooled

if __name__ == "__main__":
    model = LLaVA()
    print(model)