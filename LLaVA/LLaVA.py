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

        # Add classification head (simple linear layer)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification: 0 or 1
        ).to(self.device).half()

    def forward(self, images, prompts):
        # Preprocess inputs
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get last hidden state from LLaVA encoder
        with torch.no_grad():
            outputs = self.model.vision_tower(pixel_values=inputs["pixel_values"])
            image_embeds = outputs.last_hidden_state.to(self.device)  # shape: [B, N, D]

        # Aggregate embeddings (e.g., mean-pooling over spatial tokens)
        pooled = image_embeds.mean(dim=1)  # shape: [B, D]

        # Classify
        logits = self.classifier(pooled)
        return logits

if __name__ == "__main__":
    model = LLaVA()
    print(model)