from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

class LLaVA:
    def __init__(self):
        model_id = "llava-hf/llava-1.5-7b-hf"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')

        # Freeze LLaVA model if needed
        for param in self.model.parameters():
            param.requires_grad = False

        # Add classification head (simple linear layer)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 2)  # Binary classification: 0 or 1
        )

    def forward(self, images, prompts):
        # Preprocess inputs
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True).to(0)

        # Get last hidden state from LLaVA encoder
        with torch.no_grad():
            outputs = self.model.vision_tower(images=inputs["pixel_values"])
            image_embeds = outputs.last_hidden_state  # shape: [B, N, D]

        # Aggregate embeddings (e.g., mean-pooling over spatial tokens)
        pooled = image_embeds.mean(dim=1)  # shape: [B, D]

        # Classify
        logits = self.classifier(pooled)
        return logits

if __name__ == "__main__":
    model = LLaVA()
    print(model)