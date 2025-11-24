"""
LLaVA Paper Exploration

Key Innovations Explored:
1. Visual Instruction Tuning concept
2. Simple linear projection (vs complex Q-Former/cross-attention)
3. GPT-4 assisted data generation
4. Two-stage training methodology
5. Comparison with BLIP-2 and Flamingo approaches

Paper: Visual Instruction Tuning (NeurIPS 2023)
Authors: Haotian Liu et al.

"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from data_utils import ImageDataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ============================================================================
# Part 1: Core LLaVA Architecture Components
# ============================================================================


class CLIPVisionEncoder(nn.Module):
    """
    Simplified CLIP ViT-L/14 vision encoder
    """

    def __init__(self, embed_dim=1024, image_size=224, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Calculate actual number of patches: (224/16)^2 = 196
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=16,
                    dim_feedforward=4096,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(6)  # 6 layers instead of 24
            ]
        )

        self.ln_post = nn.LayerNorm(embed_dim)

        print(f"CLIP Vision Encoder initialized: {embed_dim}D, {self.num_patches} patches")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, 224, 224]
        Returns:
            visual_features: [batch, 196, 1024]
        """
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class LinearProjection(nn.Module):
    """
    LLaVA's Key Innovation:
    - Just a single linear layer (2.5M params)
    - Projects CLIP features (1024D) to LLM space (4096D for 7B, 5120D for 13B)
    - No complex Q-Former (BLIP-2) or cross-attention (Flamingo)
    - Trained in Stage 1, keeps training in Stage 2

    """

    def __init__(self, input_dim=1024, output_dim=4096):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)  # Xavier initialization (as used in paper)
        nn.init.zeros_(self.projection.bias)

        print(f"Linear Projection: {input_dim}D → {output_dim}D ({self.get_param_count()/1e6:.2f}M params)")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, num_patches, 1024] from CLIP
        Returns:
            projected: [batch, num_patches, 4096] for LLM
        """
        return self.projection(visual_features)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class SimplifiedLLM(nn.Module):
    """
    Simplified LLM (Vicuna-7B/13B)
    """

    def __init__(self, hidden_dim=4096, vocab_size=2000, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=16,
                    dim_feedforward=4096,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        print(f"LLM initialized: {hidden_dim}D, {num_layers} layers")

    def forward(
        self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: [batch, num_visual_tokens, 4096] projected visual features
            text_tokens: [batch, seq_len] text token IDs
        Returns:
            logits: [batch, num_visual_tokens + seq_len, vocab_size]
        """
        text_embeds = self.token_embed(text_tokens)  # [B, seq_len, 4096]
        combined = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, num_visual+seq_len, 4096]

        # Create causal mask for text tokens (visual tokens can attend to all)
        seq_len = combined.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(combined.device)

        hidden = combined
        for layer in self.layers:
            hidden = layer(hidden, hidden, tgt_mask=causal_mask)
        hidden = self.ln_f(hidden)
        if return_hidden:
            return hidden
        logits = self.lm_head(hidden)
        return logits


class LLaVAModel(nn.Module):
    """
    Complete LLaVA Architecture

    Components:
    1. Vision Encoder: CLIP ViT-L/14 (frozen)
    2. Projection: Linear layer (trainable)
    3. LLM: Vicuna-7B/13B (frozen Stage 1, fine-tuned Stage 2)
    """

    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()

        self.vision_encoder = CLIPVisionEncoder(embed_dim=vision_dim)
        self.projection = LinearProjection(input_dim=vision_dim, output_dim=llm_dim)
        self.llm = SimplifiedLLM(hidden_dim=llm_dim)

        for param in self.vision_encoder.parameters():
            param.requires_grad = False  # Freeze vision encoder

    @property
    def num_visual_tokens(self):
        """Get the number of visual tokens from the vision encoder"""
        return self.vision_encoder.num_patches

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, 224, 224]
            text_tokens: [batch, seq_len]
        Returns:
            logits: [batch, total_len, vocab_size]
        """

        with torch.no_grad():
            visual_features = self.vision_encoder(images)  # Extract visual features
        projected = self.projection(visual_features)  # Project to LLM space
        logits = self.llm(projected, text_tokens)  # Generate with LLM
        return logits


def demo_llava_architecture():
    """Demo LLaVA's linear projection architecture"""
    print("\n" + "=" * 80)
    print("PART 1: Architecture Comparison")
    print("=" * 80)

    # LLaVA: Linear Projection
    llava_proj = LinearProjection(input_dim=1024, output_dim=4096).to(device)

    llava_params = sum(p.numel() for p in llava_proj.parameters())
    print(f"LLaVA Linear Projection: {llava_params/1e6:.2f}M params")

    # 196 patches from 224x224 image with 16x16 patches (14x14 grid)
    num_patches = 196
    visual_features = torch.randn(2, num_patches, 1024).to(device)
    start = time.time()
    for _ in range(100):
        _ = llava_proj(visual_features)
    llava_time = (time.time() - start) / 100

    print("\nInference Time (batch=2):")
    print(f"LLaVA: {llava_time*1000:.2f}ms")

    llava_output = llava_proj(visual_features)
    print("\nOutput Shape:")
    print(f"LLaVA: {llava_output.shape} (keeps all {num_patches} tokens)")


# ============================================================================
# Part 2: Two-Stage Training Methodology
# ============================================================================
class Stage1PreTraining:
    """
    LLaVA Stage 1: Feature Alignment Pre-training

    Goal: Align vision features with language space
    """

    def __init__(self, model: LLaVAModel, learning_rate=2e-3):
        self.model = model
        self.learning_rate = learning_rate

        # Only optimize projection layer
        self.optimizer = torch.optim.AdamW(self.model.projection.parameters(), lr=learning_rate, weight_decay=0.0)

        print(f"Learning rate: {learning_rate}")
        print(f"Trainable params: {sum(p.numel() for p in self.model.projection.parameters())/1e6:.2f}M")

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Standard language modeling loss on caption tokens

        For next-token prediction:
        - logits[:, i, :] predicts the token at position i+1
        - logits[:, num_visual-1, :] predicts text_tokens[:, 0]
        - logits[:, num_visual, :] predicts text_tokens[:, 1], etc.
        """
        num_visual_tokens = self.model.num_visual_tokens
        vocab_size = logits.shape[-1]
        text_len = targets.shape[1]

        # Get logits that predict text tokens
        # We need positions [num_visual-1, num_visual, ..., num_visual-1+text_len-1]
        text_logits = logits[:, num_visual_tokens - 1 : num_visual_tokens - 1 + text_len, :]

        # Reshape for cross entropy
        logits_flat = text_logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        return loss

    def train_step(self, images: torch.Tensor, text_tokens: torch.Tensor) -> float:
        """Single training step"""
        self.model.train()
        logits = self.model(images, text_tokens)
        loss = self.compute_loss(logits, text_tokens)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Stage2InstructionTuning:
    """
    LLaVA Stage 2: Visual Instruction Tuning

    Goal: Learn to follow diverse visual instructions
    """

    def __init__(self, model: LLaVAModel, learning_rate=2e-5, use_lora=True):
        self.model = model
        self.learning_rate = learning_rate
        self.use_lora = use_lora

        # Unfreeze LLM parameters
        for param in self.model.llm.parameters():
            param.requires_grad = True

        # For simplicity, optimize all LLM params
        trainable_params = list(self.model.projection.parameters()) + list(self.model.llm.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)

        print(f"Learning rate: {learning_rate}")
        print(f"Use LoRA: {use_lora}")
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"Trainable params: {total_trainable/1e6:.1f}M")
        print("Objective: Instruction following (conversation, reasoning, description)")

    def compute_instruction_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, instruction_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss only on assistant's response, not on instruction

        Args:
            logits: [batch, total_len, vocab_size]
            targets: [batch, seq_len]
            instruction_mask: [batch, seq_len] - True for instruction tokens (no loss)
        """
        # Mask out instruction tokens
        masked_targets = targets.clone()
        masked_targets[instruction_mask] = -100

        # Extract text token logits
        num_visual_tokens = self.model.num_visual_tokens
        vocab_size = logits.shape[-1]
        text_len = targets.shape[1]
        text_logits = logits[:, num_visual_tokens - 1 : num_visual_tokens - 1 + text_len, :]

        logits_flat = text_logits.reshape(-1, vocab_size)
        targets_flat = masked_targets.reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        return loss

    def train_step(self, images: torch.Tensor, text_tokens: torch.Tensor, instruction_mask: torch.Tensor) -> float:
        """Single training step with instruction masking"""
        self.model.train()
        logits = self.model(images, text_tokens)
        loss = self.compute_instruction_loss(logits, text_tokens, instruction_mask)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def demo_two_stage_training():
    """
    Demonstrate two-stage training methodology
    1. Stage 1: Feature alignment (projection only)
    2. Stage 2: Instruction tuning (projection + LLM)
    """
    print("\n" + "=" * 80)
    print("PART 2: Two-Stage Training Demonstration")
    print("=" * 80)

    data_loader = ImageDataLoader(device=str(device))
    images = data_loader.get_cifar_batch(batch_size=2, start_idx=0)
    model = LLaVAModel(vision_dim=1024, llm_dim=4096).to(device)
    text_tokens = torch.randint(0, 2000, (2, 32)).to(device)

    # Stage 1: Feature Alignment
    print("\n" + "-" * 80)
    print("Stage 1: Feature Alignment Pre-training")
    print("-" * 80)

    stage1 = Stage1PreTraining(model, learning_rate=2e-3)

    # Training simulation
    print("\nTraining for 3 iterations")
    for i in range(3):
        loss = stage1.train_step(images, text_tokens)
        print(f"Iteration {i+1}: Loss = {loss:.4f}")

    print("\nStage 1 complete: Visual features aligned with language space")

    # Stage 2: Instruction Tuning
    print("\n" + "-" * 80)
    print("Stage 2: Visual Instruction Tuning")
    print("-" * 80)

    stage2 = Stage2InstructionTuning(model, learning_rate=2e-5)

    # Create instruction mask
    instruction_mask = torch.zeros(2, 32, dtype=torch.bool).to(device)
    instruction_mask[:, :16] = True  # First 16 tokens are instruction

    print("\nTraining for 3 iterations")
    for i in range(3):
        loss = stage2.train_step(images, text_tokens, instruction_mask)
        print(f"Iteration {i+1}: Loss = {loss:.4f}")

    print("\nStage 2 complete: Model can follow visual instructions")


# ============================================================================
# Part 3: LLaVA-NeXT Model Setup Testing + CIFAR Image Demo
# ============================================================================


def setup_llava_next(model_size="7b"):
    """
    Initialize LLaVA-NeXT model and processor

    Available models:
    - llava-hf/llava-v1.6-mistral-7b-hf (Mistral-7B base)
    - llava-hf/llava-v1.6-vicuna-7b-hf (Vicuna-7B base)
    - llava-hf/llava-v1.6-vicuna-13b-hf (Vicuna-13B base)
    """

    if model_size == "7b":
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    elif model_size == "13b":
        model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
    else:
        model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"

    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_id)
    print("Processor loaded")

    # Load model
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"Model:{model_id} loaded")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print("\nModel Statistics:")
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Precision: {'FP16' if device == 'cuda' else 'FP32'}")
    print(f"Memory footprint: ~{total_params * 2 / 1e9:.1f}GB (FP16)")

    return model, processor, device


def llava_inference(model, processor, image, prompt, device, max_new_tokens=100):
    """
    Run LLaVA inference on a single image
    """
    # Format conversation (LLaVA uses chat format)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


def demo_cifar10_testing():
    """
    Test LLaVA on CIFAR-10 images

    """
    print("\n" + "=" * 80)
    print("PART 3: LLaVA on CIFAR-10 Images")
    print("=" * 80)

    model, processor, device = setup_llava_next(model_size="7b")
    data_loader = ImageDataLoader(device=str(device))
    test_classes = [0, 1, 2, 3, 8]

    results = []
    for class_id in test_classes:
        print(f"\n{'-'*80}")
        class_name = data_loader.get_class_name(class_id)
        print(f"Testing: {class_name.upper()}")

        cifar_tensor = data_loader.get_cifar_batch_by_class(batch_size=1, class_id=class_id)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = cifar_tensor[0].cpu() * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(img_array)

        prompts = [
            "What do you see in this image?",
            "Describe this image in one sentence.",
            "What is the main object in this image?",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}: '{prompt}'")

            start_time = time.time()
            response = llava_inference(model, processor, pil_image, prompt, device, max_new_tokens=50)
            inference_time = time.time() - start_time

            print(f"Response: {response}")
            print(f"Time: {inference_time:.2f}s")

            results.append({"true_label": class_name, "prompt": prompt, "response": response, "time": inference_time})

        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("CIFAR-10 Testing Summary")
    print("=" * 80)
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Total prompts tested: {len(results)}")

    return results


# ======================================
# Part 4: Demo CLIP Images
# ======================================


def demo_clip_test_images():
    """
    Test LLaVA on high-quality CLIP test images

    Test images:
    - bird.jpg, building.jpg, car.jpg, cat.jpg
    - dog.jpg, flower.jpg, food.jpg, person.jpg
    """
    print("\n" + "=" * 80)
    print("PART 4: LLaVA on High-Quality Test Images")
    print("=" * 80)

    # Setup
    model, processor, device = setup_llava_next(model_size="7b")

    # Test image paths
    test_image_paths = [
        "./data/llava_test_images/bird.jpg",
        "./data/llava_test_images/cat.jpg",
        "./data/llava_test_images/dog.jpg",
        "./data/llava_test_images/flower.jpg",
    ]

    results = []

    for img_path in test_image_paths:
        print(f"\n{'='*80}")
        img_name = os.path.basename(img_path)
        print(f"Testing: {img_name}")
        print("=" * 80)

        try:
            pil_image = Image.open(img_path).convert("RGB")
            print(f"Image size: {pil_image.size}")

            prompts = {
                "Simple": "What is in this image?",
                "Detailed": "Describe this image in detail, including colors, objects, and setting.",
                "Creative": "Write a creative caption for this image.",
                "Analytical": "What can you infer about this scene?",
            }

            for prompt_type, prompt_text in prompts.items():
                print(f"\n{prompt_type} Prompt: '{prompt_text}'")
                start_time = time.time()
                response = llava_inference(model, processor, pil_image, prompt_text, device, max_new_tokens=100)
                inference_time = time.time() - start_time
                print(f"Response: {response}")
                print(f"Time: {inference_time:.2f}s")
                results.append(
                    {"image": img_name, "prompt_type": prompt_type, "response": response, "time": inference_time}
                )
            print("-" * 80)

        except Exception as e:
            print(f"Could not load {img_path}: {e}")
            continue

    if results:
        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"\nAverage inference time: {avg_time:.2f}s")
        print(f"Total images tested: {len(set(r['image'] for r in results))}")
        print(f"Total prompts: {len(results)}")

    return results


# ============================================================================
# Part 5: Demo Instruction Following
# ============================================================================


def demo_instruction_following():
    """
    Deep dive into instruction following capabilities

    Tests:
    1. Simple questions (What is X?)
    2. Detailed descriptions (Describe in detail...)
    3. Reasoning (Why might...?)
    4. Counting (How many...?)
    5. Spatial relationships (Where is...?)
    6. Comparisons (Compare X and Y...)
    """
    print("\n" + "=" * 80)
    print("PART 5: Instruction Following")
    print("=" * 80)

    # Setup
    model, processor, device = setup_llava_next(model_size="7b")

    pil_image = Image.open("./data/llava_test_images/cat.jpg").convert("RGB")
    test_image_name = "cat.jpg"

    print(f"Using image: {test_image_name}")
    print(f"Size: {pil_image.size}")

    # Diverse instruction types
    instruction_types = {
        "Simple Question": "What animal is in this image?",
        "Detailed Description": "Provide a detailed description of this image, including the animal, "
        "its appearance, colors, and any visible background elements.",
        "Reasoning": "Based on what you see, what might this animal be doing or feeling?",
        "Comparative": "How would you describe this animal to someone who has never seen one?",
        "Creative": "If this image were in a magazine, what caption would you give it?",
        "Analytical": "What details in this image make it distinctive or memorable?",
        "Yes/No Question": "Is this a domestic animal?",
        "Open-ended": "Tell me a story about what might have happened before this photo was taken.",
    }

    results = []

    for inst_type, instruction in instruction_types.items():
        print(f"\n{'='*80}")
        print(f"Instruction Type: {inst_type}")
        print("=" * 80)
        print(f"Instruction: '{instruction}'")

        start_time = time.time()
        response = llava_inference(model, processor, pil_image, instruction, device, max_new_tokens=150)
        inference_time = time.time() - start_time

        print("\nResponse:")
        print(f"{response}")
        print(f"\nInference time: {inference_time:.2f}s")
        print(f"Response length: {len(response)} characters")

        results.append(
            {
                "type": inst_type,
                "instruction": instruction,
                "response": response,
                "time": inference_time,
                "length": len(response),
            }
        )

    # Analysis
    print("\n" + "=" * 80)
    print("Instruction Following Analysis")
    print("=" * 80)

    avg_time = sum(r["time"] for r in results) / len(results)
    avg_length = sum(r["length"] for r in results) / len(results)

    print("\nStatistics:")
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Average response length: {avg_length:.0f} characters")
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("LLAVA EXPLORATION")
    print("=" * 50)

    # Part 1: LLAVA Architecture
    demo_llava_architecture()

    # Part 2: Two Stage Training
    demo_two_stage_training()

    # Part 3: CIFAR10 Testing
    demo_cifar10_testing()

    # Part 4: CLIP Image Testing
    demo_clip_test_images()

    # # Part 5: Instruction Following
    demo_instruction_following()
