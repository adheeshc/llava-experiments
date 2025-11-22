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
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ============================================================================
# Part 1: Core LLaVA Architecture Components
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CLIPVisionEncoder(nn.Module):
    """
    Simplified CLIP ViT-L/14 vision encoder
    """

    def __init__(self, embed_dim=1024, num_patches=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

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

        print(f"CLIP Vision Encoder initialized: {embed_dim}D, {num_patches} patches")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, 224, 224]
        Returns:
            visual_features: [batch, 256, 1024]
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
            visual_features: [batch, 256, 1024] from CLIP
        Returns:
            projected: [batch, 256, 4096] for LLM
        """
        return self.projection(visual_features)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class SimplifiedLLM(nn.Module):
    """
    Simplified LLM (Vicuna-7B/13B)
    """

    def __init__(self, hidden_dim=4096, vocab_size=32000, num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=32,
                    dim_feedforward=11008,
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
            visual_tokens: [batch, 256, 4096] projected visual features
            text_tokens: [batch, seq_len] text token IDs
        Returns:
            logits: [batch, total_len, vocab_size]
        """
        text_embeds = self.token_embed(text_tokens)  # [B, seq_len, 4096]
        combined = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, 256+seq_len, 4096]

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

    visual_features = torch.randn(4, 256, 1024).to(device)
    start = time.time()
    for _ in range(100):
        _ = llava_proj(visual_features)
    llava_time = (time.time() - start) / 100

    print("\nInference Time (batch=4):")
    print(f"LLaVA: {llava_time*1000:.2f}ms")

    llava_output = llava_proj(visual_features)
    print("\nOutput Shape:")
    print(f"LLaVA: {llava_output.shape} (keeps all 256 tokens)")


# ============================================================================
# Part 2: Two-Stage Training Methodology
# ============================================================================


if __name__ == "__main__":
    demo_llava_architecture()
