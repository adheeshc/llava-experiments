"""Data loader"""

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10


class ImageDataLoader:
    """Load real images from CIFAR-10 and test images"""

    # CIFAR-10 class names mapping
    CIFAR10_CLASSES = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    def __init__(self, data_root: str = "./data", device: str = "cuda"):
        self.data_root = data_root
        self.device = device

        # CIFAR-10 transform
        self.cifar_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Load CIFAR-10
        try:
            self.cifar_dataset = CIFAR10(
                root=self.data_root, train=False, download=True, transform=self.cifar_transform
            )
            print(f"SUCCESS: CIFAR-10 loaded {len(self.cifar_dataset)} images")
        except Exception as e:
            print(f"ERROR: Could not load CIFAR-10: {e}")
            self.cifar_dataset = None

        self.test_images = [
            "./data/blip2_test_images/bird.jpg",
            "./data/blip2_test_images/building.jpg",
            "./data/blip2_test_images/car.jpg",
            "./data/blip2_test_images/cat.jpg",
            "./data/blip2_test_images/dog.jpg",
            "./data/blip2_test_images/flower.jpg",
            "./data/blip2_test_images/food.jpg",
            "./data/blip2_test_images/person.jpg",
        ]

    def get_cifar_batch(self, batch_size: int, start_idx: int = 0) -> torch.Tensor:
        """Get batch of CIFAR-10 images"""
        if self.cifar_dataset is None:
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        batch_images = []
        for i in range(batch_size):
            idx = (start_idx + i) % len(self.cifar_dataset)
            img, _ = self.cifar_dataset[idx]
            batch_images.append(img)
        return torch.stack(batch_images).to(self.device)

    def get_cifar_batch_by_class(self, batch_size: int, class_id: int) -> torch.Tensor:
        """Get batch of CIFAR-10 images from a specific class"""
        if self.cifar_dataset is None:
            return torch.randn(batch_size, 3, 224, 224).to(self.device)

        batch_images = []
        count = 0

        # Iterate through dataset to find images of the specified class
        for img, label in self.cifar_dataset:
            if label == class_id:
                batch_images.append(img)
                count += 1
                if count >= batch_size:
                    break

        if len(batch_images) < batch_size:
            print(f"Warning: Only found {len(batch_images)} images for class {class_id}")

        return torch.stack(batch_images).to(self.device)

    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a given class ID"""
        return self.CIFAR10_CLASSES.get(class_id, "unknown")

    def get_test_images(self, num_images: int = 4) -> torch.Tensor:
        """Load test images from disk"""
        images = []
        for i, img_path in enumerate(self.test_images[:num_images]):
            try:
                img = Image.open(img_path).convert("RGB")
                img = self.cifar_transform(img)
                images.append(img)
            except Exception as e:
                print(f"Could not load image: {e}")
                images.append(torch.randn(3, 224, 224))

        if len(images) == 0:
            return torch.randn(num_images, 3, 224, 224).to(self.device)

        return torch.stack(images).to(self.device)


class SimpleVisionEncoder(nn.Module):
    """Simple vision encoder to process real images into patch features"""

    def __init__(self, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, 224, 224]
        Returns:
            features: [batch, num_patches, embed_dim]
        """
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.norm(x)
        return x
