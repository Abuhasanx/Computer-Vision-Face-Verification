import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class FaceEmbeddingModel(nn.Module):
    """
    Wraps a torchvision ResNet backbone and replaces its classification
    head with a fixed-length embedding head.

    Args:
        backbone_name (str): one of "resnet18", "resnet34", "resnet50"
        embedding_dim (int): output embedding size (assessment asks for 512)
        pretrained (bool): if True, load ImageNet-pretrained weights
        num_classes (int, optional): if provided, also builds a classification
            head on top of the embedding (for Cross-Entropy identity training).
            Leave as None if training purely with triplet/contrastive loss.
    """

    def __init__(self, backbone_name="resnet50", embedding_dim=512,
                 pretrained=True, num_classes=None):
        super().__init__()

        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # ── Build backbone ──────────────────────────────────────────
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            backbone_out_features = backbone.fc.in_features  # 512
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
            backbone_out_features = backbone.fc.in_features  # 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet50(weights=weights)
            backbone_out_features = backbone.fc.in_features  # 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. "
                              f"Choose from resnet18, resnet34, resnet50.")

        # Strip the original classification head (backbone.fc).
        # Keep everything up to and including global average pooling,
        # which outputs a flat feature vector of size backbone_out_features.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ── Embedding head ──────────────────────────────────────────
        # Projects backbone features -> fixed embedding_dim (512-D as required)
        self.embedding_layer = nn.Linear(backbone_out_features, embedding_dim)

        # Optional classification head, only used if training with
        # Cross-Entropy identity classification (Task 2, approach 1).
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_logits=False):
        """
        Args:
            x: input tensor, shape (B, 3, H, W)
            return_logits: if True and a classifier head exists, also
                return classification logits (for Cross-Entropy training).

        Returns:
            embedding: L2-normalized (B, embedding_dim) tensor
            logits (optional): (B, num_classes) tensor, only if
                return_logits=True and self.classifier is not None
        """
        features = self.backbone(x)                  # (B, backbone_out_features)
        embedding = self.embedding_layer(features)    # (B, embedding_dim)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)  # L2 normalize

        if return_logits:
            if self.classifier is None:
                raise ValueError("return_logits=True but no classifier head "
                                  "was built (num_classes was None at init).")
            logits = self.classifier(embedding)
            return embedding, logits

        return embedding


# ── Preprocessing for real face images ──────────────────────────────
# NOTE: this expects an already-cropped (and ideally aligned) face image,
# i.e. the OUTPUT of dataset_preparation.py (MTCNN detect + crop), not a
# raw uncropped photo. Cropping/alignment happens upstream, not here.
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),          # match ResNet's pretrained input size
    transforms.ToTensor(),                   # HWC [0,255] uint8 -> CHW [0,1] float
    transforms.Normalize(                    # ImageNet mean/std (required since
        mean=[0.485, 0.456, 0.406],          # we're using ImageNet-pretrained weights)
        std=[0.229, 0.224, 0.225],
    ),
])


def load_image_as_tensor(image_path, device="cpu"):
    """
    Loads a single face image from disk and converts it into the
    (1, 3, 224, 224) tensor the model expects.

    Args:
        image_path (str): path to a cropped face image (jpg/png)
        device (str): "cpu" or "cuda"

    Returns:
        torch.Tensor of shape (1, 3, 224, 224)
    """
    img = Image.open(image_path).convert("RGB")  # ensure 3 channels
    tensor = _preprocess(img)                     # (3, 224, 224)
    tensor = tensor.unsqueeze(0)                   # (1, 3, 224, 224) — add batch dim
    return tensor.to(device)


def build_model(backbone_name="resnet50", embedding_dim=512,
                 pretrained=True, num_classes=None, device="cpu"):
    """Convenience factory function."""
    model = FaceEmbeddingModel(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    # ── Sanity check: build model, run a dummy forward pass, print info ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    BACKBONE = "resnet50"
    EMBEDDING_DIM = 512
    NUM_CLASSES = 50  # example: set to your actual number of training identities

    print(f"\nBuilding {BACKBONE} face embedding model...")
    print(f"  embedding_dim = {EMBEDDING_DIM}")
    print(f"  num_classes   = {NUM_CLASSES} (classification head enabled)")

    model = build_model(
        backbone_name=BACKBONE,
        embedding_dim=EMBEDDING_DIM,
        pretrained=True,
        num_classes=NUM_CLASSES,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # Dummy forward pass — 224x224 matches ImageNet-pretrained ResNet input
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # batch of 4

    # Embedding-only forward (used at inference / verification time)
    embedding = model(dummy_input)
    print(f"\nEmbedding-only forward pass:")
    print(f"  Input shape     : {tuple(dummy_input.shape)}")
    print(f"  Embedding shape : {tuple(embedding.shape)}")
    print(f"  Embedding norm (should be ~1.0 per row due to L2 norm): "
          f"{embedding.norm(dim=1)[:4].detach().cpu().numpy()}")

    # Embedding + logits forward (used during Cross-Entropy training)
    embedding, logits = model(dummy_input, return_logits=True)
    print(f"\nEmbedding + classification forward pass:")
    print(f"  Embedding shape : {tuple(embedding.shape)}")
    print(f"  Logits shape    : {tuple(logits.shape)}")

    print("\nModel built and verified successfully.")

    # ── OPTIONAL: test with a REAL image instead of random noise ────────
    # Set this to a real cropped face image path to sanity-check the
    # full load -> preprocess -> embed pipeline end to end.
    input_image = r"D:\biztech\dataset\person_001\img_01.jpg"  # <-- EDIT THIS

    import os
    if os.path.exists(input_image):
        print(f"\n{'='*50}")
        print(f"Testing with real image: {input_image}")
        img_tensor = load_image_as_tensor(input_image, device=device)
        print(f"  Preprocessed tensor shape: {tuple(img_tensor.shape)}")

        model.eval()  # disable dropout/batchnorm updates for inference
        with torch.no_grad():
            real_embedding = model(img_tensor)

        print(f"  Embedding shape : {tuple(real_embedding.shape)}")
        print(f"  Embedding norm  : {real_embedding.norm(dim=1).item():.4f} (should be ~1.0)")
        print(f"  First 5 values : {real_embedding[0][:5].cpu().numpy()}")
    else:
        print(f"\n(Skipping real-image test — path not found: {input_image})")
        print("Edit `input_image` above to point at a real cropped face to test.")