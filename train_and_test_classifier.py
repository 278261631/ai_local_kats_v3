from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_paths(root_dir: Path, class_names: list[str]) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []
    for class_name in class_names:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
                labels.append(class_name)
    return image_paths, labels


def build_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(random_state=random_state, max_iter=5000, dual="auto")),
        ]
    )


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return "cpu"
    return device_arg


def build_resnet18_feature_extractor(device: str) -> tuple[torch.nn.Module, object]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.to(device)
    preprocess = weights.transforms()
    return backbone, preprocess


def load_resnet18_features(image_paths: list[Path], batch_size: int, device: str) -> np.ndarray:
    model, preprocess = build_resnet18_feature_extractor(device)
    features: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    batch_tensors.append(preprocess(rgb))
            batch = torch.stack(batch_tensors).to(device)
            batch_features = model(batch).cpu().numpy().astype(np.float32)
            features.append(batch_features)

    if not features:
        raise ValueError("No images were loaded. Please verify data paths and file formats.")
    return np.concatenate(features, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test an image classifier from folder data.")
    parser.add_argument("--train-dir", type=Path, default=Path("train_data"), help="Training data root folder.")
    parser.add_argument("--test-dir", type=Path, default=Path("test_data"), help="Testing data root folder.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["good", "process_problem", "sat"],
        help="Class folder names used under train/test directories.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for pretrained feature extraction.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="PyTorch device selection.")
    parser.add_argument("--model-out", type=Path, default=Path("classifier_model.joblib"), help="Model output path.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths, y_train = list_image_paths(args.train_dir, args.classes)
    test_paths, y_test = list_image_paths(args.test_dir, args.classes)

    print(f"Training images: {len(train_paths)}")
    print(f"Testing images:  {len(test_paths)}")

    if not train_paths:
        raise ValueError(f"No training images found in: {args.train_dir}")
    if not test_paths:
        raise ValueError(f"No testing images found in: {args.test_dir}")

    device = resolve_device(args.device)
    print(f"Feature backend: pretrained_resnet18 (device={device}, batch_size={args.batch_size})")
    X_train = load_resnet18_features(train_paths, args.batch_size, device)
    X_test = load_resnet18_features(test_paths, args.batch_size, device)

    model = build_model(args.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    report = classification_report(y_test, y_pred, labels=args.classes, digits=4, zero_division=0)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=args.classes)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"Labels order: {args.classes}")
    print(cm)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "classes": args.classes,
            "feature_backend": "pretrained_resnet18",
        },
        args.model_out,
    )
    print(f"\nSaved model to: {args.model_out}")


if __name__ == "__main__":
    main()
