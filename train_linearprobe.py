import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from PIL import Image
import open_clip
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


DATA_DIR = Path("train_data")
POS_DIR = DATA_DIR / "datacenter"
NEG_DIR = DATA_DIR / "non_datacenter"

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = ARTIFACT_DIR / "linearprobe.joblib"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


@torch.no_grad()
def encode_image(model, preprocess, image_path: Path, device: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def build_dataset(model, preprocess, device: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pos_files = list_images(POS_DIR)
    neg_files = list_images(NEG_DIR)

    if len(pos_files) == 0:
        raise ValueError(f"양성 이미지가 없습니다: {POS_DIR}")
    if len(neg_files) == 0:
        raise ValueError(f"음성 이미지가 없습니다: {NEG_DIR}")

    X = []
    y = []
    names = []

    for p in pos_files:
        emb = encode_image(model, preprocess, p, device)
        X.append(emb)
        y.append(1)
        names.append(str(p))

    for p in neg_files:
        emb = encode_image(model, preprocess, p, device)
        X.append(emb)
        y.append(0)
        names.append(str(p))

    return np.array(X), np.array(y), names


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    X, y, names = build_dataset(model, preprocess, device)
    print(f"[INFO] dataset size = {len(y)}")
    print(f"[INFO] positive = {(y == 1).sum()}, negative = {(y == 0).sum()}")
    print(f"[INFO] embedding dim = {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"[ROC AUC] {auc:.4f}")
    except Exception as e:
        print(f"[ROC AUC 계산 실패] {e}")

    joblib.dump(clf, OUTPUT_PATH)
    print(f"\n[SAVED] {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
