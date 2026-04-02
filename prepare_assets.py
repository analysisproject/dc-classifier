import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(data_root: Path) -> List[Path]:
    files = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def infer_label_from_path(p: Path) -> int:
    parent = p.parent.name.lower()
    name = p.name.lower()
    if parent == "pos":
        return 1
    if parent == "neg":
        return 0
    if name.startswith("pos_"):
        return 1
    if name.startswith("neg_"):
        return 0
    raise ValueError(f"Cannot infer label from path: {p}")


def load_image(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


@torch.no_grad()
def encode_images(model, preprocess, image_paths: List[Path], device: torch.device, batch_size: int = 32):
    embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
        batch_paths = image_paths[i:i+batch_size]
        x = torch.stack([preprocess(load_image(p)) for p in batch_paths]).to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.detach().cpu().numpy())
    return np.vstack(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--model_name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model = model.to(device).eval()

    image_paths = list_images(data_root)
    y = np.array([infer_label_from_path(p) for p in image_paths], dtype=np.int64)
    emb = encode_images(model, preprocess, image_paths, device, batch_size=args.batch_size)

    pos_centroid = emb[y == 1].mean(axis=0)
    neg_centroid = emb[y == 0].mean(axis=0)
    pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-9)
    neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-9)
    np.savez(out_dir / "centroids.npz", pos_centroid=pos_centroid, neg_centroid=neg_centroid)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1, random_state=42)
    clf.fit(emb, y)
    joblib.dump(clf, out_dir / "linearprobe.joblib")

    print(f"Saved: {out_dir / 'centroids.npz'}")
    print(f"Saved: {out_dir / 'linearprobe.joblib'}")


if __name__ == "__main__":
    main()
