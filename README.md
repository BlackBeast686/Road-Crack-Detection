# Road Crack Detection

Hybrid deep learning models for **classifying road surface damage** from images into four categories: **pothole**, **alligator crack**, **longitudinal crack**, and **lateral crack**. The notebook explores several hybrid CNN architectures that fuse features from multiple backbones to improve classification accuracy.

> **Note on scope:** This is a research/experimental notebook for comparing model architectures, not a production road-inspection system.

## Task & data

- **Inputs:** road surface images (resized to 256×256)
- **Outputs:** one of four damage classes — Pothole, Alligator Crack, Longitudinal Crack, Lateral Crack
- The class label is read from the first token of each image's accompanying `.txt` annotation file.

The dataset is **not** included in this repository. It is downloaded automatically from Kaggle via `kagglehub`:

- **Road Damage:** `alvarobasily/road-damage`

You'll need to configure your own Kaggle credentials.

## Approach

The notebook iterates through several hybrid designs, all sharing the same idea: combine complementary CNN backbones, pool their features, and classify through a dense head. Techniques explored include:

- **Backbone fusion** — MobileNetV3 + a small custom CNN; YOLOv8 backbone + EfficientNet-B0; EfficientNetV2-S with multi-level feature fusion.
- **Attention & pooling** — CBAM attention blocks and GeM (generalized mean) pooling.
- **Training tricks** — strong augmentation (Albumentations), focal / label-smoothing loss, class weighting, cosine LR scheduling, EMA weights, and test-time augmentation (TTA).

## Results

Best validation metrics reported across the notebook's model variants:

| Metric | Score |
|--------|-------|
| Accuracy | ~0.90 |
| F1 (weighted) | ~0.90 |
| ROC-AUC | ~0.93 |
| Log loss | ~0.47 |

Per-class F1 (best run): Pothole ~0.93, Lateral Crack ~0.91, Alligator Crack ~0.86, Longitudinal Crack ~0.84.

> Figures come from single runs on the validation split and vary between the notebook's architecture variants. Treat them as indicative rather than definitive.

## Requirements

Developed in Google Colab (GPU recommended). Main dependencies:

- `torch`
- `torchvision`
- `timm`
- `ultralytics`
- `albumentations`
- `opencv-python`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `kagglehub`

Install everything with:

```bash
pip install -r requirements.txt
```

## Getting started

1. Open the notebook in Google Colab or Jupyter with a GPU.
2. Ensure your Kaggle credentials are configured so `kagglehub` can download the dataset.
3. Run the cells from top to bottom. Each model variant is a self-contained cell you can run independently.

## Repository structure

```
.
├── Cracks_Detection_in_road_code.ipynb
└── README.md
```

