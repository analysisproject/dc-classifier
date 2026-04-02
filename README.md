# KakaoMap Data Center Classifier

## What this does
- Address or GPS input in Streamlit
- Address -> coordinates via Kakao Local REST API
- Kakao satellite map rendered in dashboard
- High-resolution Kakao SKYVIEW screenshot captured with Playwright
- CLIP-based classification of whether the location looks like a data center
- Probability shown on screen

## Files
- `app.py`: Streamlit dashboard
- `prepare_assets.py`: exports centroid and linear-probe artifacts from your labeled image dataset
- `requirements.txt`: Python dependencies

## 1) Install
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 2) Optional: train deployment artifacts from your pilot dataset
Your pilot dataset should follow the same labeling rule as your current script:
- `pos/` = data center images
- `neg/` = non-data-center images

```bash
python prepare_assets.py --data_root ./dataset_dc --out_dir ./artifacts
```

This creates:
- `artifacts/centroids.npz`
- `artifacts/linearprobe.joblib`

## 3) Run app
```bash
streamlit run app.py
```

## 4) Recommended settings
- Quick prototype: `zeroshot`
- More stable production-style inference: `linearprobe`
- No extra training step but still dataset-aware: `centroid`

## Notes
- For address input, you need both a Kakao JavaScript Key and a Kakao REST API Key.
- For GPS-only input, JavaScript Key is enough to render and capture the map, but REST API Key helps reverse-geocode the location.
- The screenshot-based approach is used because the browser-rendered Kakao map is convenient for visual inspection, while the classifier needs actual image pixels.
