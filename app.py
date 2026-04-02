import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import open_clip
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from playwright.sync_api import sync_playwright

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="KakaoMap Data Center Classifier", layout="wide")

DEFAULT_POS_PROMPTS = [
    "a satellite image of a data center",
    "an aerial photo of a data center facility",
    "aerial view of a data center campus",
    "a satellite image of large rectangular industrial buildings with cooling infrastructure",
]

DEFAULT_NEG_PROMPTS = [
    "a satellite image of a warehouse",
    "an aerial photo of a logistics center",
    "aerial view of a factory",
    "a satellite image of a commercial building",
]

# ============================================================
# Helpers: Kakao Local REST
# ============================================================
def geocode_address(rest_api_key: str, address: str) -> Dict:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {rest_api_key}"}
    params = {"query": address}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    docs = data.get("documents", [])
    if not docs:
        raise ValueError("주소 검색 결과가 없습니다.")
    top = docs[0]
    return {
        "lat": float(top["y"]),
        "lng": float(top["x"]),
        "address_name": top.get("address_name", address),
        "road_address": (top.get("road_address") or {}).get("address_name"),
    }


def coord_to_address(rest_api_key: str, lat: float, lng: float) -> Dict:
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {rest_api_key}"}
    params = {"x": lng, "y": lat}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    docs = data.get("documents", [])
    if not docs:
        return {"address_name": None, "road_address": None}

    top = docs[0]
    return {
        "address_name": (top.get("address") or {}).get("address_name"),
        "road_address": (top.get("road_address") or {}).get("address_name"),
    }


# ============================================================
# Kakao map HTML / screenshot
# ============================================================
def build_kakao_map_html(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 2,
    width: int = 1280,
    height: int = 960,
    map_type: str = "HYBRID",
    marker: bool = True,
) -> str:
    marker_js = """
        const marker = new kakao.maps.Marker({ position: center });
        marker.setMap(map);
    """ if marker else ""

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Kakao Map Capture</title>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:#fff; }}
    #map {{ width:{width}px; height:{height}px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey={js_key}"></script>
  <script>
    const container = document.getElementById('map');
    const center = new kakao.maps.LatLng({lat}, {lng});
    const options = {{ center: center, level: {level} }};
    const map = new kakao.maps.Map(container, options);
    map.setMapTypeId(kakao.maps.MapTypeId.{map_type});
    {marker_js}
  </script>
</body>
</html>
"""


def render_map_component(js_key: str, lat: float, lng: float, level: int, height: int = 520):
    html = build_kakao_map_html(
        js_key=js_key,
        lat=lat,
        lng=lng,
        level=level,
        width=1200,
        height=height,
        map_type="HYBRID",
        marker=True,
    )
    components.html(html, height=height + 10, scrolling=False)


@st.cache_data(show_spinner=False)
def capture_kakao_map(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 2,
    width: int = 1280,
    height: int = 960,
    map_type: str = "SKYVIEW",
) -> bytes:
    html = build_kakao_map_html(
        js_key=js_key,
        lat=lat,
        lng=lng,
        level=level,
        width=width,
        height=height,
        map_type=map_type,
        marker=False,
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        html_path = td_path / "map.html"
        png_path = td_path / "map.png"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.wait_for_timeout(2500)
            page.locator("#map").screenshot(path=str(png_path))
            browser.close()

        return png_path.read_bytes()


# ============================================================
# CLIP + inference helpers
# ============================================================
@st.cache_resource(show_spinner=False)
def load_clip_model(model_name: str, pretrained: str, device_preference: str):
    device = torch.device(device_preference if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_image_bytes(
    image_bytes: bytes,
    model,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)
    feats = model.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()[0]


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: torch.device) -> np.ndarray:
    tokens = tokenizer(texts).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict_zeroshot(
    emb: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> Dict:
    pos_text = encode_texts(model, tokenizer, pos_prompts, device)
    neg_text = encode_texts(model, tokenizer, neg_prompts, device)
    sim_pos = emb @ pos_text.T
    sim_neg = emb @ neg_text.T
    raw_score = float(sim_pos.max() - sim_neg.max())
    prob = float(sigmoid(raw_score * 5.0))
    return {
        "label": int(prob >= 0.5),
        "probability": prob,
        "raw_score": raw_score,
        "max_pos_similarity": float(sim_pos.max()),
        "max_neg_similarity": float(sim_neg.max()),
    }


def predict_centroid(emb: np.ndarray, centroid_npz_path: str) -> Dict:
    data = np.load(centroid_npz_path)
    pos_cent = data["pos_centroid"]
    neg_cent = data["neg_centroid"]
    sim_pos = float(emb @ pos_cent)
    sim_neg = float(emb @ neg_cent)
    raw_score = sim_pos - sim_neg
    prob = float(sigmoid(raw_score * 8.0))
    return {
        "label": int(prob >= 0.5),
        "probability": prob,
        "raw_score": raw_score,
        "max_pos_similarity": sim_pos,
        "max_neg_similarity": sim_neg,
    }


def predict_linearprobe(emb: np.ndarray, linearprobe_path: str) -> Dict:
    clf = joblib.load(linearprobe_path)
    prob = float(clf.predict_proba(emb.reshape(1, -1))[0, 1])
    raw_score = float(clf.decision_function(emb.reshape(1, -1))[0])
    return {
        "label": int(prob >= 0.5),
        "probability": prob,
        "raw_score": raw_score,
    }


# ============================================================
# UI
# ============================================================
st.title("KakaoMap 기반 데이터센터 분류 대시보드")
st.caption("주소 또는 GPS 좌표를 넣으면 카카오맵 위성 이미지를 불러와 데이터센터 여부를 예측합니다.")

with st.sidebar:
    st.header("설정")
    kakao_js_key = st.text_input("Kakao JavaScript Key", type="password")
    kakao_rest_key = st.text_input("Kakao REST API Key", type="password")

    st.markdown("---")
    st.subheader("지도 캡처 옵션")
    zoom_level = st.slider("Kakao level (작을수록 더 확대)", min_value=0, max_value=14, value=2)
    capture_width = st.selectbox("캡처 너비", [768, 1024, 1280, 1536], index=2)
    capture_height = st.selectbox("캡처 높이", [512, 768, 960, 1024], index=2)

    st.markdown("---")
    st.subheader("모델 옵션")
    model_name = st.text_input("CLIP model_name", value="ViT-B-32")
    pretrained = st.text_input("CLIP pretrained", value="openai")
    device_pref = st.selectbox("device", ["cuda:0", "cpu"], index=0)
    inference_mode = st.selectbox("분류 방식", ["zeroshot", "centroid", "linearprobe"], index=0)

    centroid_path = st.text_input("centroid npz 경로", value="artifacts/centroids.npz")
    linearprobe_path = st.text_input("linear probe joblib 경로", value="artifacts/linearprobe.joblib")

    st.markdown("---")
    st.subheader("Zero-shot prompt")
    pos_prompts_text = st.text_area("Positive prompts", value="\n".join(DEFAULT_POS_PROMPTS), height=120)
    neg_prompts_text = st.text_area("Negative prompts", value="\n".join(DEFAULT_NEG_PROMPTS), height=120)

left, right = st.columns([1, 1])

with left:
    input_mode = st.radio("입력 방식", ["주소", "GPS 좌표"], horizontal=True)
    if input_mode == "주소":
        address_query = st.text_input("주소", placeholder="예: 경기도 성남시 분당구 ...")
        lat = None
        lng = None
    else:
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("위도 (latitude)", value=37.5665, format="%.8f")
        with c2:
            lng = st.number_input("경도 (longitude)", value=126.9780, format="%.8f")
        address_query = None

    run_btn = st.button("지도 불러오고 분류하기", type="primary", use_container_width=True)

with right:
    st.info(
        "권장 흐름: zeroshot으로 먼저 확인 → 정확도를 높이려면 pilot 코드로 학습한 centroid 또는 linear probe 가중치를 연결"
    )

if run_btn:
    if not kakao_js_key:
        st.error("Kakao JavaScript Key가 필요합니다.")
        st.stop()

    if input_mode == "주소" and not kakao_rest_key:
        st.error("주소 입력을 쓰려면 Kakao REST API Key가 필요합니다.")
        st.stop()

    try:
        if input_mode == "주소":
            info = geocode_address(kakao_rest_key, address_query)
            lat = info["lat"]
            lng = info["lng"]
            resolved_address = info.get("road_address") or info.get("address_name") or address_query
        else:
            resolved = coord_to_address(kakao_rest_key, lat, lng) if kakao_rest_key else {}
            resolved_address = resolved.get("road_address") or resolved.get("address_name") or "주소 정보 없음"

        st.subheader("1) 위치 확인")
        st.write(f"- 좌표: **{lat:.6f}, {lng:.6f}**")
        st.write(f"- 주소: **{resolved_address}**")
        render_map_component(kakao_js_key, lat, lng, zoom_level)

        st.subheader("2) 분석용 위성 이미지 캡처")
        with st.spinner("Kakao 위성 지도를 캡처하는 중..."):
            png_bytes = capture_kakao_map(
                kakao_js_key,
                lat,
                lng,
                level=zoom_level,
                width=int(capture_width),
                height=int(capture_height),
                map_type="SKYVIEW",
            )

        st.image(png_bytes, caption="분류에 사용된 Kakao SKYVIEW 캡처 이미지", use_container_width=True)

        with st.spinner("CLIP 임베딩/분류 수행 중..."):
            model, preprocess, tokenizer, device = load_clip_model(model_name, pretrained, device_pref)
            emb = encode_image_bytes(png_bytes, model, preprocess, device)

            if inference_mode == "zeroshot":
                pos_prompts = [x.strip() for x in pos_prompts_text.splitlines() if x.strip()]
                neg_prompts = [x.strip() for x in neg_prompts_text.splitlines() if x.strip()]
                result = predict_zeroshot(emb, model, tokenizer, device, pos_prompts, neg_prompts)
            elif inference_mode == "centroid":
                if not Path(centroid_path).exists():
                    st.error(f"centroid 파일을 찾을 수 없습니다: {centroid_path}")
                    st.stop()
                result = predict_centroid(emb, centroid_path)
            else:
                if not Path(linearprobe_path).exists():
                    st.error(f"linear probe 파일을 찾을 수 없습니다: {linearprobe_path}")
                    st.stop()
                result = predict_linearprobe(emb, linearprobe_path)

        st.subheader("3) 예측 결과")
        probability = float(result["probability"])
        label = "데이터센터 가능성 높음" if result["label"] == 1 else "데이터센터 아님 가능성 높음"
        st.metric("데이터센터 확률", f"{probability * 100:.2f}%")
        st.success(label) if result["label"] == 1 else st.warning(label)

        detail_cols = st.columns(3)
        detail_cols[0].metric("예측 라벨", "1" if result["label"] == 1 else "0")
        detail_cols[1].metric("Raw score", f"{result.get('raw_score', float('nan')):.4f}")
        detail_cols[2].metric("Mode", inference_mode)

        with st.expander("세부 결과(JSON)"):
            st.json(result)

        with st.expander("캡처 이미지 저장"):
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            href = f'<a download="kakao_capture.png" href="data:image/png;base64,{b64}">PNG 다운로드</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.exception(e)
