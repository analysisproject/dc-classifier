import subprocess
import os

def ensure_playwright():
    cache_path = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(cache_path):
        subprocess.run(["playwright", "install", "chromium"], check=False)

ensure_playwright()

import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
st.set_page_config(
    page_title="Kakao Satellite DC Classifier",
    page_icon="🛰️",
    layout="wide",
)

# ============================================================
# Paths / constants
# ============================================================
ARTIFACT_DIR = Path("artifacts")
LINEARPROBE_PATH = ARTIFACT_DIR / "linearprobe.joblib"
CENTROIDS_PATH = ARTIFACT_DIR / "centroids.npz"

DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"

POS_PROMPTS = [
    "a satellite image of a data center",
    "an aerial photo of a data center building",
    "aerial view of a large data center facility",
    "satellite view of an industrial data center campus",
    "a data center with rooftop cooling units seen from above",
]

NEG_PROMPTS = [
    "a satellite image of a warehouse",
    "an aerial view of a factory",
    "an aerial photo of a logistics center",
    "a satellite image of an office building",
    "an aerial view of a commercial building complex",
]

# ============================================================
# Secrets / env helpers
# ============================================================
def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:
            v = st.secrets[key]
            if v is not None and str(v).strip():
                return str(v).strip()
    except Exception:
        pass
    v = os.getenv(key, default)
    if v is not None and str(v).strip():
        return str(v).strip()
    return None


# ============================================================
# Model helpers
# ============================================================
@st.cache_resource(show_spinner=True)
def load_clip_model(model_name: str = DEFAULT_MODEL_NAME, pretrained: str = DEFAULT_PRETRAINED):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_pil_image(model, preprocess, pil_img: Image.Image, device: str) -> np.ndarray:
    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    feats = model.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()[0]


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device: str) -> np.ndarray:
    tokens = tokenizer(texts).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    if LINEARPROBE_PATH.exists():
        artifacts["linearprobe"] = joblib.load(LINEARPROBE_PATH)
    if CENTROIDS_PATH.exists():
        data = np.load(CENTROIDS_PATH)
        artifacts["pos_centroid"] = data["pos_centroid"]
        artifacts["neg_centroid"] = data["neg_centroid"]
    return artifacts


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def classify_pil_image(
    pil_img: Image.Image,
    mode: str,
    model,
    preprocess,
    tokenizer,
    device: str,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    img_emb = encode_pil_image(model, preprocess, pil_img, device)

    result = {
        "mode": mode,
        "score": None,
        "probability": None,
        "label": None,
        "details": {},
    }

    if mode == "linearprobe":
        if "linearprobe" not in artifacts:
            raise RuntimeError("artifacts/linearprobe.joblib 파일이 없습니다.")
        clf = artifacts["linearprobe"]
        proba = float(clf.predict_proba(img_emb.reshape(1, -1))[0, 1])
        result["score"] = proba
        result["probability"] = proba
        result["label"] = "데이터센터" if proba >= 0.5 else "비데이터센터"
        return result

    if mode == "centroid":
        if "pos_centroid" not in artifacts or "neg_centroid" not in artifacts:
            raise RuntimeError("artifacts/centroids.npz 파일이 없습니다.")
        pos_cent = artifacts["pos_centroid"]
        neg_cent = artifacts["neg_centroid"]
        pos_cent = pos_cent / (np.linalg.norm(pos_cent) + 1e-9)
        neg_cent = neg_cent / (np.linalg.norm(neg_cent) + 1e-9)

        sim_pos = float(img_emb @ pos_cent)
        sim_neg = float(img_emb @ neg_cent)
        score = sim_pos - sim_neg
        prob = float(sigmoid(score * 5.0))

        result["score"] = score
        result["probability"] = prob
        result["label"] = "데이터센터" if score > 0 else "비데이터센터"
        result["details"] = {"sim_pos": sim_pos, "sim_neg": sim_neg}
        return result

    text_emb_pos = encode_texts(model, tokenizer, POS_PROMPTS, device)
    text_emb_neg = encode_texts(model, tokenizer, NEG_PROMPTS, device)

    sim_pos = img_emb @ text_emb_pos.T
    sim_neg = img_emb @ text_emb_neg.T
    pos_max = float(sim_pos.max())
    neg_max = float(sim_neg.max())
    score = pos_max - neg_max
    prob = float(sigmoid(score * 8.0))

    result["score"] = score
    result["probability"] = prob
    result["label"] = "데이터센터" if score > 0 else "비데이터센터"
    result["details"] = {
        "best_pos_similarity": pos_max,
        "best_neg_similarity": neg_max,
    }
    return result


# ============================================================
# Kakao Local REST API
# ============================================================
def get_auth_headers(rest_key: str) -> Dict[str, str]:
    if not rest_key or not rest_key.strip():
        raise ValueError("REST API Key가 비어 있습니다.")
    return {"Authorization": f"KakaoAK {rest_key.strip()}"}


def geocode_address(rest_key: str, query: str) -> Optional[Tuple[float, float, dict]]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = get_auth_headers(rest_key)
    params = {"query": query}
    r = requests.get(url, headers=headers, params=params, timeout=20)

    if r.status_code == 403:
        raise RuntimeError(
            "주소 검색이 403으로 거부되었습니다. REST API Key가 맞는지 확인하세요."
        )
    r.raise_for_status()

    data = r.json()
    docs = data.get("documents", [])
    if not docs:
        return None

    doc = docs[0]
    lng = float(doc["x"])
    lat = float(doc["y"])
    return lat, lng, doc


# ============================================================
# Kakao Map HTML
# ============================================================
def build_kakao_map_html(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 3,
    map_type: str = "SKYVIEW",
    width: int = 1200,
    height: int = 900,
) -> str:
    if not js_key or not js_key.strip():
        raise ValueError("JavaScript Key가 비어 있습니다.")

    map_type_js = "kakao.maps.MapTypeId.SKYVIEW"
    if map_type.upper() == "HYBRID":
        map_type_js = "kakao.maps.MapTypeId.HYBRID"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: {width}px;
                height: {height}px;
                overflow: hidden;
                background: #000;
            }}
            #map {{
                width: {width}px;
                height: {height}px;
            }}
        </style>
        <script type="text/javascript" src="//dapi.kakao.com/v2/maps/sdk.js?appkey={js_key}"></script>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var container = document.getElementById('map');
            var options = {{
                center: new kakao.maps.LatLng({lat}, {lng}),
                level: {level}
            }};
            var map = new kakao.maps.Map(container, options);
            map.setMapTypeId({map_type_js});

            var markerPosition = new kakao.maps.LatLng({lat}, {lng});
            var marker = new kakao.maps.Marker({{
                position: markerPosition
            }});
            marker.setMap(map);

            var iwContent = '<div style="padding:6px 10px;font-size:12px;">분석 위치</div>';
            var infowindow = new kakao.maps.InfoWindow({{
                content: iwContent
            }});
            infowindow.open(map, marker);
        </script>
    </body>
    </html>
    """


def capture_kakao_satellite(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 3,
    map_type: str = "SKYVIEW",
    width: int = 1200,
    height: int = 900,
) -> Image.Image:
    html = build_kakao_map_html(
        js_key=js_key,
        lat=lat,
        lng=lng,
        level=level,
        map_type=map_type,
        width=width,
        height=height,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "map.html"
        png_path = Path(tmpdir) / "map.png"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(f"file://{html_path.resolve()}", wait_until="networkidle")
            page.wait_for_timeout(3500)
            page.screenshot(path=str(png_path), full_page=False)
            browser.close()

        img = Image.open(png_path).convert("RGB")
        return img


# ============================================================
# UI
# ============================================================
st.title("🛰️ Kakao Satellite Data Center Classifier")
st.caption("주소 또는 GPS 중 하나를 입력하면 카카오 위성사진을 자동 캡처하고 데이터센터 여부를 바로 분류합니다.")

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

with st.sidebar:
    st.header("설정")
    js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
    rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")

    input_mode = st.radio("입력 방식", ["주소 입력", "GPS 입력"], index=0)
    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
    map_type = st.selectbox(
        "지도 타입",
        ["SKYVIEW", "HYBRID"],
        index=0,
        help="SKYVIEW = 순수 위성사진, HYBRID = 위성사진 + 라벨",
    )
    level = st.slider("확대 수준(level)", min_value=1, max_value=8, value=3)
    st.markdown("---")
    st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
    st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

col_left, col_right = st.columns([1.2, 0.8], gap="large")

lat = None
lng = None
meta = None
resolved_text = None

with col_left:
    st.subheader("1) 위치 입력")

    if input_mode == "주소 입력":
        address = st.text_input(
            "주소",
            placeholder="예: 세종특별자치시 도움6로 11 또는 824 Haengbok-daero"
        )
        if st.button("위성사진 불러오기 및 분석", type="primary", use_container_width=True):
            if not rest_key:
                st.error("주소 입력 모드에서는 REST API Key가 필요합니다.")
                st.stop()
            if not address.strip():
                st.warning("주소를 입력하세요.")
                st.stop()

            try:
                geo = geocode_address(rest_key, address.strip())
                if geo is None:
                    st.warning("주소 검색 결과가 없습니다.")
                    st.stop()
                lat, lng, meta = geo
                resolved_text = address.strip()
            except Exception as e:
                st.error(f"주소 검색 오류: {e}")
                st.stop()

    else:
        c1, c2 = st.columns(2)
        with c1:
            lat_text = st.text_input("위도 (Latitude)", placeholder="예: 36.504073")
        with c2:
            lng_text = st.text_input("경도 (Longitude)", placeholder="예: 127.249485")

        if st.button("위성사진 불러오기 및 분석", type="primary", use_container_width=True):
            try:
                lat = float(lat_text)
                lng = float(lng_text)
                resolved_text = "GPS 좌표 입력"
            except Exception as e:
                st.error(f"GPS 좌표 입력 오류: {e}")
                st.stop()

    if lat is not None and lng is not None:
        st.success("입력된 위치를 기준으로 분석을 수행합니다.")
        st.write(f"**위도 / 경도**: {lat:.6f}, {lng:.6f}")
        if resolved_text:
            st.write(f"**입력값**: {resolved_text}")
        if meta:
            with st.expander("주소 검색 상세 응답", expanded=False):
                st.json(meta, expanded=False)

        preview_html = build_kakao_map_html(
            js_key=js_key,
            lat=lat,
            lng=lng,
            level=level,
            map_type=map_type,
            width=1000,
            height=700,
        )
        components.html(preview_html, height=700, scrolling=False)

with col_right:
    st.subheader("2) 분석 결과")

    if lat is not None and lng is not None:
        if not js_key:
            st.error("JavaScript Key가 필요합니다.")
            st.stop()

        try:
            with st.spinner("카카오 위성사진 캡처 중..."):
                sat_img = capture_kakao_satellite(
                    js_key=js_key,
                    lat=lat,
                    lng=lng,
                    level=level,
                    map_type=map_type,
                    width=1200,
                    height=900,
                )

            st.image(sat_img, caption="자동 캡처된 카카오 위성사진", use_container_width=True)

            with st.spinner("데이터센터 분류 중..."):
                model, preprocess, tokenizer, device = load_clip_model()
                artifacts = load_artifacts()
                result = classify_pil_image(
                    pil_img=sat_img,
                    mode=mode,
                    model=model,
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    device=device,
                    artifacts=artifacts,
                )

            prob = float(result["probability"])
            score = float(result["score"])
            label = result["label"]

            if label == "데이터센터":
                st.success(f"판정: **{label}**")
            else:
                st.info(f"판정: **{label}**")

            st.metric("데이터센터 확률", f"{prob * 100:.2f}%")
            st.write(f"점수(score): `{score:.6f}`")
            st.write(f"분류 모드: `{result['mode']}`")

            if result.get("details"):
                with st.expander("세부 점수", expanded=False):
                    st.json(result["details"], expanded=True)

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
    else:
        st.info("왼쪽에서 주소 또는 GPS 중 하나를 입력한 뒤 분석 버튼을 누르세요.")

st.markdown("---")
st.markdown(
    """
    **동작 방식**
    
    - **주소 입력**: REST API로 주소를 좌표로 변환한 뒤 분석합니다. Local API는 `Authorization: KakaoAK {REST_API_KEY}` 헤더를 요구합니다. :contentReference[oaicite:1]{index=1}
    - **GPS 입력**: 좌표를 바로 사용해 위성지도를 띄우고 분석합니다.
    - **지도 타입**: `SKYVIEW`는 순수 위성사진, `HYBRID`는 위성사진+라벨입니다.
    - **분류**: 자동 캡처된 위성사진을 CLIP 기반 분류기로 분석합니다.
    """
)

st.markdown(
    """
    **중요**
    
    이 버전은 Playwright가 필요하므로, Streamlit Community Cloud에서는 설치/실행이 막힐 수 있습니다.  
    로컬 환경이나 Render/Railway 같은 환경에서 실행하는 쪽이 더 안정적입니다.
    """
)
