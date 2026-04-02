import io
import os
import json
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

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="KakaoMap Data Center Classifier",
    page_icon="🛰️",
    layout="wide",
)

# ============================================================
# Constants
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
    "aerial view of a factory",
    "an aerial photo of a logistics center",
    "a satellite image of an office building",
    "aerial view of a commercial building complex",
]

# ============================================================
# Helpers: secrets / env
# ============================================================
def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def require_api_key(name: str) -> Optional[str]:
    return get_secret_or_env(name)

# ============================================================
# Helpers: model loading
# ============================================================
@st.cache_resource(show_spinner=True)
def load_clip_model(model_name: str = DEFAULT_MODEL_NAME, pretrained: str = DEFAULT_PRETRAINED):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
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

def classify_image(
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
        "details": {}
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
        prob = float(sigmoid(score * 5.0))  # heuristic scaling

        result["score"] = score
        result["probability"] = prob
        result["label"] = "데이터센터" if score > 0 else "비데이터센터"
        result["details"] = {
            "sim_pos": sim_pos,
            "sim_neg": sim_neg,
        }
        return result

    # zeroshot
    text_emb_pos = encode_texts(model, tokenizer, POS_PROMPTS, device)
    text_emb_neg = encode_texts(model, tokenizer, NEG_PROMPTS, device)

    sim_pos = img_emb @ text_emb_pos.T
    sim_neg = img_emb @ text_emb_neg.T

    pos_max = float(sim_pos.max())
    neg_max = float(sim_neg.max())
    score = pos_max - neg_max
    prob = float(sigmoid(score * 8.0))  # heuristic scaling

    result["score"] = score
    result["probability"] = prob
    result["label"] = "데이터센터" if score > 0 else "비데이터센터"
    result["details"] = {
        "best_pos_similarity": pos_max,
        "best_neg_similarity": neg_max,
    }
    return result

# ============================================================
# Helpers: Kakao API
# ============================================================
def geocode_address(rest_key: str, query: str) -> Optional[Tuple[float, float, dict]]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {rest_key}"}
    params = {"query": query}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    docs = data.get("documents", [])
    if not docs:
        return None

    doc = docs[0]
    lng = float(doc["x"])
    lat = float(doc["y"])
    return lat, lng, doc

def reverse_geocode(rest_key: str, lat: float, lng: float) -> Optional[dict]:
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {rest_key}"}
    params = {"x": lng, "y": lat}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    docs = data.get("documents", [])
    return docs[0] if docs else None

def build_kakao_map_html(js_key: str, lat: float, lng: float, level: int = 3, map_type: str = "HYBRID") -> str:
    map_type_js = "kakao.maps.MapTypeId.HYBRID" if map_type.upper() == "HYBRID" else "kakao.maps.MapTypeId.SKYVIEW"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
            }}
            #map {{
                width: 100%;
                height: 100vh;
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

            var iwContent = '<div style="padding:6px 10px;font-size:12px;">분석 대상 위치</div>';
            var infowindow = new kakao.maps.InfoWindow({{
                content: iwContent
            }});
            infowindow.open(map, marker);
        </script>
    </body>
    </html>
    """
    return html

# ============================================================
# UI
# ============================================================
st.title("🛰️ KakaoMap Data Center Classifier")
st.caption("주소 또는 GPS 좌표로 위치를 확인하고, 위성 이미지 파일을 업로드해 데이터센터 여부를 분류합니다.")

with st.expander("배포용 구조 안내", expanded=False):
    st.markdown(
        """
        이 버전은 Streamlit Cloud에서 안정적으로 돌아가도록 Playwright를 제거한 버전입니다.

        현재 동작 방식:
        - 주소/좌표 입력
        - Kakao 지도 표시
        - 사용자가 위성 캡처 이미지를 업로드
        - CLIP 기반 데이터센터 분류

        자동 캡처 없이도 URL 배포가 쉽고, 분류 모델 테스트와 데모에 적합합니다.
        """
    )

js_key = require_api_key("KAKAO_JS_KEY")
rest_key = require_api_key("KAKAO_REST_KEY")

if not js_key or not rest_key:
    st.error("KAKAO_JS_KEY, KAKAO_REST_KEY를 Streamlit Secrets 또는 환경변수에 설정해야 합니다.")
    st.stop()

# Sidebar
st.sidebar.header("설정")
mode = st.sidebar.selectbox(
    "분류 모드",
    options=["zeroshot", "centroid", "linearprobe"],
    index=0,
    help="artifacts 파일이 있으면 centroid / linearprobe 사용 가능",
)
map_type = st.sidebar.selectbox("지도 타입", ["HYBRID", "SKYVIEW"], index=0)
level = st.sidebar.slider("지도 확대 수준(level)", min_value=1, max_value=8, value=3)

st.sidebar.markdown("---")
st.sidebar.subheader("모델 파일 상태")
artifact_exists = {
    "linearprobe.joblib": LINEARPROBE_PATH.exists(),
    "centroids.npz": CENTROIDS_PATH.exists(),
}
for name, ok in artifact_exists.items():
    st.sidebar.write(f"- {name}: {'있음' if ok else '없음'}")

# Input tabs
tab1, tab2 = st.tabs(["주소 입력", "GPS 좌표 입력"])

resolved_lat = None
resolved_lng = None
resolved_text = None
resolved_meta = None

with tab1:
    address_query = st.text_input("주소", placeholder="예: 서울특별시 중구 세종대로 110")
    if st.button("주소 검색", use_container_width=True):
        if not address_query.strip():
            st.warning("주소를 입력하세요.")
        else:
            try:
                result = geocode_address(rest_key, address_query.strip())
                if result is None:
                    st.warning("검색 결과가 없습니다.")
                else:
                    resolved_lat, resolved_lng, resolved_meta = result
                    resolved_text = address_query.strip()
                    st.session_state["lat"] = resolved_lat
                    st.session_state["lng"] = resolved_lng
                    st.session_state["resolved_text"] = resolved_text
                    st.session_state["resolved_meta"] = resolved_meta
            except Exception as e:
                st.error(f"주소 검색 중 오류: {e}")

with tab2:
    lat_input = st.text_input("위도 (Latitude)", value=str(st.session_state.get("lat", "")))
    lng_input = st.text_input("경도 (Longitude)", value=str(st.session_state.get("lng", "")))
    if st.button("좌표 확인", use_container_width=True):
        try:
            resolved_lat = float(lat_input)
            resolved_lng = float(lng_input)
            st.session_state["lat"] = resolved_lat
            st.session_state["lng"] = resolved_lng
            rev = reverse_geocode(rest_key, resolved_lat, resolved_lng)
            st.session_state["resolved_meta"] = rev
            st.session_state["resolved_text"] = "좌표 직접 입력"
        except Exception as e:
            st.error(f"좌표 처리 중 오류: {e}")

resolved_lat = st.session_state.get("lat")
resolved_lng = st.session_state.get("lng")
resolved_text = st.session_state.get("resolved_text")
resolved_meta = st.session_state.get("resolved_meta")

col1, col2 = st.columns([1.15, 0.85], gap="large")

with col1:
    st.subheader("1) 지도 보기")
    if resolved_lat is not None and resolved_lng is not None:
        st.write(f"**위도 / 경도**: {resolved_lat:.6f}, {resolved_lng:.6f}")
        if resolved_meta:
            st.json(resolved_meta, expanded=False)

        map_html = build_kakao_map_html(
            js_key=js_key,
            lat=resolved_lat,
            lng=resolved_lng,
            level=level,
            map_type=map_type,
        )
        components.html(map_html, height=650, scrolling=False)
    else:
        st.info("먼저 주소 검색 또는 좌표 입력을 해주세요.")

with col2:
    st.subheader("2) 위성 이미지 업로드 후 분류")
    st.markdown(
        """
        카카오맵 또는 다른 위성지도에서 **대상 위치의 상세 위성 이미지**를 캡처해서 업로드하세요.  
        자동 캡처 기능은 배포 안정성을 위해 제외했습니다.
        """
    )

    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="업로드한 분석 이미지", use_container_width=True)

        classify_btn = st.button("분류 실행", type="primary", use_container_width=True)
        if classify_btn:
            try:
                with st.spinner("모델 로딩 및 분류 중..."):
                    model, preprocess, tokenizer, device = load_clip_model()
                    artifacts = load_artifacts()
                    result = classify_image(
                        pil_img=pil_img,
                        mode=mode,
                        model=model,
                        preprocess=preprocess,
                        tokenizer=tokenizer,
                        device=device,
                        artifacts=artifacts,
                    )

                prob = result["probability"]
                label = result["label"]
                score = result["score"]

                if label == "데이터센터":
                    st.success(f"판정: **{label}**")
                else:
                    st.info(f"판정: **{label}**")

                st.metric("데이터센터 확률", f"{prob * 100:.2f}%")
                st.write(f"점수(score): `{score:.6f}`")
                st.write(f"모드: `{result['mode']}`")

                if result.get("details"):
                    with st.expander("세부 점수", expanded=False):
                        st.json(result["details"], expanded=True)

            except Exception as e:
                st.error(f"분류 중 오류: {e}")
    else:
        st.caption("이미지를 업로드하면 분류 버튼이 활성화됩니다.")

st.markdown("---")
st.subheader("3) 배포 전 체크리스트")
st.markdown(
    """
    - `packages.txt`는 제거
    - `requirements.txt`에는 `playwright` 제외
    - Streamlit Cloud의 **Secrets**에 아래 값 설정
      - `KAKAO_JS_KEY`
      - `KAKAO_REST_KEY`
    - `artifacts/linearprobe.joblib` 또는 `artifacts/centroids.npz`가 있으면 더 안정적인 분류 가능
    """
)
