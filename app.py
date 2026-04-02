import io
import math
import os
import subprocess
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

try:
except Exception:
    sync_playwright = None

st.set_page_config(page_title="KakaoMap Data Center Classifier", layout="wide")

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
DEFAULT_JS_KEY = "be08d23343801208f920082c12ebd18f"
DEFAULT_REST_KEY = "2462cfd32ba9f050882d01198fdf3c8f"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

DEFAULT_POS_PROMPTS = [
    "a satellite image of a data center",
    "an aerial photo of a data center facility",
    "aerial view of a data center campus",
    "a satellite image of a large industrial building with cooling equipment and secure perimeter",
    "a satellite image of a server farm building",
]

DEFAULT_NEG_PROMPTS = [
    "a satellite image of a warehouse",
    "an aerial photo of a logistics center",
    "aerial view of a factory",
    "a satellite image of a residential block",
    "a satellite image of an office building",
]


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


KAKAO_JS_KEY = get_secret("KAKAO_JS_KEY", DEFAULT_JS_KEY)
KAKAO_REST_KEY = get_secret("KAKAO_REST_KEY", DEFAULT_REST_KEY)


@st.cache_data(show_spinner=False)
def geocode_address(rest_api_key: str, address: str) -> Dict[str, Optional[str]]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {rest_api_key}"}
    params = {"query": address}
    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
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


@st.cache_data(show_spinner=False)
def coord_to_address(rest_api_key: str, lat: float, lng: float) -> Dict[str, Optional[str]]:
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = {"Authorization": f"KakaoAK {rest_api_key}"}
    params = {"x": lng, "y": lat}
    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    docs = data.get("documents", [])
    if not docs:
        return {"address_name": None, "road_address": None}
    top = docs[0]
    return {
        "address_name": (top.get("address") or {}).get("address_name"),
        "road_address": (top.get("road_address") or {}).get("address_name"),
    }



def build_kakao_map_html(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 2,
    width: int = 1200,
    height: int = 720,
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
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:#fff; }}
    #map {{ width:{width}px; height:{height}px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={js_key}"></script>
  <script>
    const center = new kakao.maps.LatLng({lat}, {lng});
    const options = {{ center: center, level: {level} }};
    const map = new kakao.maps.Map(document.getElementById('map'), options);
    map.setMapTypeId(kakao.maps.MapTypeId.{map_type});
    const mapTypeControl = new kakao.maps.MapTypeControl();
    map.addControl(mapTypeControl, kakao.maps.ControlPosition.TOPRIGHT);
    const zoomControl = new kakao.maps.ZoomControl();
    map.addControl(zoomControl, kakao.maps.ControlPosition.RIGHT);
    {marker_js}
  </script>
</body>
</html>
"""



def render_map_component(js_key: str, lat: float, lng: float, level: int, height: int = 560) -> None:
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



def ensure_playwright_browser() -> Tuple[bool, str]:
    if sync_playwright is None:
        return False, "playwright 패키지가 로드되지 않았습니다."

    browser_hint = Path.home() / ".cache" / "ms-playwright"
    if browser_hint.exists() and any(browser_hint.iterdir()):
        return True, ""

    try:
        subprocess.run(
            ["python", "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return True, ""
    except Exception as exc:
        return False, f"Chromium 설치 실패: {exc}"


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
    ok, message = ensure_playwright_browser()
    if not ok:
        raise RuntimeError(message)

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
        html_path = Path(td) / "map.html"
        png_path = Path(td) / "map.png"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.wait_for_timeout(2500)
            page.locator("#map").screenshot(path=str(png_path))
            browser.close()
        return png_path.read_bytes()


@st.cache_resource(show_spinner=False)
def load_clip_model(model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_image_bytes(image_bytes: bytes, model, preprocess, device: torch.device) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    features = model.encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().numpy()[0]


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: torch.device) -> np.ndarray:
    tokens = tokenizer(texts).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().numpy()



def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))



def predict_zeroshot(
    emb: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> Dict[str, float]:
    pos_text = encode_texts(model, tokenizer, pos_prompts, device)
    neg_text = encode_texts(model, tokenizer, neg_prompts, device)
    sim_pos = emb @ pos_text.T
    sim_neg = emb @ neg_text.T
    raw_score = float(sim_pos.max() - sim_neg.max())
    probability = float(sigmoid(raw_score * 5.0))
    return {
        "label": int(probability >= 0.5),
        "probability": probability,
        "raw_score": raw_score,
        "max_pos_similarity": float(sim_pos.max()),
        "max_neg_similarity": float(sim_neg.max()),
        "mode": "zeroshot",
    }



def predict_centroid(emb: np.ndarray, centroid_path: Path) -> Dict[str, float]:
    data = np.load(centroid_path)
    pos_centroid = data["pos_centroid"]
    neg_centroid = data["neg_centroid"]
    sim_pos = float(emb @ pos_centroid)
    sim_neg = float(emb @ neg_centroid)
    raw_score = sim_pos - sim_neg
    probability = float(sigmoid(raw_score * 8.0))
    return {
        "label": int(probability >= 0.5),
        "probability": probability,
        "raw_score": raw_score,
        "max_pos_similarity": sim_pos,
        "max_neg_similarity": sim_neg,
        "mode": "centroid",
    }



def predict_linearprobe(emb: np.ndarray, model_path: Path) -> Dict[str, float]:
    clf = joblib.load(model_path)
    probability = float(clf.predict_proba([emb])[0, 1])
    raw_score = float(clf.decision_function([emb])[0]) if hasattr(clf, "decision_function") else probability
    return {
        "label": int(probability >= 0.5),
        "probability": probability,
        "raw_score": raw_score,
        "mode": "linearprobe",
    }



def check_artifacts() -> Dict[str, bool]:
    return {
        "centroid": (ARTIFACTS_DIR / "centroids.npz").exists(),
        "linearprobe": (ARTIFACTS_DIR / "linearprobe.joblib").exists(),
    }



def parse_coordinate_inputs(lat_str: str, lng_str: str) -> Tuple[float, float]:
    lat = float(lat_str.strip())
    lng = float(lng_str.strip())
    if not (-90 <= lat <= 90):
        raise ValueError("위도는 -90~90 범위여야 합니다.")
    if not (-180 <= lng <= 180):
        raise ValueError("경도는 -180~180 범위여야 합니다.")
    return lat, lng


st.title("KakaoMap 기반 데이터센터 분류 대시보드")
st.caption("주소 또는 GPS 좌표를 입력하면 카카오 위성지도를 캡처하고, CLIP 기반 모델로 데이터센터 여부를 예측합니다.")

with st.sidebar:
    st.header("입력 설정")
    input_mode = st.radio("입력 방식", ["주소", "GPS 좌표"], horizontal=True)
    level = st.slider("지도 확대 레벨", min_value=0, max_value=14, value=2)
    inference_mode = st.selectbox(
        "분류 방식",
        ["zeroshot", "centroid", "linearprobe"],
        index=0,
        help="centroid/linearprobe는 artifacts 폴더에 학습 결과 파일이 있어야 동작합니다.",
    )

    st.markdown("---")
    st.subheader("API 키")
    st.text_input("Kakao JavaScript Key", value=KAKAO_JS_KEY, key="js_key")
    st.text_input("Kakao REST API Key", value=KAKAO_REST_KEY, key="rest_key")

    st.markdown("---")
    st.subheader("Zero-shot Prompt")
    pos_prompt_text = st.text_area("Positive prompts", value="\n".join(DEFAULT_POS_PROMPTS), height=140)
    neg_prompt_text = st.text_area("Negative prompts", value="\n".join(DEFAULT_NEG_PROMPTS), height=140)

artifact_status = check_artifacts()
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("centroid artifact", "있음" if artifact_status["centroid"] else "없음")
info_col2.metric("linearprobe artifact", "있음" if artifact_status["linearprobe"] else "없음")
info_col3.metric("device", "CUDA" if torch.cuda.is_available() else "CPU")

query_address = ""
lat = lng = None

if input_mode == "주소":
    query_address = st.text_input("주소 입력", placeholder="예: 경기도 성남시 분당구 불정로 6")
else:
    col_lat, col_lng = st.columns(2)
    with col_lat:
        lat_input = st.text_input("위도 (lat)", value="37.3947")
    with col_lng:
        lng_input = st.text_input("경도 (lng)", value="127.1112")

run_button = st.button("지도 불러오고 분류하기", type="primary", use_container_width=True)

if run_button:
    js_key = st.session_state["js_key"].strip()
    rest_key = st.session_state["rest_key"].strip()
    pos_prompts = [x.strip() for x in pos_prompt_text.splitlines() if x.strip()]
    neg_prompts = [x.strip() for x in neg_prompt_text.splitlines() if x.strip()]

    try:
        if not js_key:
            raise ValueError("JavaScript Key가 필요합니다.")

        if input_mode == "주소":
            if not rest_key:
                raise ValueError("주소 검색에는 REST API Key가 필요합니다.")
            if not query_address.strip():
                raise ValueError("주소를 입력해 주세요.")
            geo = geocode_address(rest_key, query_address.strip())
            lat = float(geo["lat"])
            lng = float(geo["lng"])
            resolved_address = geo.get("road_address") or geo.get("address_name") or query_address.strip()
        else:
            lat, lng = parse_coordinate_inputs(lat_input, lng_input)
            resolved = coord_to_address(rest_key, lat, lng) if rest_key else {"road_address": None, "address_name": None}
            resolved_address = resolved.get("road_address") or resolved.get("address_name") or f"{lat}, {lng}"

        st.success(f"대상 위치: {resolved_address}")

        map_col, image_col = st.columns([1.2, 1.0])
        with map_col:
            st.subheader("카카오맵 상세 위성지도")
            render_map_component(js_key, lat, lng, level, height=540)

        with st.spinner("카카오 위성지도를 캡처하는 중입니다..."):
            image_bytes = capture_kakao_map(js_key, lat, lng, level=level, map_type="SKYVIEW")

        with image_col:
            st.subheader("분류에 사용된 캡처 이미지")
            st.image(image_bytes, use_container_width=True)

        model, preprocess, tokenizer, device = load_clip_model()
        emb = encode_image_bytes(image_bytes, model, preprocess, device)

        if inference_mode == "zeroshot":
            pred = predict_zeroshot(emb, model, tokenizer, device, pos_prompts, neg_prompts)
        elif inference_mode == "centroid":
            centroid_path = ARTIFACTS_DIR / "centroids.npz"
            if not centroid_path.exists():
                raise FileNotFoundError("artifacts/centroids.npz 파일이 없습니다. prepare_assets.py로 생성해 주세요.")
            pred = predict_centroid(emb, centroid_path)
        else:
            linearprobe_path = ARTIFACTS_DIR / "linearprobe.joblib"
            if not linearprobe_path.exists():
                raise FileNotFoundError("artifacts/linearprobe.joblib 파일이 없습니다. prepare_assets.py로 생성해 주세요.")
            pred = predict_linearprobe(emb, linearprobe_path)

        st.markdown("---")
        result_col1, result_col2, result_col3 = st.columns(3)
        label_text = "데이터센터 가능성 높음" if pred["label"] == 1 else "데이터센터 가능성 낮음"
        result_col1.metric("예측 결과", label_text)
        result_col2.metric("확률", f"{pred['probability'] * 100:.2f}%")
        result_col3.metric("모드", pred["mode"])

        with st.expander("상세 점수 보기", expanded=True):
            st.json({
                "lat": lat,
                "lng": lng,
                "resolved_address": resolved_address,
                **pred,
            })

    except Exception as exc:
        st.error(str(exc))

st.markdown("---")
with st.expander("GitHub / 배포 안내", expanded=False):
    st.markdown(
        """
- 공개 저장소에 실제 API 키를 올리면 노출됩니다.
- 로컬에서는 `.streamlit/secrets.toml`을 사용하고, Streamlit Community Cloud에서는 **App settings → Secrets**에 같은 키를 넣으면 됩니다.
- `centroid`, `linearprobe` 모드를 쓰려면 `artifacts/centroids.npz`, `artifacts/linearprobe.joblib`가 필요합니다.
        """
    )
