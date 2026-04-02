import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

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
    page_title="Kakao Satellite Data Center Classifier",
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

# zeroshot prompts
POS_PROMPTS = [
    "a satellite image of a data center",
    "an aerial photo of a data center building",
    "aerial view of a large data center facility",
    "satellite view of an industrial data center campus",
    "a data center with rooftop cooling units seen from above",
    "a data center complex with utility infrastructure seen from above",
]

NEG_PROMPTS = [
    "a satellite image of a warehouse",
    "an aerial view of a factory",
    "an aerial photo of a logistics center",
    "a satellite image of an office building",
    "an aerial view of a commercial building complex",
    "a residential apartment complex seen from above",
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
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
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
def encode_texts(model, tokenizer, texts: List[str], device: str) -> np.ndarray:
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
        "reason_text": "",
    }

    # -------------------------
    # linearprobe
    # -------------------------
    if mode == "linearprobe":
        if "linearprobe" not in artifacts:
            raise RuntimeError("artifacts/linearprobe.joblib 파일이 없습니다.")

        clf = artifacts["linearprobe"]
        proba = float(clf.predict_proba(img_emb.reshape(1, -1))[0, 1])

        result["score"] = proba
        result["probability"] = proba
        result["label"] = "데이터센터" if proba >= 0.5 else "비데이터센터"
        result["details"] = {
            "predict_proba_positive": proba,
            "decision_threshold": 0.5,
        }

        if proba >= 0.5:
            result["reason_text"] = (
                f"Linear probe 분류기가 양성 클래스(데이터센터) 확률을 {proba:.4f}로 추정했습니다. "
                f"기준값 0.5 이상이므로 데이터센터로 판정했습니다."
            )
        else:
            result["reason_text"] = (
                f"Linear probe 분류기가 양성 클래스(데이터센터) 확률을 {proba:.4f}로 추정했습니다. "
                f"기준값 0.5 미만이므로 비데이터센터로 판정했습니다."
            )
        return result

    # -------------------------
    # centroid
    # -------------------------
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
        result["details"] = {
            "sim_pos_centroid": sim_pos,
            "sim_neg_centroid": sim_neg,
            "margin": score,
        }

        if score > 0:
            result["reason_text"] = (
                f"입력 이미지가 positive centroid와의 유사도({sim_pos:.4f})가 "
                f"negative centroid와의 유사도({sim_neg:.4f})보다 높았습니다. "
                f"margin={score:.4f} 이므로 데이터센터로 판정했습니다."
            )
        else:
            result["reason_text"] = (
                f"입력 이미지가 negative centroid와의 유사도({sim_neg:.4f})가 "
                f"positive centroid와의 유사도({sim_pos:.4f})보다 높았습니다. "
                f"margin={score:.4f} 이므로 비데이터센터로 판정했습니다."
            )
        return result

    # -------------------------
    # zeroshot
    # -------------------------
    text_emb_pos = encode_texts(model, tokenizer, POS_PROMPTS, device)
    text_emb_neg = encode_texts(model, tokenizer, NEG_PROMPTS, device)

    sim_pos = img_emb @ text_emb_pos.T
    sim_neg = img_emb @ text_emb_neg.T

    pos_idx = int(np.argmax(sim_pos))
    neg_idx = int(np.argmax(sim_neg))
    pos_max = float(sim_pos[pos_idx])
    neg_max = float(sim_neg[neg_idx])

    score = pos_max - neg_max
    prob = float(sigmoid(score * 8.0))

    result["score"] = score
    result["probability"] = prob
    result["label"] = "데이터센터" if score > 0 else "비데이터센터"
    result["details"] = {
        "best_positive_prompt": POS_PROMPTS[pos_idx],
        "best_positive_similarity": pos_max,
        "best_negative_prompt": NEG_PROMPTS[neg_idx],
        "best_negative_similarity": neg_max,
        "margin": score,
    }

    if score > 0:
        result["reason_text"] = (
            f"zeroshot 기준에서 가장 강한 데이터센터 프롬프트는 "
            f"'{POS_PROMPTS[pos_idx]}' 이고 유사도는 {pos_max:.4f}였습니다. "
            f"가장 강한 비데이터센터 프롬프트 "
            f"'{NEG_PROMPTS[neg_idx]}' 의 유사도 {neg_max:.4f}보다 높아 "
            f"margin={score:.4f}로 데이터센터로 판정했습니다."
        )
    else:
        result["reason_text"] = (
            f"zeroshot 기준에서 가장 강한 비데이터센터 프롬프트는 "
            f"'{NEG_PROMPTS[neg_idx]}' 이고 유사도는 {neg_max:.4f}였습니다. "
            f"가장 강한 데이터센터 프롬프트 "
            f"'{POS_PROMPTS[pos_idx]}' 의 유사도 {pos_max:.4f}보다 높아 "
            f"margin={score:.4f}로 비데이터센터로 판정했습니다."
        )
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
        raise RuntimeError("주소 검색이 403으로 거부되었습니다. REST API Key와 앱 설정을 확인하세요.")
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
    headers = get_auth_headers(rest_key)
    params = {"x": lng, "y": lat}
    r = requests.get(url, headers=headers, params=params, timeout=20)

    if r.status_code == 403:
        raise RuntimeError("좌표→주소 변환이 403으로 거부되었습니다. REST API Key와 앱 설정을 확인하세요.")
    r.raise_for_status()

    data = r.json()
    docs = data.get("documents", [])
    return docs[0] if docs else None

def format_reverse_address(doc: Optional[dict]) -> str:
    if not doc:
        return "주소를 찾지 못했습니다."
    road = doc.get("road_address")
    addr = doc.get("address")
    if road and road.get("address_name"):
        return road["address_name"]
    if addr and addr.get("address_name"):
        return addr["address_name"]
    return "주소를 찾지 못했습니다."

# ============================================================
# Kakao visible static map HTML
# 공식 Web API 문서에는 StaticMap 생성 예제가 있습니다.
# 여기서는 화면 표시에 그 방식을 사용합니다.
# ============================================================
def build_kakao_map_html(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 3,
    map_type: str = "SKYVIEW",
    width: int = 920,
    height: int = 620,
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
                background: #111;
            }}
            #map {{
                width: {width}px;
                height: {height}px;
            }}
        </style>
        <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={js_key}"></script>
    </head>
    <body>
        <div id="map"></div>
        <script>
            const container = document.getElementById('map');
            const options = {{
                center: new kakao.maps.LatLng({lat}, {lng}),
                level: {level}
            }};
            const map = new kakao.maps.Map(container, options);
            map.setMapTypeId({map_type_js});

            const markerPosition = new kakao.maps.LatLng({lat}, {lng});
            const marker = new kakao.maps.Marker({{
                position: markerPosition
            }});
            marker.setMap(map);

            const iwContent = '<div style="padding:6px 10px;font-size:12px;">분석 위치</div>';
            const infowindow = new kakao.maps.InfoWindow({{
                content: iwContent
            }});
            infowindow.open(map, marker);
        </script>
    </body>
    </html>
    """

# ============================================================
# Analysis image retrieval
# 주의:
# 공식 문서에서 브라우저용 StaticMap은 확인되지만,
# 서버 직접 이미지 다운로드용 URL은 공식 문서로 확인하지 못했습니다.
# 아래 URL 패턴은 실무에서 자주 쓰이는 형태를 사용합니다.
# 환경에 따라 차단되면 이 함수만 교체하면 됩니다.
# ============================================================
def fetch_satellite_image_for_analysis(
    lat: float,
    lng: float,
    width: int = 1024,
    height: int = 768,
    scale: int = 2,
    zoom: int = 2,
) -> Image.Image:
    url = "https://map.kakao.com/etc/getStaticMap"

    params = {
        "SCALE": scale,
        "MX": lng,
        "MY": lat,
        "WIDTH": width,
        "HEIGHT": height,
        "ZOOM": zoom,
        "FORMAT": "png",
        "SERVICE": "roadmap",
    }

    r = requests.get(url, params=params, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"정적 지도 요청 실패: status={r.status_code}")

    content_type = r.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        preview = r.text[:500] if hasattr(r, "text") else str(r.content[:200])
        raise RuntimeError(
            "정적 지도 응답이 이미지가 아닙니다. "
            f"Content-Type={content_type}, response preview={preview}"
        )

    try:
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise RuntimeError(
            f"이미지 디코딩 실패: {e}. "
            f"Content-Type={content_type}, bytes_length={len(r.content)}"
        )

# ============================================================
# Session state
# ============================================================
for k, v in {
    "lat": None,
    "lng": None,
    "resolved_text": None,
    "resolved_meta": None,
    "resolved_address_str": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# UI
# ============================================================
st.title("🛰️ Kakao Satellite Data Center Classifier")
st.caption("주소 또는 GPS 좌표 중 하나를 입력하면 위성 이미지를 기반으로 데이터센터 여부를 즉시 분류합니다.")

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

with st.sidebar:
    st.header("설정")
    js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
    rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")

    input_mode = st.radio("입력 방식", ["주소 입력", "GPS 입력"], index=0)
    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)

    map_type = st.selectbox(
        "지도 표현",
        ["SKYVIEW", "HYBRID"],
        index=0,
        help="화면 표시 기준. HYBRID는 위성 + 라벨입니다."
    )

    level = st.slider(
        "표시 확대 수준(level)",
        min_value=0,
        max_value=8,
        value=3,
        help="SKYVIEW/HYBRID 기준 확대 수준"
    )

    image_zoom = st.slider(
        "분석 이미지 zoom",
        min_value=0,
        max_value=5,
        value=2,
        help="분석용 정적 이미지의 확대 정도"
    )

    st.markdown("---")
    st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
    st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

left, right = st.columns([1.3, 0.9], gap="large")

lat = None
lng = None
meta = None
resolved_text = None
resolved_address_str = None

with left:
    st.subheader("1) 위치 입력")

    if input_mode == "주소 입력":
        address = st.text_input(
            "주소",
            placeholder="예: 세종특별자치시 도움6로 11"
        )

        if st.button("위치 확인 및 분석", type="primary", use_container_width=True):
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
                resolved_address_str = meta.get("address_name", address.strip())

                st.session_state["lat"] = lat
                st.session_state["lng"] = lng
                st.session_state["resolved_text"] = resolved_text
                st.session_state["resolved_meta"] = meta
                st.session_state["resolved_address_str"] = resolved_address_str

            except Exception as e:
                st.error(f"주소 검색 오류: {e}")
                st.stop()

    else:
        c1, c2 = st.columns(2)
        with c1:
            lat_text = st.text_input("위도 (Latitude)", placeholder="예: 36.504073")
        with c2:
            lng_text = st.text_input("경도 (Longitude)", placeholder="예: 127.249485")

        if st.button("위치 확인 및 분석", type="primary", use_container_width=True):
            try:
                lat = float(lat_text)
                lng = float(lng_text)
                resolved_text = "GPS 좌표 입력"

                st.session_state["lat"] = lat
                st.session_state["lng"] = lng
                st.session_state["resolved_text"] = resolved_text

                if rest_key:
                    reverse_meta = reverse_geocode(rest_key, lat, lng)
                    st.session_state["resolved_meta"] = reverse_meta
                    st.session_state["resolved_address_str"] = format_reverse_address(reverse_meta)
                else:
                    st.session_state["resolved_meta"] = None
                    st.session_state["resolved_address_str"] = None

            except Exception as e:
                st.error(f"GPS 좌표 입력 오류: {e}")
                st.stop()

    # session에서 현재 선택 위치 읽기
    lat = st.session_state.get("lat")
    lng = st.session_state.get("lng")
    meta = st.session_state.get("resolved_meta")
    resolved_text = st.session_state.get("resolved_text")
    resolved_address_str = st.session_state.get("resolved_address_str")

    st.markdown("---")
    st.subheader("2) 현재 선택 위치")

    if lat is not None and lng is not None:
        st.write(f"**위도 / 경도**: {lat:.6f}, {lng:.6f}")
        if resolved_text:
            st.write(f"**입력값**: {resolved_text}")
        if resolved_address_str:
            st.write(f"**주소**: {resolved_address_str}")

        if meta:
            with st.expander("상세 위치 응답", expanded=False):
                st.json(meta, expanded=False)

        st.subheader("3) 표시용 지도")
        st.caption("화면 표시는 Kakao StaticMap 방식으로 구성했습니다.")
        try:
            map_html = build_kakao_map_html(
                js_key=js_key,
                lat=lat,
                lng=lng,
                level=level,
                map_type=map_type,
                width=920,
                height=620,
            )
            components.html(map_html, height=620, scrolling=False)
        except Exception as e:
            st.error(f"지도 표시 오류: {e}")

    else:
        st.info("주소 또는 GPS 좌표 중 하나를 입력하고 분석 버튼을 누르세요.")

with right:
    st.subheader("4) 위성 이미지 및 분석 결과")

    if lat is not None and lng is not None:
        try:
            with st.spinner("분석용 위성 이미지 생성 중..."):
                sat_img = fetch_satellite_image_for_analysis(
                    lat=lat,
                    lng=lng,
                    width=1024,
                    height=768,
                    scale=2,
                    zoom=image_zoom,
                )

            st.image(sat_img, caption="분석에 사용된 위성 이미지", use_container_width=True)

            with st.spinner("모델 분석 중..."):
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

            st.markdown("### 5) 분류 결과")
            if label == "데이터센터":
                st.success(f"판정: **{label}**")
            else:
                st.info(f"판정: **{label}**")

            st.metric("데이터센터 확률", f"{prob * 100:.2f}%")
            st.write(f"**점수(score)**: `{score:.6f}`")
            st.write(f"**분류 모드**: `{result['mode']}`")

            st.markdown("### 6) 분류 근거")
            st.write(result["reason_text"])

            with st.expander("세부 점수 보기", expanded=True):
                st.json(result["details"], expanded=True)

            # 시각적 해설
            st.markdown("### 7) 해석 가이드")
            if result["mode"] == "zeroshot":
                st.markdown(
                    """
                    - positive prompt와의 유사도가 negative prompt보다 높을수록 데이터센터 쪽으로 기웁니다.
                    - `margin = best_positive_similarity - best_negative_similarity` 가 양수면 데이터센터 판정입니다.
                    """
                )
            elif result["mode"] == "centroid":
                st.markdown(
                    """
                    - 학습 데이터에서 만든 positive centroid / negative centroid와의 거리 차이를 사용합니다.
                    - `sim_pos_centroid`가 `sim_neg_centroid`보다 크면 데이터센터 쪽으로 판단합니다.
                    """
                )
            else:
                st.markdown(
                    """
                    - CLIP 이미지 임베딩을 logistic classifier에 넣은 결과입니다.
                    - `predict_proba_positive`가 0.5 이상이면 데이터센터로 분류합니다.
                    """
                )

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
    else:
        st.info("왼쪽에서 위치를 입력하면, 여기서 위성 이미지와 분류 결과가 바로 표시됩니다.")

st.markdown("---")
st.markdown(
    """
    **구성 요약**

    - 주소 입력 모드에서는 Kakao Local REST API로 좌표를 찾습니다.
    - GPS 입력 모드에서는 좌표를 직접 사용합니다.
    - 지도 타입은 Kakao Maps Web API에서 `SKYVIEW`, `HYBRID` 타입을 지원합니다. :contentReference[oaicite:2]{index=2}
    - 화면 표시는 Kakao Web API의 StaticMap 예제 구조를 따릅니다. :contentReference[oaicite:3]{index=3}
    - 분석 결과는 확률, 점수, 세부 근거까지 함께 보여줍니다.
    """
)
