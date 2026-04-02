import os
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

st.set_page_config(
    page_title="KakaoMap Data Center Classifier",
    page_icon="🛰️",
    layout="wide",
)

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

def get_secret_or_env(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass

    value = os.getenv(key, default)
    if value is not None and str(value).strip():
        return str(value).strip()
    return None

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

def get_auth_headers(rest_key: str) -> Dict[str, str]:
    if not rest_key or not rest_key.strip():
        raise ValueError("REST API 키가 비어 있습니다.")
    return {"Authorization": f"KakaoAK {rest_key.strip()}"}

def geocode_address(rest_key: str, query: str) -> Optional[Tuple[float, float, dict]]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = get_auth_headers(rest_key)
    params = {"query": query}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code == 403:
        raise RuntimeError(
            "Kakao Local API 403 Forbidden입니다. "
            "REST API 키 종류/오입력, 앱 설정, Local API 권한 상태를 확인하세요."
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

def reverse_geocode(rest_key: str, lat: float, lng: float) -> Optional[dict]:
    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    headers = get_auth_headers(rest_key)
    params = {"x": lng, "y": lat}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code == 403:
        raise RuntimeError(
            "Kakao Local API 403 Forbidden입니다. "
            "REST API 키 종류/오입력, 앱 설정, Local API 권한 상태를 확인하세요."
        )
    r.raise_for_status()
    data = r.json()
    docs = data.get("documents", [])
    return docs[0] if docs else None

def format_address_from_reverse_result(doc: Optional[dict]) -> str:
    if not doc:
        return "주소를 찾지 못했습니다."

    road = doc.get("road_address")
    addr = doc.get("address")

    if road and road.get("address_name"):
        return road["address_name"]
    if addr and addr.get("address_name"):
        return addr["address_name"]
    return "주소를 찾지 못했습니다."

def build_kakao_map_html(js_key: str, lat: float, lng: float, level: int = 3, map_type: str = "HYBRID") -> str:
    if not js_key or not js_key.strip():
        raise ValueError("JavaScript 키가 비어 있습니다.")

    map_type_js = "kakao.maps.MapTypeId.HYBRID" if map_type.upper() == "HYBRID" else "kakao.maps.MapTypeId.SKYVIEW"

    return f"""
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

st.title("🛰️ KakaoMap Data Center Classifier")
st.caption("주소 ↔ GPS 좌표 변환, 지도 표시, 위성 이미지 분류")

for k, v in {
    "lat": None,
    "lng": None,
    "resolved_text": None,
    "resolved_meta": None,
    "resolved_address_str": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.header("설정")
default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

mode = st.sidebar.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
map_type = st.sidebar.selectbox("지도 타입", ["HYBRID", "SKYVIEW"], index=0)
level = st.sidebar.slider("지도 확대 수준(level)", 1, 8, 3)

st.sidebar.markdown("---")
js_key = st.sidebar.text_input("JavaScript Key", value=default_js_key, type="password")
rest_key = st.sidebar.text_input("REST API Key", value=default_rest_key, type="password")

st.sidebar.markdown("---")
st.sidebar.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
st.sidebar.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("1) 주소로 좌표 찾기")
    address_query = st.text_input(
        "주소 입력",
        placeholder="예: 824 Haengbok-daero 또는 세종특별자치시 도움6로 11"
    )
    if st.button("주소 → 좌표 검색", use_container_width=True):
        if not rest_key:
            st.error("REST API Key를 먼저 입력하세요.")
        elif not address_query.strip():
            st.warning("주소를 입력하세요.")
        else:
            try:
                lat, lng, meta = geocode_address(rest_key, address_query.strip())
                st.session_state["lat"] = lat
                st.session_state["lng"] = lng
                st.session_state["resolved_text"] = address_query.strip()
                st.session_state["resolved_meta"] = meta
                st.session_state["resolved_address_str"] = meta.get("address_name", address_query.strip())
            except Exception as e:
                st.error(f"주소 검색 오류: {e}")

    st.subheader("2) GPS 좌표로 주소 찾기")
    c1, c2 = st.columns(2)
    with c1:
        lat_str = st.text_input("위도 (Latitude)", value="" if st.session_state["lat"] is None else str(st.session_state["lat"]))
    with c2:
        lng_str = st.text_input("경도 (Longitude)", value="" if st.session_state["lng"] is None else str(st.session_state["lng"]))

    if st.button("GPS → 주소 확인", use_container_width=True):
        if not rest_key:
            st.error("REST API Key를 먼저 입력하세요.")
        else:
            try:
                lat = float(lat_str)
                lng = float(lng_str)
                meta = reverse_geocode(rest_key, lat, lng)

                st.session_state["lat"] = lat
                st.session_state["lng"] = lng
                st.session_state["resolved_text"] = "GPS 좌표 입력"
                st.session_state["resolved_meta"] = meta
                st.session_state["resolved_address_str"] = format_address_from_reverse_result(meta)
            except Exception as e:
                st.error(f"GPS 주소 변환 오류: {e}")

    st.subheader("3) 현재 선택된 위치")
    if st.session_state["lat"] is not None and st.session_state["lng"] is not None:
        st.write(f"**위도 / 경도**: {st.session_state['lat']:.6f}, {st.session_state['lng']:.6f}")
        if st.session_state["resolved_address_str"]:
            st.write(f"**주소**: {st.session_state['resolved_address_str']}")
        if st.session_state["resolved_meta"] is not None:
            with st.expander("상세 응답 보기", expanded=False):
                st.json(st.session_state["resolved_meta"], expanded=False)

        try:
            map_html = build_kakao_map_html(
                js_key=js_key,
                lat=st.session_state["lat"],
                lng=st.session_state["lng"],
                level=level,
                map_type=map_type,
            )
            components.html(map_html, height=650, scrolling=False)
        except Exception as e:
            st.error(f"지도 표시 오류: {e}")
    else:
        st.info("주소를 입력하거나 GPS 좌표를 입력하세요.")

with right:
    st.subheader("4) 위성 이미지 업로드 후 분류")
    uploaded = st.file_uploader(
        "이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="업로드한 분석 이미지", use_container_width=True)

        if st.button("분류 실행", type="primary", use_container_width=True):
            try:
                with st.spinner("분류 중..."):
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
                st.error(f"분류 오류: {e}")

st.markdown("---")
st.markdown(
    """
    **403이 계속 뜰 때 점검**
    1. REST API Key 칸에 **REST API 키**가 들어갔는지 확인
    2. JavaScript 키를 REST 칸에 잘못 넣지 않았는지 확인
    3. Kakao Developers에서 Local/Map 관련 기능 설정이 활성화되어 있는지 확인
    4. 지도 자체가 안 뜨면 JavaScript 키의 도메인 등록 상태도 같이 확인
    """
)
