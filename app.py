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

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Kakao Satellite Data Center Classifier",
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
    "an aerial view of a factory",
    "an aerial photo of a logistics center",
    "a satellite image of an office building",
    "an aerial view of a commercial building complex",
]

# ============================================================
# Secret / env helpers
# ============================================================
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


# ============================================================
# Model helpers
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
        result["details"] = {
            "sim_pos": sim_pos,
            "sim_neg": sim_neg,
        }
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
# Kakao Local REST helpers
# ============================================================
def get_auth_headers(rest_key: str) -> Dict[str, str]:
    if not rest_key or not rest_key.strip():
        raise ValueError("REST API 키가 비어 있습니다.")
    return {
        "Authorization": f"KakaoAK {rest_key.strip()}",
    }


def geocode_address(rest_key: str, query: str) -> Optional[Tuple[float, float, dict]]:
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = get_auth_headers(rest_key)
    params = {"query": query}
    r = requests.get(url, headers=headers, params=params, timeout=20)

    if r.status_code == 403:
        raise RuntimeError(
            "주소 검색이 403으로 거부되었습니다. REST API Key가 맞는지, "
            "JavaScript Key를 잘못 넣지 않았는지 확인하세요."
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
            "GPS 주소 변환이 403으로 거부되었습니다. REST API Key가 맞는지, "
            "JavaScript Key를 잘못 넣지 않았는지 확인하세요."
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


# ============================================================
# Kakao Map HTML
# ============================================================
def build_kakao_map_html(
    js_key: str,
    lat: float,
    lng: float,
    level: int = 3,
    map_type: str = "SKYVIEW",
) -> str:
    if not js_key or not js_key.strip():
        raise ValueError("JavaScript Key가 비어 있습니다.")

    # SKYVIEW = 위성사진
    # HYBRID = 위성사진 + 라벨
    if map_type.upper() == "HYBRID":
        map_type_js = "kakao.maps.MapTypeId.HYBRID"
    else:
        map_type_js = "kakao.maps.MapTypeId.SKYVIEW"

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
                background: #000;
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

            // 반드시 위성사진으로 설정
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


# ============================================================
# Initialize session state
# ============================================================
default_state = {
    "lat": None,
    "lng": None,
    "resolved_text": None,
    "resolved_meta": None,
    "resolved_address_str": None,
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# UI
# ============================================================
st.title("🛰️ Kakao Satellite Data Center Classifier")
st.caption("GPS 좌표 또는 주소를 입력해 위성사진 지도를 띄우고, 위성 이미지 업로드로 데이터센터 여부를 분류합니다.")

# Sidebar
st.sidebar.header("설정")

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

mode = st.sidebar.selectbox(
    "분류 모드",
    ["zeroshot", "centroid", "linearprobe"],
    index=0,
)

# 기본값을 SKYVIEW로 고정
map_type = st.sidebar.selectbox(
    "지도 타입",
    ["SKYVIEW", "HYBRID"],
    index=0,
    help="SKYVIEW = 순수 위성사진, HYBRID = 위성사진 + 라벨",
)

level = st.sidebar.slider(
    "지도 확대 수준(level)",
    min_value=1,
    max_value=8,
    value=3,
    help="숫자가 작을수록 더 확대됩니다.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Kakao API Key")
js_key = st.sidebar.text_input("JavaScript Key", value=default_js_key, type="password")
rest_key = st.sidebar.text_input("REST API Key", value=default_rest_key, type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("모델 파일 상태")
st.sidebar.write(f"- linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
st.sidebar.write(f"- centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("1) GPS 좌표로 바로 위성지도 보기")

    c1, c2 = st.columns(2)
    with c1:
        lat_str = st.text_input(
            "위도 (Latitude)",
            value="" if st.session_state["lat"] is None else str(st.session_state["lat"]),
            placeholder="예: 36.504073",
        )
    with c2:
        lng_str = st.text_input(
            "경도 (Longitude)",
            value="" if st.session_state["lng"] is None else str(st.session_state["lng"]),
            placeholder="예: 127.249485",
        )

    g1, g2 = st.columns(2)

    with g1:
        if st.button("GPS로 위성지도 보기", use_container_width=True):
            try:
                lat = float(lat_str)
                lng = float(lng_str)

                st.session_state["lat"] = lat
                st.session_state["lng"] = lng
                st.session_state["resolved_text"] = "GPS 좌표 입력"
                st.session_state["resolved_meta"] = None
                st.session_state["resolved_address_str"] = None

                st.success("GPS 좌표로 위성사진 지도를 표시합니다.")
            except Exception as e:
                st.error(f"GPS 좌표 처리 오류: {e}")

    with g2:
        if st.button("GPS → 주소 변환", use_container_width=True):
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

                    st.success("GPS 좌표를 주소로 변환했습니다.")
                except Exception as e:
                    st.error(f"GPS 주소 변환 오류: {e}")

    st.markdown("---")
    st.subheader("2) 주소로 좌표 검색")

    address_query = st.text_input(
        "주소 입력",
        placeholder="예: 세종특별자치시 도움6로 11 또는 824 Haengbok-daero",
    )

    if st.button("주소 → 좌표 검색", use_container_width=True):
        if not rest_key:
            st.error("REST API Key를 먼저 입력하세요.")
        elif not address_query.strip():
            st.warning("주소를 입력하세요.")
        else:
            try:
                result = geocode_address(rest_key, address_query.strip())

                if result is None:
                    st.warning("검색 결과가 없습니다.")
                else:
                    lat, lng, meta = result
                    st.session_state["lat"] = lat
                    st.session_state["lng"] = lng
                    st.session_state["resolved_text"] = address_query.strip()
                    st.session_state["resolved_meta"] = meta
                    st.session_state["resolved_address_str"] = meta.get("address_name", address_query.strip())

                    st.success("주소를 좌표로 변환했습니다.")
            except Exception as e:
                st.error(f"주소 검색 오류: {e}")

    st.markdown("---")
    st.subheader("3) 현재 선택된 위치")

    if st.session_state["lat"] is not None and st.session_state["lng"] is not None:
        st.write(f"**위도 / 경도**: {st.session_state['lat']:.6f}, {st.session_state['lng']:.6f}")

        if st.session_state["resolved_address_str"]:
            st.write(f"**주소**: {st.session_state['resolved_address_str']}")
        else:
            st.caption("현재는 GPS 좌표만 선택된 상태입니다. 주소가 필요하면 'GPS → 주소 변환'을 누르세요.")

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
            components.html(map_html, height=680, scrolling=False)
        except Exception as e:
            st.error(f"지도 표시 오류: {e}")
    else:
        st.info("GPS 좌표를 넣거나 주소를 입력해 위치를 먼저 선택하세요.")

with right:
    st.subheader("4) 위성 이미지 업로드 후 분류")
    st.write("카카오 위성지도 또는 다른 위성지도에서 캡처한 이미지를 업로드하면 데이터센터 여부와 확률을 보여줍니다.")

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
    else:
        st.caption("먼저 위성 이미지를 업로드하세요.")

st.markdown("---")
st.markdown(
    """
    **사용 방법**
    
    1. GPS 좌표를 바로 넣고 **GPS로 위성지도 보기**를 누르면 REST API 없이도 위성사진 지도가 표시됩니다.  
    2. 주소를 알고 싶을 때만 **GPS → 주소 변환**을 누르세요. 이 단계에서만 REST API Key가 필요합니다.  
    3. 주소를 직접 넣고 찾고 싶으면 **주소 → 좌표 검색**을 사용하세요.  
    4. 분류는 지도 화면 자체를 자동 캡처하지 않으므로, 위성사진을 캡처해서 업로드해야 합니다.
    """
)

st.markdown(
    """
    **403 오류가 계속 뜰 때**
    
    - `REST API Key` 칸에 반드시 **REST API 키**를 넣었는지 확인
    - `JavaScript Key`를 `REST API Key` 칸에 잘못 넣지 않았는지 확인
    - GPS로 지도만 보는 기능은 REST API 없이 동작해야 정상
    - 주소 검색 / GPS 주소 변환에서만 403이 뜬다면 REST API 설정 문제로 좁혀집니다
    """
)
