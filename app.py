import os
import threading
import tempfile
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from urllib.parse import urlencode
from typing import Optional, Tuple, Dict, Any, List
from io import BytesIO

import joblib
import numpy as np
import open_clip
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
    page = browser.new_page()
# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Satellite Data Center Classifier",
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

DEFAULT_LAT = 36.508393
DEFAULT_LNG = 127.340573

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

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Skyview Capture</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #ffffff;
      overflow: hidden;
    }
    #map {
      width: 100vw;
      height: 100vh;
    }
  </style>
  <script src="https://dapi.kakao.com/v2/maps/sdk.js?appkey=__KAKAO_JS_KEY__&autoload=false"></script>
</head>
<body>
  <div id="map"></div>

  <script>
    function getParam(name, fallback) {
      const url = new URL(window.location.href);
      const v = url.searchParams.get(name);
      return v === null ? fallback : v;
    }

    const lat = parseFloat(getParam("lat", "37.5665"));
    const lon = parseFloat(getParam("lon", "126.9780"));
    const level = parseInt(getParam("level", "2"), 10);
    const mapType = getParam("mapType", "SKYVIEW");

    window.__MAP_READY__ = false;
    window.__MAP_ERROR__ = null;

    kakao.maps.load(function () {
      try {
        const container = document.getElementById("map");
        const options = {
          center: new kakao.maps.LatLng(lat, lon),
          level: level
        };

        const map = new kakao.maps.Map(container, options);

        if (mapType === "SKYVIEW") {
          map.setMapTypeId(kakao.maps.MapTypeId.SKYVIEW);
        } else if (mapType === "HYBRID") {
          map.setMapTypeId(kakao.maps.MapTypeId.HYBRID);
        } else {
          map.setMapTypeId(kakao.maps.MapTypeId.ROADMAP);
        }

        map.setDraggable(false);
        map.setZoomable(false);

        const markerPosition = new kakao.maps.LatLng(lat, lon);
        const marker = new kakao.maps.Marker({ position: markerPosition });
        marker.setMap(map);

        let fired = false;
        function markReadyOnce() {
          if (fired) return;
          fired = true;
          setTimeout(() => {
            window.__MAP_READY__ = true;
          }, 1200);
        }

        kakao.maps.event.addListener(map, "tilesloaded", markReadyOnce);
        kakao.maps.event.addListener(map, "idle", markReadyOnce);
        setTimeout(markReadyOnce, 5000);

      } catch (e) {
        window.__MAP_ERROR__ = String(e);
      }
    });
  </script>
</body>
</html>
"""

# ============================================================
# Helpers
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
        model_name,
        pretrained=pretrained
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
        result["reason_text"] = (
            f"Linear probe가 데이터센터 확률을 {proba:.4f}로 추정했습니다. "
            f"기준값 0.5 {'이상' if proba >= 0.5 else '미만'}이므로 "
            f"{'데이터센터' if proba >= 0.5 else '비데이터센터'}로 판정했습니다."
        )
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
            "sim_pos_centroid": sim_pos,
            "sim_neg_centroid": sim_neg,
            "margin": score,
        }
        result["reason_text"] = (
            f"positive centroid 유사도={sim_pos:.4f}, "
            f"negative centroid 유사도={sim_neg:.4f}, "
            f"margin={score:.4f} 입니다."
        )
        return result

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
    result["reason_text"] = (
        f"최고 positive prompt='{POS_PROMPTS[pos_idx]}' ({pos_max:.4f}), "
        f"최고 negative prompt='{NEG_PROMPTS[neg_idx]}' ({neg_max:.4f}), "
        f"margin={score:.4f} 입니다."
    )
    return result


# ============================================================
# Kakao Local API
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
        raise RuntimeError("주소 검색이 403으로 거부되었습니다. REST API Key와 Kakao 설정을 확인하세요.")
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
        raise RuntimeError("좌표→주소 변환이 403으로 거부되었습니다. REST API Key와 Kakao 설정을 확인하세요.")
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
# Local HTTP server for Kakao rendering
# ============================================================
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_server(directory: Path, host: str, port: int):
    handler = partial(QuietHandler, directory=str(directory))
    httpd = ThreadingHTTPServer((host, port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def render_one(page, base_url: str, lat: float, lon: float, level: int, out_path: Path, map_type: str):
    params = urlencode({
        "lat": lat,
        "lon": lon,
        "level": level,
        "mapType": map_type,
    })
    url = f"{base_url}?{params}"

    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_function(
        "typeof kakao !== 'undefined' && typeof kakao.maps !== 'undefined'",
        timeout=30000
    )
    page.wait_for_function("window.__MAP_READY__ === true", timeout=30000)

    err = page.evaluate("window.__MAP_ERROR__")
    if err:
        raise RuntimeError(f"Kakao map render error: {err}")

    page.locator("#map").screenshot(path=str(out_path))


def capture_kakao_satellite_http(
    js_key: str,
    lat: float,
    lon: float,
    wide_level: int = 2,
    roof_level: int = 1,
    map_type: str = "SKYVIEW",
    width: int = 1600,
    height: int = 900,
    host: str = "127.0.0.1",
    port: int = 0,
) -> Dict[str, Image.Image]:
    if not js_key:
        raise RuntimeError("KAKAO_JS_KEY가 없습니다.")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        html = HTML_TEMPLATE.replace("__KAKAO_JS_KEY__", js_key)
        (tmpdir / "index.html").write_text(html, encoding="utf-8")

        httpd = start_server(tmpdir, host, port)
        actual_port = httpd.server_address[1]
        base_url = f"http://{host}:{actual_port}/index.html"

        wide_path = tmpdir / f"wide_z{wide_level}.png"
        roof_path = tmpdir / f"roof_z{roof_level}.png"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"]
                )
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                page = context.new_page()

                render_one(page, base_url, lat, lon, wide_level, wide_path, map_type)
                render_one(page, base_url, lat, lon, roof_level, roof_path, map_type)

                context.close()
                browser.close()
        finally:
            httpd.shutdown()
            httpd.server_close()

        return {
            "wide": Image.open(wide_path).convert("RGB"),
            "roof": Image.open(roof_path).convert("RGB"),
        }


# ============================================================
# Batch helpers
# ============================================================
def analyze_one_coordinate(
    js_key: str,
    rest_key: Optional[str],
    lat: float,
    lng: float,
    mode: str,
    map_type: str,
    wide_level: int,
    roof_level: int,
    model,
    preprocess,
    tokenizer,
    device: str,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    images = capture_kakao_satellite_http(
        js_key=js_key,
        lat=lat,
        lon=lng,
        wide_level=wide_level,
        roof_level=roof_level,
        map_type=map_type,
        width=1600,
        height=900,
    )

    wide_img = images["wide"]
    roof_img = images["roof"]

    roof_result = classify_pil_image(
        pil_img=roof_img,
        mode=mode,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        artifacts=artifacts,
    )

    wide_result = classify_pil_image(
        pil_img=wide_img,
        mode=mode,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        artifacts=artifacts,
    )

    address_text = None
    if rest_key:
        try:
            rev = reverse_geocode(rest_key, lat, lng)
            address_text = format_reverse_address(rev)
        except Exception:
            address_text = None

    return {
        "latitude": lat,
        "longitude": lng,
        "resolved_address": address_text,
        "roof_label": roof_result["label"],
        "roof_probability": float(roof_result["probability"]),
        "roof_score": float(roof_result["score"]),
        "wide_label": wide_result["label"],
        "wide_probability": float(wide_result["probability"]),
        "wide_score": float(wide_result["score"]),
        "final_label": roof_result["label"],
        "final_probability": float(roof_result["probability"]),
        "mode": mode,
    }


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


# ============================================================
# Session state
# ============================================================
defaults = {
    "lat": DEFAULT_LAT,
    "lng": DEFAULT_LNG,
    "resolved_text": "GPS 좌표 입력",
    "resolved_meta": None,
    "resolved_address_str": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# UI
# ============================================================
st.title("🛰️ Satellite Data Center Classifier")
st.caption("GPS 또는 주소를 입력하면 위성사진을 불러오고 roof view 기준으로 데이터센터 여부를 판정합니다.")

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

with st.sidebar:
    st.header("설정")
    js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
    rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")

    input_mode = st.radio("입력 방식", ["GPS 입력", "주소 입력"], index=0)
    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
    map_type = st.selectbox("지도 타입", ["SKYVIEW", "HYBRID"], index=0)
    wide_level = st.slider("wide level", 0, 6, 2)
    roof_level = st.slider("roof level", 0, 6, 1)

    st.markdown("---")
    st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
    st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

    st.markdown("---")
    st.subheader("엑셀 일괄 분석")
    uploaded_file = st.file_uploader(
        "엑셀 업로드 (.xlsx, .csv)",
        type=["xlsx", "csv"]
    )

col1, col2, col3 = st.columns([0.9, 1.05, 1.05], gap="large")

roof_result = None
wide_result = None
wide_img = None
roof_img = None

with col1:
    st.subheader("1) 위치 입력")

    if input_mode == "GPS 입력":
        lat_text = st.text_input("위도 (Latitude)", value=f"{st.session_state['lat']:.6f}")
        lng_text = st.text_input("경도 (Longitude)", value=f"{st.session_state['lng']:.6f}")

        if st.button("위치 확인 및 분석", type="primary", use_container_width=True):
            try:
                lat = float(lat_text)
                lng = float(lng_text)

                st.session_state["lat"] = lat
                st.session_state["lng"] = lng
                st.session_state["resolved_text"] = "GPS 좌표 입력"

                if rest_key:
                    rev = reverse_geocode(rest_key, lat, lng)
                    st.session_state["resolved_meta"] = rev
                    st.session_state["resolved_address_str"] = format_reverse_address(rev)
                else:
                    st.session_state["resolved_meta"] = None
                    st.session_state["resolved_address_str"] = None

            except Exception as e:
                st.error(f"GPS 좌표 입력 오류: {e}")

    else:
        address = st.text_input("주소", placeholder="예: 세종특별자치시 도움6로 11")

        if st.button("위치 확인 및 분석", type="primary", use_container_width=True):
            if not rest_key:
                st.error("주소 입력 모드에서는 REST API Key가 필요합니다.")
            elif not address.strip():
                st.warning("주소를 입력하세요.")
            else:
                try:
                    geo = geocode_address(rest_key, address.strip())
                    if geo is None:
                        st.warning("주소 검색 결과가 없습니다.")
                    else:
                        lat, lng, meta = geo
                        st.session_state["lat"] = lat
                        st.session_state["lng"] = lng
                        st.session_state["resolved_text"] = address.strip()
                        st.session_state["resolved_meta"] = meta
                        st.session_state["resolved_address_str"] = meta.get("address_name", address.strip())
                except Exception as e:
                    st.error(f"주소 검색 오류: {e}")

    st.markdown("---")
    st.subheader("2) 현재 선택 위치")
    st.write(f"**위도 / 경도**: {st.session_state['lat']:.6f}, {st.session_state['lng']:.6f}")
    if st.session_state.get("resolved_text"):
        st.write(f"**입력값**: {st.session_state['resolved_text']}")
    if st.session_state.get("resolved_address_str"):
        st.write(f"**주소**: {st.session_state['resolved_address_str']}")
    if st.session_state.get("resolved_meta") is not None:
        with st.expander("상세 위치 정보", expanded=False):
            st.json(st.session_state["resolved_meta"], expanded=False)

with col2:
    st.subheader("3) 위성 사진")

    try:
        with st.spinner("위성사진 렌더링 중..."):
            images = capture_kakao_satellite_http(
                js_key=js_key,
                lat=st.session_state["lat"],
                lon=st.session_state["lng"],
                wide_level=wide_level,
                roof_level=roof_level,
                map_type=map_type,
                width=1600,
                height=900,
            )

        wide_img = images["wide"]
        roof_img = images["roof"]

        st.markdown("**wide view**")
        st.image(wide_img, use_container_width=True)

        st.markdown("**roof view**")
        st.image(roof_img, use_container_width=True)

        with st.spinner("모델 분석 중..."):
            model, preprocess, tokenizer, device = load_clip_model()
            artifacts = load_artifacts()

            roof_result = classify_pil_image(
                pil_img=roof_img,
                mode=mode,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                artifacts=artifacts,
            )

            wide_result = classify_pil_image(
                pil_img=wide_img,
                mode=mode,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
                artifacts=artifacts,
            )

    except Exception as e:
        st.error(f"분석 중 오류: {e}")
        roof_result = None
        wide_result = None
        wide_img = None
        roof_img = None

with col3:
    st.subheader("4) 최종 판정")

    try:
        if roof_result is not None:
            final_prob = float(roof_result["probability"])
            final_label = roof_result["label"]
            roof_score = float(roof_result["score"])

            wide_prob = float(wide_result["probability"]) if wide_result is not None else None
            wide_score = float(wide_result["score"]) if wide_result is not None else None
            wide_label = wide_result["label"] if wide_result is not None else None

            if roof_result["mode"] in ["zeroshot", "centroid"]:
                roof_score_label = "roof 판정 margin (유사도 차이)"
                roof_score_display = f"{roof_score:.6f}"
                roof_score_text = f"margin {roof_score:.4f}"
            else:
                roof_score_label = "roof 데이터센터 예측 확률(score)"
                roof_score_display = f"{roof_score * 100:.2f}%"
                roof_score_text = f"예측확률(score) {roof_score:.4f}"

            if wide_result is not None:
                if wide_result["mode"] in ["zeroshot", "centroid"]:
                    wide_score_label = "wide 판정 margin (유사도 차이)"
                    wide_score_display = f"{wide_score:.6f}"
                    wide_score_text = f"margin {wide_score:.4f}"
                else:
                    wide_score_label = "wide 데이터센터 예측 확률(score)"
                    wide_score_display = f"{wide_score * 100:.2f}%"
                    wide_score_text = f"예측확률(score) {wide_score:.4f}"

            if final_label == "데이터센터":
                st.success(f"판정: **{final_label}**")
            else:
                st.info(f"판정: **{final_label}**")

            st.metric("roof 기준 데이터센터 확률", f"{final_prob * 100:.2f}%")
            st.write(f"**{roof_score_label}**: `{roof_score_display}`")
            st.write(f"**분류 모드**: `{roof_result['mode']}`")

            if wide_result is not None:
                st.write(f"**wide 결과 라벨**: `{wide_label}`")
                st.write(f"**wide 기준 데이터센터 확률**: `{wide_prob * 100:.2f}%`")
                st.write(f"**{wide_score_label}**: `{wide_score_display}`")

            st.markdown("**해석**")
            if wide_result is not None:
                st.write(
                    f"최종 판정은 **roof view 단일 결과**를 기준으로 했습니다. "
                    f"roof 결과는 **{final_label}** 이고, "
                    f"roof 기준 데이터센터 확률은 **{final_prob:.4f}**, "
                    f"내부 판정값은 **{roof_score_text}** 입니다. "
                    f"반면 wide 결과는 **{wide_label}** 이고, "
                    f"wide 기준 데이터센터 확률은 **{wide_prob:.4f}**, "
                    f"내부 판정값은 **{wide_score_text}** 입니다.\n\n"
                    f"이처럼 roof와 wide를 함께 보는 이유는 두 이미지가 제공하는 정보의 성격이 다르기 때문입니다. "
                    f"**roof view**는 대상 건물의 지붕 형상, 건물의 평면적 배치, "
                    f"설비가 놓여 있을 가능성이 있는 구조를 더 직접적으로 보여주므로 "
                    f"건물 자체를 판별하는 데 더 적합합니다. "
                    f"반면 **wide view**는 주변 도로, 인접 건물, 산업단지나 상업지역 같은 "
                    f"입지 맥락을 더 많이 포함하므로, 대상 건물 자체보다는 주변 환경을 해석하는 참고 정보에 가깝습니다.\n\n"
                    f"따라서 현재 화면에서는 wide 결과도 함께 제시하지만, "
                    f"**최종 라벨은 roof 결과만으로 결정**했습니다. "
                    f"즉, 이번 사례에서는 roof에서 관찰되는 특징이 데이터센터 판정에 더 직접적이라고 보고, "
                    f"wide는 그 판정을 보조적으로 해석하는 역할만 하도록 설계했습니다."
                )
            else:
                st.write(
                    f"최종 판정은 **roof view 단일 결과**를 기준으로 했습니다. "
                    f"roof 결과는 **{final_label}** 이고, "
                    f"roof 기준 데이터센터 확률은 **{final_prob:.4f}**, "
                    f"내부 판정값은 **{roof_score_text}** 입니다.\n\n"
                    f"이 결과는 대상 건물의 지붕 구조와 평면 배치가 데이터센터형 특징에 얼마나 가까운지를 바탕으로 계산된 것입니다. "
                    f"즉, 단순히 하나의 라벨만 출력한 것이 아니라, "
                    f"모델이 데이터센터 쪽 특징과 비데이터센터 쪽 특징을 비교한 뒤 그 차이를 수치화하여 판정한 결과라고 이해하면 됩니다."
                )

            with st.expander("score 계산 방식 설명", expanded=False):
                if roof_result["mode"] == "zeroshot":
                    st.write(
                        "zeroshot에서는 score = 가장 높은 positive prompt 유사도 - "
                        "가장 높은 negative prompt 유사도 입니다."
                    )
                elif roof_result["mode"] == "centroid":
                    st.write(
                        "centroid에서는 score = positive centroid 유사도 - "
                        "negative centroid 유사도 입니다."
                    )
                else:
                    st.write(
                        "linearprobe에서는 score를 predict_proba(데이터센터 확률)와 동일하게 사용했습니다."
                    )

        else:
            st.info("위성 사진이 생성되면 최종 판정이 여기에 표시됩니다.")

    except Exception as e:
        st.error(f"최종 판정 표시 오류: {e}")

st.markdown("---")
st.header("5) 엑셀 일괄 분석")

st.write("엑셀 또는 CSV에 `latitude`, `longitude` 컬럼이 있으면 각 행별로 확률을 계산합니다. `name` 컬럼이 있으면 결과에 함께 포함됩니다.")

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)

        st.write("업로드된 데이터 미리보기")
        st.dataframe(batch_df.head(), use_container_width=True)

        normalized_cols = {c.lower().strip(): c for c in batch_df.columns}
        lat_col = normalized_cols.get("latitude") or normalized_cols.get("lat")
        lng_col = (
            normalized_cols.get("longitude")
            or normalized_cols.get("lng")
            or normalized_cols.get("lon")
            or normalized_cols.get("long")
        )
        name_col = normalized_cols.get("name") if "name" in normalized_cols else None

        if lat_col is None or lng_col is None:
            st.error("파일에 `latitude`, `longitude` 컬럼이 반드시 있어야 합니다.")
        else:
            run_batch = st.button("엑셀 일괄 분석 실행", type="primary", use_container_width=True)

            if run_batch:
                if not js_key:
                    st.error("JavaScript Key가 필요합니다.")
                else:
                    work_df = batch_df.copy()

                    work_df[lat_col] = pd.to_numeric(work_df[lat_col], errors="coerce")
                    work_df[lng_col] = pd.to_numeric(work_df[lng_col], errors="coerce")

                    valid_df = work_df.dropna(subset=[lat_col, lng_col]).copy()

                    if valid_df.empty:
                        st.warning("유효한 latitude/longitude 값이 없습니다.")
                    else:
                        model, preprocess, tokenizer, device = load_clip_model()
                        artifacts = load_artifacts()

                        results = []
                        progress = st.progress(0)
                        status = st.empty()

                        total = len(valid_df)

                        for idx, (_, row) in enumerate(valid_df.iterrows(), start=1):
                            lat = float(row[lat_col])
                            lng = float(row[lng_col])

                            try:
                                one = analyze_one_coordinate(
                                    js_key=js_key,
                                    rest_key=rest_key if rest_key else None,
                                    lat=lat,
                                    lng=lng,
                                    mode=mode,
                                    map_type=map_type,
                                    wide_level=wide_level,
                                    roof_level=roof_level,
                                    model=model,
                                    preprocess=preprocess,
                                    tokenizer=tokenizer,
                                    device=device,
                                    artifacts=artifacts,
                                )

                                if name_col is not None:
                                    one["name"] = row[name_col]

                                results.append(one)

                            except Exception as e:
                                err_row = {
                                    "latitude": lat,
                                    "longitude": lng,
                                    "resolved_address": None,
                                    "roof_label": None,
                                    "roof_probability": None,
                                    "roof_score": None,
                                    "wide_label": None,
                                    "wide_probability": None,
                                    "wide_score": None,
                                    "final_label": None,
                                    "final_probability": None,
                                    "mode": mode,
                                    "error": str(e),
                                }
                                if name_col is not None:
                                    err_row["name"] = row[name_col]
                                results.append(err_row)

                            progress.progress(idx / total)
                            status.write(f"처리 중: {idx} / {total}")

                        result_df = pd.DataFrame(results)

                        preferred_cols = []
                        if "name" in result_df.columns:
                            preferred_cols.append("name")

                        preferred_cols += [
                            "latitude",
                            "longitude",
                            "resolved_address",
                            "final_label",
                            "final_probability",
                            "roof_label",
                            "roof_probability",
                            "roof_score",
                            "wide_label",
                            "wide_probability",
                            "wide_score",
                            "mode",
                            "error",
                        ]

                        existing_cols = [c for c in preferred_cols if c in result_df.columns]
                        other_cols = [c for c in result_df.columns if c not in existing_cols]
                        result_df = result_df[existing_cols + other_cols]

                        st.success("일괄 분석 완료")
                        st.dataframe(result_df, use_container_width=True)

                        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "결과 CSV 다운로드",
                                data=csv_bytes,
                                file_name="satellite_classifier_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                        with c2:
                            try:
                                xlsx_bytes = dataframe_to_excel_bytes(result_df)
                                st.download_button(
                                    "결과 Excel 다운로드",
                                    data=xlsx_bytes,
                                    file_name="satellite_classifier_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.warning(f"Excel 다운로드 비활성화: {e}")

    except Exception as e:
        st.error(f"파일 처리 오류: {e}")
