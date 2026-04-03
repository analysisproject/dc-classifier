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
# General helpers
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


def init_single_session_state():
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
    httpd.allow_reuse_address = True

    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True
    )
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

    page.on("console", lambda msg: print(f"[BROWSER:{msg.type}] {msg.text}"))
    page.on("pageerror", lambda exc: print(f"[PAGEERROR] {exc}"))

    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    page.wait_for_function(
        "typeof kakao !== 'undefined' && typeof kakao.maps !== 'undefined'",
        timeout=30000
    )

    page.wait_for_function(
        "window.__MAP_READY__ === true",
        timeout=30000
    )

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
    compute_wide: bool = True,
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

    wide_result = None
    if compute_wide:
        wide_img = images["wide"]
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

    row = {
        "latitude": lat,
        "longitude": lng,
        "resolved_address": address_text,
        "roof_label": roof_result["label"],
        "roof_probability": float(roof_result["probability"]),
        "roof_score": float(roof_result["score"]),
        "final_label": roof_result["label"],
        "final_probability": float(roof_result["probability"]),
        "mode": mode,
        "error": None,
    }

    if compute_wide and wide_result is not None:
        row["wide_label"] = wide_result["label"]
        row["wide_probability"] = float(wide_result["probability"])
        row["wide_score"] = float(wide_result["score"])
    else:
        row["wide_label"] = None
        row["wide_probability"] = None
        row["wide_score"] = None

    return row


def read_batch_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def detect_lat_lng_name_columns(df: pd.DataFrame):
    normalized_cols = {c.lower().strip(): c for c in df.columns}

    lat_col = normalized_cols.get("latitude") or normalized_cols.get("lat")
    lng_col = (
        normalized_cols.get("longitude")
        or normalized_cols.get("lng")
        or normalized_cols.get("lon")
        or normalized_cols.get("long")
    )
    name_col = normalized_cols.get("name")

    return lat_col, lng_col, name_col


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


def render_shared_sidebar(page_title: str):
    default_js_key = os.getenv("KAKAO_JS_KEY", "")
    default_rest_key = os.getenv("KAKAO_REST_KEY", "")

    with st.sidebar:
        st.header(page_title)
        js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
        rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")
        mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
        map_type = st.selectbox("지도 타입", ["SKYVIEW", "HYBRID"], index=0)
        wide_level = st.slider("wide level", 0, 6, 2)
        roof_level = st.slider("roof level", 0, 6, 1)

        st.markdown("---")
        st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
        st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

    return js_key, rest_key, mode, map_type, wide_level, roof_level
