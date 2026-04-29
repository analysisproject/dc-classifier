import streamlit as st
import pandas as pd
import re
import time

from utils.core import (
    get_secret_or_env,
    geocode_address,
    capture_kakao_satellite_http,
    load_clip_model,
    load_artifacts,
    classify_pil_image,
)


# =========================================================
# 1. 주소 컬럼 자동 인식 함수
# =========================================================
def detect_address_column(df):
    """
    엑셀/CSV 파일에서 주소 컬럼을 자동으로 찾는다.
    1차: 컬럼명 기준
    2차: 실제 값의 한국 주소 패턴 기준
    """

    address_keywords = [
        "주소", "도로명", "지번", "원주소", "지번주소", "도로명주소",
        "소재지", "위치", "사업장주소", "기관주소",
        "address", "addr", "location", "place", "site"
    ]

    # 1) 컬럼명 기준
    for col in df.columns:
        col_lower = str(col).lower().replace(" ", "")
        if any(keyword.lower().replace(" ", "") in col_lower for keyword in address_keywords):
            return col

    # 2) 값 기준
    korean_address_patterns = [
        r"서울|부산|대구|인천|광주|대전|울산|세종",
        r"경기|강원|충북|충남|전북|전남|경북|경남|제주",
        r"특별시|광역시|특별자치시|특별자치도",
        r"시\s|군\s|구\s",
        r"로\s?\d+|길\s?\d+",
        r"동\s?\d*|읍|면|리"
    ]

    best_col = None
    best_score = 0

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(30)

        score = 0
        for value in sample_values:
            for pattern in korean_address_patterns:
                if re.search(pattern, value):
                    score += 1
                    break

        if score > best_score:
            best_score = score
            best_col = col

    if best_score >= 2:
        return best_col

    return None


# =========================================================
# 2. 위도/경도 컬럼 자동 인식 함수
# =========================================================
def detect_lat_lng_columns(df):
    """
    엑셀/CSV 파일에서 latitude / longitude 컬럼을 자동으로 찾는다.
    """

    lat_keywords = [
        "latitude", "lat", "위도", "y좌표", "y", "gps_y"
    ]

    lng_keywords = [
        "longitude", "lng", "lon", "long", "경도", "x좌표", "x", "gps_x"
    ]

    lat_col = None
    lng_col = None

    for col in df.columns:
        col_lower = str(col).lower().replace(" ", "")

        if lat_col is None and any(k.lower().replace(" ", "") == col_lower for k in lat_keywords):
            lat_col = col

        if lng_col is None and any(k.lower().replace(" ", "") == col_lower for k in lng_keywords):
            lng_col = col

    # 완전 일치가 안 되면 포함 여부로 한 번 더 확인
    if lat_col is None:
        for col in df.columns:
            col_lower = str(col).lower().replace(" ", "")
            if any(k.lower().replace(" ", "") in col_lower for k in lat_keywords):
                lat_col = col
                break

    if lng_col is None:
        for col in df.columns:
            col_lower = str(col).lower().replace(" ", "")
            if any(k.lower().replace(" ", "") in col_lower for k in lng_keywords):
                lng_col = col
                break

    return lat_col, lng_col


# =========================================================
# 3. 파일 읽기 함수
# =========================================================
def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")

    return pd.read_excel(uploaded_file)


# =========================================================
# 4. 주소 → GPS 변환 함수
# =========================================================
def add_coordinates_from_address(df, address_col, rest_key, sleep_sec=0.15):
    """
    주소 컬럼을 이용해 latitude / longitude 컬럼을 생성한다.
    """

    lat_list = []
    lng_list = []
    resolved_address_list = []
    geocode_status_list = []

    progress = st.progress(0)
    status_text = st.empty()

    total = len(df)

    for i, addr in enumerate(df[address_col].astype(str)):
        progress.progress((i + 1) / total)
        status_text.write(f"주소 변환 중: {i + 1} / {total}")

        if not addr or addr.strip() == "" or addr.lower() == "nan":
            lat_list.append(None)
            lng_list.append(None)
            resolved_address_list.append(None)
            geocode_status_list.append("EMPTY_ADDRESS")
            continue

        try:
            geo = geocode_address(rest_key, addr.strip())

            if geo is None:
                lat_list.append(None)
                lng_list.append(None)
                resolved_address_list.append(None)
                geocode_status_list.append("NOT_FOUND")
            else:
                lat, lng, meta = geo
                lat_list.append(float(lat))
                lng_list.append(float(lng))
                resolved_address_list.append(meta.get("address_name", addr.strip()))
                geocode_status_list.append("OK")

        except Exception as e:
            lat_list.append(None)
            lng_list.append(None)
            resolved_address_list.append(None)
            geocode_status_list.append(f"ERROR: {e}")

        time.sleep(sleep_sec)

    df["latitude"] = lat_list
    df["longitude"] = lng_list
    df["resolved_address"] = resolved_address_list
    df["geocode_status"] = geocode_status_list

    progress.empty()
    status_text.empty()

    return df


# =========================================================
# 5. Streamlit Batch Excel Analysis 화면
# =========================================================
st.set_page_config(
    page_title="Batch Excel Analysis",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Batch Excel Analysis")
st.caption("엑셀 또는 CSV의 latitude / longitude 컬럼 또는 주소 컬럼을 이용해 일괄 분석을 수행합니다.")

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

with st.sidebar:
    st.header("설정")

    js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
    rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")

    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
    map_type = st.selectbox("지도 타입", ["SKYVIEW", "HYBRID"], index=0)

    wide_level = st.slider("wide level", 0, 6, 2)
    roof_level = st.slider("roof level", 0, 6, 1)

    max_rows = st.number_input(
        "최대 분석 행 수",
        min_value=1,
        max_value=1000,
        value=20,
        step=1
    )

    image_width = 1024
    image_height = 768


uploaded_file = st.file_uploader(
    "엑셀 업로드 (.xlsx, .csv)",
    type=["xlsx", "xls", "csv"]
)


if uploaded_file is not None:
    df = read_uploaded_file(uploaded_file)

    st.subheader("업로드된 데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    detected_lat_col, detected_lng_col = detect_lat_lng_columns(df)
    detected_address_col = detect_address_column(df)

    st.markdown("### 컬럼 자동 인식 결과")

    if detected_lat_col is not None and detected_lng_col is not None:
        st.success(f"GPS 컬럼 인식: 위도 `{detected_lat_col}`, 경도 `{detected_lng_col}`")
    else:
        st.info("latitude / longitude 컬럼은 자동 인식되지 않았습니다.")

    if detected_address_col is not None:
        st.success(f"주소 컬럼 인식: `{detected_address_col}`")
    else:
        st.warning("주소 컬럼을 자동으로 찾지 못했습니다.")

    columns = list(df.columns)
    options = [None] + columns

    st.markdown("### 사용할 컬럼 선택")

    lat_col = st.selectbox(
        "위도 컬럼",
        options=options,
        index=options.index(detected_lat_col) if detected_lat_col in columns else 0
    )

    lng_col = st.selectbox(
        "경도 컬럼",
        options=options,
        index=options.index(detected_lng_col) if detected_lng_col in columns else 0
    )

    address_col = st.selectbox(
        "주소 컬럼",
        options=options,
        index=options.index(detected_address_col) if detected_address_col in columns else 0
    )

    name_col = st.selectbox(
        "이름/기관명 컬럼",
        options=options,
        index=0
    )

    st.info(
        "처리 우선순위: latitude/longitude 컬럼이 있으면 GPS를 그대로 사용하고, "
        "GPS가 없으면 주소 컬럼을 이용해 좌표를 생성합니다."
    )

    if st.button("Batch 분석 실행", type="primary", use_container_width=True):

        if not js_key:
            st.error("JavaScript Key가 필요합니다.")
            st.stop()

        processed_df = df.copy()

        # -------------------------------------------------
        # A. GPS 컬럼이 있는 경우
        # -------------------------------------------------
        if lat_col is not None and lng_col is not None:
            processed_df["latitude"] = pd.to_numeric(processed_df[lat_col], errors="coerce")
            processed_df["longitude"] = pd.to_numeric(processed_df[lng_col], errors="coerce")

            if address_col is not None:
                processed_df["resolved_address"] = processed_df[address_col].astype(str)
            else:
                processed_df["resolved_address"] = None

            processed_df["geocode_status"] = "GPS_FROM_FILE"

        # -------------------------------------------------
        # B. GPS 컬럼은 없고 주소 컬럼만 있는 경우
        # -------------------------------------------------
        elif address_col is not None:
            if not rest_key:
                st.error("주소만 있는 파일을 분석하려면 REST API Key가 필요합니다.")
                st.stop()

            processed_df = add_coordinates_from_address(
                df=processed_df,
                address_col=address_col,
                rest_key=rest_key
            )

        # -------------------------------------------------
        # C. 둘 다 없는 경우
        # -------------------------------------------------
        else:
            st.error("latitude / longitude 컬럼 또는 주소 컬럼이 필요합니다.")
            st.stop()

        # 분석 가능한 행만 남김
        valid_df = processed_df.dropna(subset=["latitude", "longitude"]).copy()
        valid_df = valid_df.head(int(max_rows))

        if len(valid_df) == 0:
            st.error("분석 가능한 좌표가 없습니다. 주소 변환 결과를 확인하세요.")
            st.dataframe(processed_df, use_container_width=True)
            st.stop()

        st.success(f"분석 가능한 행: {len(valid_df)}개")

        st.subheader("좌표 변환 결과")
        st.dataframe(processed_df, use_container_width=True)

        # -------------------------------------------------
        # 모델 로딩
        # -------------------------------------------------
        with st.spinner("모델 로딩 중..."):
            model, preprocess, tokenizer, device = load_clip_model()

            if mode in ["linearprobe", "centroid"]:
                artifacts = load_artifacts()
            else:
                artifacts = {}

        # -------------------------------------------------
        # Batch 분석
        # -------------------------------------------------
        results = []

        progress = st.progress(0)
        status_text = st.empty()

        for i, row in valid_df.reset_index(drop=True).iterrows():
            progress.progress((i + 1) / len(valid_df))
            status_text.write(f"위성사진 분석 중: {i + 1} / {len(valid_df)}")

            lat = float(row["latitude"])
            lng = float(row["longitude"])

            if name_col is not None:
                site_name = row.get(name_col, f"row_{i}")
            else:
                site_name = f"row_{i}"

            resolved_address = row.get("resolved_address", None)

            try:
                images = capture_kakao_satellite_http(
                    js_key=js_key,
                    lat=lat,
                    lon=lng,
                    wide_level=wide_level,
                    roof_level=roof_level,
                    map_type=map_type,
                    width=image_width,
                    height=image_height,
                    capture_wide=True,
                )

                roof_img = images["roof"]
                wide_img = images.get("wide")

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
                if wide_img is not None:
                    wide_result = classify_pil_image(
                        pil_img=wide_img,
                        mode=mode,
                        model=model,
                        preprocess=preprocess,
                        tokenizer=tokenizer,
                        device=device,
                        artifacts=artifacts,
                    )

                results.append({
                    "name": site_name,
                    "latitude": lat,
                    "longitude": lng,
                    "resolved_address": resolved_address,
                    "geocode_status": row.get("geocode_status", None),

                    "roof_label": roof_result["label"],
                    "roof_probability": float(roof_result["probability"]),
                    "roof_score": float(roof_result["score"]),

                    "wide_label": wide_result["label"] if wide_result is not None else None,
                    "wide_probability": float(wide_result["probability"]) if wide_result is not None else None,
                    "wide_score": float(wide_result["score"]) if wide_result is not None else None,

                    "status": "OK",
                })

            except Exception as e:
                results.append({
                    "name": site_name,
                    "latitude": lat,
                    "longitude": lng,
                    "resolved_address": resolved_address,
                    "geocode_status": row.get("geocode_status", None),
                    "roof_label": None,
                    "roof_probability": None,
                    "roof_score": None,
                    "wide_label": None,
                    "wide_probability": None,
                    "wide_score": None,
                    "status": f"ERROR: {e}",
                })

        progress.empty()
        status_text.empty()

        result_df = pd.DataFrame(results)

        if "roof_probability" in result_df.columns:
            result_df = result_df.sort_values(
                by="roof_probability",
                ascending=False,
                na_position="last"
            )

        st.subheader("Batch 분석 결과")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="결과 CSV 다운로드",
            data=csv,
            file_name="batch_analysis_result.csv",
            mime="text/csv",
            use_container_width=True
        )
