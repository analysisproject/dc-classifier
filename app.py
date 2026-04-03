import pandas as pd
import streamlit as st

from utils.core import (
    init_single_session_state,
    reverse_geocode,
    format_reverse_address,
    geocode_address,
    capture_kakao_satellite_http,
    load_clip_model,
    load_artifacts,
    classify_pil_image,
    dataframe_to_excel_bytes,
    read_batch_file,
    detect_lat_lng_name_columns,
    LINEARPROBE_PATH,
    CENTROIDS_PATH,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Satellite Data Center Classifier",
    page_icon="🛰️",
    layout="wide",
)

# ============================================================
# Session init
# ============================================================
init_single_session_state()

# ============================================================
# Header
# ============================================================
st.title("🛰️ Satellite Data Center Classifier")
st.caption("GPS 또는 주소를 입력하면 위성사진을 불러오고 roof view 기준으로 데이터센터 여부를 판정합니다.")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.page_link("app.py", label="Single Analysis", icon="🛰️")
    st.page_link("pages/2_Batch_Excel_Analysis.py", label="File Upload(Excel, csv)", icon="📄")

    st.markdown("---")
    st.subheader("입력 / 분석 설정")

    input_mode = st.radio("입력 방식", ["GPS 입력", "주소 입력"], index=0)

    st.markdown("---")

    js_key = st.text_input(
        "JavaScript Key",
        value=st.secrets.get("KAKAO_JS_KEY", ""),
        type="password",
    )
    rest_key = st.text_input(
        "REST API Key",
        value=st.secrets.get("KAKAO_REST_KEY", ""),
        type="password",
    )

    st.markdown("---")

    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
    map_type = st.selectbox("지도 타입", ["SKYVIEW", "HYBRID"], index=0)

    roof_level = st.slider("roof level", 0, 6, 1)
    wide_level = st.slider("wide level", 0, 6, 2)

    show_wide = st.checkbox("wide view도 함께 분석", value=False)

    st.markdown("---")
    st.subheader("렌더링 속도 옵션")
    render_width = st.select_slider("가로 해상도", options=[768, 960, 1024, 1280], value=1024)
    render_height = st.select_slider("세로 해상도", options=[432, 540, 640, 720], value=640)

    st.markdown("---")
    st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
    st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

# ============================================================
# Layout
# ============================================================
col1, col2, col3 = st.columns([0.95, 1.05, 1.05], gap="large")

roof_result = None
wide_result = None
roof_img = None
wide_img = None

# ============================================================
# Column 1 - input
# ============================================================
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

                st.session_state["run_analysis"] = True
                st.session_state["single_result_ready"] = True

            except Exception as e:
                st.session_state["run_analysis"] = False
                st.session_state["single_result_ready"] = False
                st.error(f"GPS 좌표 입력 오류: {e}")

    else:
        address = st.text_input("주소", placeholder="예: 세종특별자치시 도움6로 11")

        if st.button("위치 확인 및 분석", type="primary", use_container_width=True):
            if not rest_key:
                st.session_state["run_analysis"] = False
                st.session_state["single_result_ready"] = False
                st.error("주소 입력 모드에서는 REST API Key가 필요합니다.")
            elif not address.strip():
                st.session_state["run_analysis"] = False
                st.session_state["single_result_ready"] = False
                st.warning("주소를 입력하세요.")
            else:
                try:
                    geo = geocode_address(rest_key, address.strip())
                    if geo is None:
                        st.session_state["run_analysis"] = False
                        st.session_state["single_result_ready"] = False
                        st.warning("주소 검색 결과가 없습니다.")
                    else:
                        lat, lng, meta = geo
                        st.session_state["lat"] = lat
                        st.session_state["lng"] = lng
                        st.session_state["resolved_text"] = address.strip()
                        st.session_state["resolved_meta"] = meta
                        st.session_state["resolved_address_str"] = meta.get("address_name", address.strip())
                        st.session_state["run_analysis"] = True
                        st.session_state["single_result_ready"] = True
                except Exception as e:
                    st.session_state["run_analysis"] = False
                    st.session_state["single_result_ready"] = False
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

# ============================================================
# Column 2 - image rendering + model analysis
# ============================================================
with col2:
    st.subheader("3) 위성 사진")

    if not st.session_state.get("run_analysis", False):
        st.info("좌측에서 위치를 입력하고 '위치 확인 및 분석' 버튼을 누르면 위성사진과 분석 결과가 표시됩니다.")
    elif not js_key:
        st.warning("JavaScript Key가 비어 있습니다.")
    else:
        try:
            with st.spinner("위성사진 렌더링 중..."):
                images = capture_kakao_satellite_http(
                    js_key=js_key,
                    lat=st.session_state["lat"],
                    lon=st.session_state["lng"],
                    wide_level=wide_level,
                    roof_level=roof_level,
                    map_type=map_type,
                    width=render_width,
                    height=render_height,
                    capture_wide=show_wide,
                )

            roof_img = images["roof"]
            wide_img = images.get("wide")

            st.markdown("**roof view**")
            st.image(roof_img, use_container_width=True)

            if wide_img is not None:
                st.markdown("**wide view**")
                st.image(wide_img, use_container_width=True)

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

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            roof_result = None
            wide_result = None
            roof_img = None
            wide_img = None

# ============================================================
# Column 3 - final interpretation
# ============================================================
with col3:
    st.subheader("4) 최종 판정")

    try:
        if not st.session_state.get("run_analysis", False):
            st.info("분석 실행 전입니다.")
        elif roof_result is not None:
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
                    f"roof view는 지붕 형상과 설비 구조를 더 직접적으로 보여주므로 최종 판정 기준으로 사용했습니다. "
                    f"wide view는 주변 입지 맥락을 해석하는 보조 정보입니다."
                )
            else:
                st.write(
                    f"최종 판정은 **roof view 단일 결과**를 기준으로 했습니다. "
                    f"roof 결과는 **{final_label}** 이고, "
                    f"roof 기준 데이터센터 확률은 **{final_prob:.4f}**, "
                    f"내부 판정값은 **{roof_score_text}** 입니다."
                )

            with st.expander("score 계산 방식 설명", expanded=False):
                if roof_result["mode"] == "zeroshot":
                    st.write("zeroshot에서는 score = 가장 높은 positive prompt 유사도 - 가장 높은 negative prompt 유사도 입니다.")
                elif roof_result["mode"] == "centroid":
                    st.write("centroid에서는 score = positive centroid 유사도 - negative centroid 유사도 입니다.")
                else:
                    st.write("linearprobe에서는 score를 predict_proba(데이터센터 확률)와 동일하게 사용했습니다.")
        else:
            st.info("위성 사진이 생성되면 최종 판정이 여기에 표시됩니다.")

    except Exception as e:
        st.error(f"최종 판정 표시 오류: {e}")

# ============================================================
# Batch analysis helper
# ============================================================
def analyze_one_coordinate_fast(
    js_key: str,
    rest_key: str,
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
    artifacts,
    render_width: int,
    render_height: int,
    compute_wide: bool = False,
):
    images = capture_kakao_satellite_http(
        js_key=js_key,
        lat=lat,
        lon=lng,
        wide_level=wide_level,
        roof_level=roof_level,
        map_type=map_type,
        width=render_width,
        height=render_height,
        capture_wide=compute_wide,
    )

    roof_img_local = images["roof"]
    roof_result_local = classify_pil_image(
        pil_img=roof_img_local,
        mode=mode,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        artifacts=artifacts,
    )

    wide_result_local = None
    if compute_wide and "wide" in images:
        wide_result_local = classify_pil_image(
            pil_img=images["wide"],
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
        "roof_label": roof_result_local["label"],
        "roof_probability": float(roof_result_local["probability"]),
        "roof_score": float(roof_result_local["score"]),
        "wide_label": None if wide_result_local is None else wide_result_local["label"],
        "wide_probability": None if wide_result_local is None else float(wide_result_local["probability"]),
        "wide_score": None if wide_result_local is None else float(wide_result_local["score"]),
        "final_label": roof_result_local["label"],
        "final_probability": float(roof_result_local["probability"]),
        "mode": mode,
        "error": None,
    }

# ============================================================
# Batch analysis UI
# ============================================================
st.markdown("---")
st.header("5) 엑셀 일괄 분석")

st.write(
    "엑셀 또는 CSV에 `latitude`, `longitude` 컬럼이 있으면 각 행별로 확률을 계산합니다. "
    "`name` 컬럼이 있으면 결과에 함께 포함됩니다."
)

uploaded_file = st.file_uploader(
    "엑셀 업로드 (.xlsx, .csv)",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:
    try:
        batch_df = read_batch_file(uploaded_file)

        st.write("업로드된 데이터 미리보기")
        st.dataframe(batch_df.head(), use_container_width=True)

        lat_col, lng_col, name_col = detect_lat_lng_name_columns(batch_df)

        if lat_col is None or lng_col is None:
            st.error("파일에 `latitude/longitude` 또는 `lat/lng` 컬럼이 있어야 합니다.")
        else:
            batch_compute_wide = st.checkbox("일괄 분석에서도 wide view 분석", value=False)
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
                        st.warning("유효한 좌표 값이 없습니다.")
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
                                one = analyze_one_coordinate_fast(
                                    js_key=js_key,
                                    rest_key=rest_key,
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
                                    render_width=render_width,
                                    render_height=render_height,
                                    compute_wide=batch_compute_wide,
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
                        xlsx_bytes = dataframe_to_excel_bytes(result_df)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                "결과 CSV 다운로드",
                                data=csv_bytes,
                                file_name="batch_analysis_results.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        with c2:
                            st.download_button(
                                "결과 Excel 다운로드",
                                data=xlsx_bytes,
                                file_name="batch_analysis_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )

    except Exception as e:
        st.error(f"파일 처리 오류: {e}")
