import streamlit as st
import pandas as pd
import re

from utils.core import (
    init_single_session_state,
    get_secret_or_env,
    geocode_address,
    reverse_geocode,
    format_reverse_address,
    capture_kakao_satellite_http,
    load_clip_model,
    load_artifacts,
    classify_pil_image,
    LINEARPROBE_PATH,
    CENTROIDS_PATH,
)


# =========================================================
# 추가 함수 1: 엑셀 파일에서 주소 컬럼 자동 인식
# =========================================================
def detect_address_column(df):
    """
    엑셀/CSV 파일에서 주소 컬럼을 자동으로 찾는 함수.
    1차: 컬럼명 기준
    2차: 실제 값의 한국 주소 패턴 기준
    """

    address_keywords = [
        "주소", "도로명", "지번", "소재지", "위치",
        "address", "addr", "location", "site", "place"
    ]

    # 1) 컬럼명 기준 탐색
    for col in df.columns:
        col_str = str(col).lower()
        if any(keyword.lower() in col_str for keyword in address_keywords):
            return col

    # 2) 실제 값 기준 탐색
    korean_address_patterns = [
        r"서울|부산|대구|인천|광주|대전|울산|세종",
        r"경기도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도|제주",
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
# 추가 함수 2: 엑셀 파일에서 위도/경도 컬럼 자동 인식
# =========================================================
def detect_lat_lng_columns(df):
    """
    엑셀/CSV 파일에서 위도/경도 컬럼을 자동으로 찾는 함수.
    """

    lat_candidates = ["lat", "latitude", "위도", "y", "y좌표"]
    lng_candidates = ["lng", "lon", "long", "longitude", "경도", "x", "x좌표"]

    lat_col = None
    lng_col = None

    for col in df.columns:
        col_lower = str(col).lower()

        if lat_col is None and any(keyword in col_lower for keyword in lat_candidates):
            lat_col = col

        if lng_col is None and any(keyword in col_lower for keyword in lng_candidates):
            lng_col = col

    return lat_col, lng_col


# =========================================================
# 추가 함수 3: 업로드된 파일 읽기
# =========================================================
def read_uploaded_table(uploaded_file):
    """
    업로드된 엑셀 또는 CSV 파일을 pandas DataFrame으로 읽는다.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")

    return pd.read_excel(uploaded_file)


# =========================================================
# 기존 Streamlit 설정
# =========================================================
st.set_page_config(
    page_title="Satellite Data Center Classifier",
    page_icon="🛰️",
    layout="wide",
)

init_single_session_state()

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# 추가: 업로드된 엑셀 처리 결과 저장용
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None

st.title("🛰️ Satellite Data Center Classifier")
st.caption("GPS 또는 주소를 입력하면 위성사진을 불러오고 roof view 기준으로 데이터센터 여부를 판정합니다.")

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("설정")
    js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
    rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")

    # 수정: 엑셀 업로드 입력 방식 추가
    input_mode = st.radio("입력 방식", ["GPS 입력", "주소 입력", "엑셀 업로드"], index=0)

    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"], index=0)
    map_type = st.selectbox("지도 타입", ["SKYVIEW", "HYBRID"], index=0)

    show_wide = st.checkbox("wide view도 함께 렌더링", value=False)

    # wide view는 항상 함께 렌더링
    show_wide = True

    # 이미지 크기는 고정
    image_width = 1024
    image_height = 768

    st.caption(f"이미지 크기: {image_width} x {image_height} (고정)")
    st.caption("wide view는 항상 함께 렌더링됩니다.")

    wide_level = st.slider("wide level", 0, 6, 2)
    roof_level = st.slider("roof level", 0, 6, 1)

    st.markdown("---")
    st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
    st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")


col1, col2, col3 = st.columns([0.9, 1.05, 1.05], gap="large")


# =========================================================
# Column 1: 위치 입력
# =========================================================
with col1:
    st.subheader("1) 위치 입력")

    # -----------------------------------------------------
    # 기존 기능 1: GPS 직접 입력
    # -----------------------------------------------------
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
                st.session_state["run_analysis"] = True

                if rest_key:
                    rev = reverse_geocode(rest_key, lat, lng)
                    st.session_state["resolved_meta"] = rev
                    st.session_state["resolved_address_str"] = format_reverse_address(rev)
                else:
                    st.session_state["resolved_meta"] = None
                    st.session_state["resolved_address_str"] = None

            except Exception as e:
                st.session_state["run_analysis"] = False
                st.error(f"GPS 좌표 입력 오류: {e}")

    # -----------------------------------------------------
    # 기존 기능 2: 주소 직접 입력
    # -----------------------------------------------------
    elif input_mode == "주소 입력":
        address = st.text_input("주소", placeholder="예: 세종특별자치시 도움6로 11")

        if st.button("위치 확인 및 분석", type="primary", use_container_width=True):
            if not rest_key:
                st.session_state["run_analysis"] = False
                st.error("주소 입력 모드에서는 REST API Key가 필요합니다.")
            elif not address.strip():
                st.session_state["run_analysis"] = False
                st.warning("주소를 입력하세요.")
            else:
                try:
                    geo = geocode_address(rest_key, address.strip())
                    if geo is None:
                        st.session_state["run_analysis"] = False
                        st.warning("주소 검색 결과가 없습니다.")
                    else:
                        lat, lng, meta = geo
                        st.session_state["lat"] = lat
                        st.session_state["lng"] = lng
                        st.session_state["resolved_text"] = address.strip()
                        st.session_state["resolved_meta"] = meta
                        st.session_state["resolved_address_str"] = meta.get("address_name", address.strip())
                        st.session_state["run_analysis"] = True
                except Exception as e:
                    st.session_state["run_analysis"] = False
                    st.error(f"주소 검색 오류: {e}")

    # -----------------------------------------------------
    # 추가 기능 3: 엑셀 업로드
    # -----------------------------------------------------
    else:
        st.markdown("엑셀 또는 CSV 파일을 업로드하면 주소/GPS 컬럼을 자동으로 인식합니다.")

        uploaded_file = st.file_uploader(
            "엑셀 또는 CSV 파일 업로드",
            type=["xlsx", "xls", "csv"]
        )

        if uploaded_file is not None:
            try:
                df = read_uploaded_table(uploaded_file)

                if df.empty:
                    st.warning("업로드된 파일에 데이터가 없습니다.")
                else:
                    st.write("업로드된 데이터 미리보기")
                    st.dataframe(df.head(20), use_container_width=True)

                    # 자동 인식
                    detected_address_col = detect_address_column(df)
                    detected_lat_col, detected_lng_col = detect_lat_lng_columns(df)

                    st.markdown("### 자동 인식 결과")

                    if detected_address_col is not None:
                        st.success(f"주소 컬럼 자동 인식: `{detected_address_col}`")
                    else:
                        st.warning("주소 컬럼을 자동으로 찾지 못했습니다.")

                    if detected_lat_col is not None and detected_lng_col is not None:
                        st.success(
                            f"GPS 컬럼 자동 인식: 위도 `{detected_lat_col}`, 경도 `{detected_lng_col}`"
                        )
                    else:
                        st.info("GPS 컬럼은 자동 인식되지 않았습니다.")

                    columns = list(df.columns)
                    select_options = [None] + columns

                    # 사용자가 직접 수정 가능
                    address_col = st.selectbox(
                        "주소 컬럼 선택",
                        options=select_options,
                        index=select_options.index(detected_address_col)
                        if detected_address_col in columns else 0
                    )

                    lat_col = st.selectbox(
                        "위도 컬럼 선택",
                        options=select_options,
                        index=select_options.index(detected_lat_col)
                        if detected_lat_col in columns else 0
                    )

                    lng_col = st.selectbox(
                        "경도 컬럼 선택",
                        options=select_options,
                        index=select_options.index(detected_lng_col)
                        if detected_lng_col in columns else 0
                    )

                    st.caption(
                        "위도/경도 컬럼이 있으면 GPS를 우선 사용합니다. "
                        "위도/경도가 없고 주소 컬럼이 있으면 주소를 GPS로 변환합니다."
                    )

                    if st.button("엑셀 주소/GPS 처리", type="primary", use_container_width=True):

                        processed_df = df.copy()

                        # 1) 위도/경도 컬럼이 있는 경우: 그대로 사용
                        if lat_col is not None and lng_col is not None:
                            processed_df["latitude"] = pd.to_numeric(
                                processed_df[lat_col], errors="coerce"
                            )
                            processed_df["longitude"] = pd.to_numeric(
                                processed_df[lng_col], errors="coerce"
                            )

                            if address_col is not None:
                                processed_df["resolved_address"] = processed_df[address_col].astype(str)
                            else:
                                processed_df["resolved_address"] = None

                            processed_df["geocode_status"] = "GPS_FROM_FILE"

                            st.session_state["uploaded_df"] = processed_df
                            st.success("엑셀의 위도/경도 컬럼을 사용해 위치 정보를 처리했습니다.")

                        # 2) 위도/경도는 없지만 주소 컬럼이 있는 경우: 주소 → GPS 변환
                        elif address_col is not None:
                            if not rest_key:
                                st.error("주소를 GPS로 변환하려면 REST API Key가 필요합니다.")
                                st.stop()

                            lat_list = []
                            lng_list = []
                            resolved_address_list = []
                            status_list = []

                            progress_bar = st.progress(0)
                            total_rows = len(processed_df)

                            for idx, addr in enumerate(processed_df[address_col].astype(str)):
                                progress_bar.progress((idx + 1) / total_rows)

                                if not addr or addr.strip() == "" or addr.lower() == "nan":
                                    lat_list.append(None)
                                    lng_list.append(None)
                                    resolved_address_list.append(None)
                                    status_list.append("EMPTY_ADDRESS")
                                    continue

                                try:
                                    geo = geocode_address(rest_key, addr.strip())

                                    if geo is None:
                                        lat_list.append(None)
                                        lng_list.append(None)
                                        resolved_address_list.append(None)
                                        status_list.append("NOT_FOUND")
                                    else:
                                        lat, lng, meta = geo
                                        lat_list.append(lat)
                                        lng_list.append(lng)
                                        resolved_address_list.append(
                                            meta.get("address_name", addr.strip())
                                        )
                                        status_list.append("OK")

                                except Exception as e:
                                    lat_list.append(None)
                                    lng_list.append(None)
                                    resolved_address_list.append(None)
                                    status_list.append(f"ERROR: {e}")

                            processed_df["latitude"] = lat_list
                            processed_df["longitude"] = lng_list
                            processed_df["resolved_address"] = resolved_address_list
                            processed_df["geocode_status"] = status_list

                            st.session_state["uploaded_df"] = processed_df

                            ok_count = sum(processed_df["geocode_status"] == "OK")
                            st.success(f"주소 변환 완료: {ok_count}개 성공 / 전체 {total_rows}개")

                        # 3) 둘 다 없는 경우
                        else:
                            st.error("주소 컬럼 또는 위도/경도 컬럼을 선택해야 합니다.")
                            st.stop()

            except Exception as e:
                st.error(f"엑셀 파일 처리 오류: {e}")

        # -------------------------------------------------
        # 처리된 엑셀 결과 표시 및 분석 대상 선택
        # -------------------------------------------------
        if st.session_state.get("uploaded_df") is not None:
            processed_df = st.session_state["uploaded_df"]

            st.markdown("### 처리 결과")
            st.dataframe(processed_df, use_container_width=True)

            valid_df = processed_df.dropna(subset=["latitude", "longitude"]).copy()

            if len(valid_df) == 0:
                st.warning("분석 가능한 좌표가 없습니다.")
            else:
                st.markdown("### 분석할 위치 선택")

                valid_df = valid_df.reset_index(drop=True)

                display_options = []

                for i, row in valid_df.iterrows():
                    address_text = row.get("resolved_address")

                    if address_text is None or str(address_text) == "nan":
                        address_text = "주소 없음"

                    display_options.append(
                        f"{i}: {address_text} / {row['latitude']:.6f}, {row['longitude']:.6f}"
                    )

                selected_option = st.selectbox(
                    "분석할 행 선택",
                    options=display_options
                )

                selected_idx = int(selected_option.split(":")[0])
                selected_row = valid_df.iloc[selected_idx]

                if st.button("선택한 위치 분석", type="primary", use_container_width=True):
                    lat = float(selected_row["latitude"])
                    lng = float(selected_row["longitude"])

                    st.session_state["lat"] = lat
                    st.session_state["lng"] = lng
                    st.session_state["resolved_text"] = "엑셀 업로드 위치"
                    st.session_state["resolved_address_str"] = selected_row.get("resolved_address")
                    st.session_state["resolved_meta"] = selected_row.to_dict()
                    st.session_state["run_analysis"] = True

                    st.success("선택한 위치를 분석 대상으로 설정했습니다.")

    # -----------------------------------------------------
    # 현재 선택 위치 표시: 기존 코드 유지
    # -----------------------------------------------------
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


# =========================================================
# 실제 분석: 기존 코드 유지
# =========================================================
if st.session_state.get("run_analysis", False):
    try:
        if not js_key:
            raise RuntimeError("JavaScript Key가 필요합니다.")

        with st.spinner("위성사진 렌더링 중..."):
            images = capture_kakao_satellite_http(
                js_key=js_key,
                lat=st.session_state["lat"],
                lon=st.session_state["lng"],
                wide_level=wide_level,
                roof_level=roof_level,
                map_type=map_type,
                width=image_width,
                height=image_height,
                capture_wide=show_wide,
            )

        roof_img = images["roof"]
        wide_img = images.get("wide")

        with st.spinner("모델 분석 중..."):
            model, preprocess, tokenizer, device = load_clip_model()

            if mode in ["linearprobe", "centroid"]:
                artifacts = load_artifacts()
            else:
                artifacts = {}

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

        st.session_state["last_result"] = {
            "roof_img": roof_img,
            "wide_img": wide_img,
            "roof_result": roof_result,
            "wide_result": wide_result,
            "mode": mode,
        }

    except Exception as e:
        st.exception(e)

    finally:
        # rerun 때 자동 재실행 방지
        st.session_state["run_analysis"] = False


result = st.session_state.get("last_result")


# =========================================================
# Column 2: 위성 사진
# =========================================================
with col2:
    st.subheader("3) 위성 사진")

    if result is None:
        st.info("위치 확인 및 분석 버튼을 누르면 여기에서 위성 사진이 생성됩니다.")
    else:
        st.markdown("**roof view (대표 판정 기준)**")
        st.image(result["roof_img"], use_container_width=True)

        st.markdown("**wide view (보조 확인용)**")
        if result["wide_img"] is not None:
            st.image(result["wide_img"], use_container_width=True)
        else:
            st.warning("wide view 이미지가 생성되지 않았습니다.")


# =========================================================
# Column 3: 최종 판정
# =========================================================
with col3:
    st.subheader("4) 최종 판정")

    if result is None:
        st.info("위성 사진이 생성되면 최종 판정이 여기에 표시됩니다.")
    else:
        roof_result = result["roof_result"]
        wide_result = result["wide_result"]

        # 대표 판정은 항상 roof 기준
        final_prob = float(roof_result["probability"])
        final_label = roof_result["label"]
        roof_score = float(roof_result["score"])

        if roof_result["mode"] in ["zeroshot", "centroid"]:
            roof_score_label = "roof 판정 margin (유사도 차이)"
            roof_score_display = f"{roof_score:.6f}"
            roof_score_text = f"margin {roof_score:.4f}"
        else:
            roof_score_label = "roof 데이터센터 예측 확률(score)"
            roof_score_display = f"{roof_score * 100:.2f}%"
            roof_score_text = f"예측확률(score) {roof_score:.4f}"

        if final_label == "데이터센터":
            st.success(f"대표 판정(roof 기준): **{final_label}**")
        else:
            st.info(f"대표 판정(roof 기준): **{final_label}**")

        st.metric("roof 기준 데이터센터 확률", f"{final_prob * 100:.2f}%")
        st.write(f"**{roof_score_label}**: `{roof_score_display}`")
        st.write(f"**분류 모드**: `{roof_result['mode']}`")

        st.markdown("---")
        st.markdown("**세부 결과 비교**")

        st.write(f"**roof 결과 라벨**: `{roof_result['label']}`")
        st.write(f"**roof 기준 데이터센터 확률**: `{float(roof_result['probability']) * 100:.2f}%`")

        if wide_result is not None:
            wide_prob = float(wide_result["probability"])
            wide_score = float(wide_result["score"])
            wide_label = wide_result["label"]

            if wide_result["mode"] in ["zeroshot", "centroid"]:
                wide_score_label = "wide 판정 margin (유사도 차이)"
                wide_score_display = f"{wide_score:.6f}"
                wide_score_text = f"margin {wide_score:.4f}"
            else:
                wide_score_label = "wide 데이터센터 예측 확률(score)"
                wide_score_display = f"{wide_score * 100:.2f}%"
                wide_score_text = f"예측확률(score) {wide_score:.4f}"

            st.write(f"**wide 결과 라벨**: `{wide_label}`")
            st.write(f"**wide 기준 데이터센터 확률**: `{wide_prob * 100:.2f}%`")
            st.write(f"**{wide_score_label}**: `{wide_score_display}`")

            st.markdown("**해석**")
            st.write(
                f"대표 판정은 **roof view 결과**를 기준으로 했습니다. "
                f"roof 결과는 **{final_label}** 이고, "
                f"roof 기준 데이터센터 확률은 **{final_prob:.4f}**, "
                f"내부 판정값은 **{roof_score_text}** 입니다. "
                f"wide 결과는 **{wide_label}**, "
                f"wide 기준 데이터센터 확률은 **{wide_prob:.4f}**, "
                f"내부 판정값은 **{wide_score_text}** 입니다."
            )
        else:
            st.warning("wide 결과가 없어 roof 기준 결과만 표시합니다.")
            st.markdown("**해석**")
            st.write(
                f"대표 판정은 **roof view 결과**를 기준으로 했습니다. "
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
