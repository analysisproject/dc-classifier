import streamlit as st

from utils.core import (
    init_single_session_state,
    get_secret_or_env,
    reverse_geocode,
    format_reverse_address,
    geocode_address,
    capture_kakao_satellite_http,
    load_clip_model,
    load_artifacts,
    classify_pil_image,
    LINEARPROBE_PATH,
    CENTROIDS_PATH,
)

st.set_page_config(
    page_title="Satellite Data Center Classifier",
    page_icon="🛰️",
    layout="wide",
)

init_single_session_state()

default_js_key = get_secret_or_env("KAKAO_JS_KEY", "")
default_rest_key = get_secret_or_env("KAKAO_REST_KEY", "")

st.title("🛰️ Satellite Data Center Classifier")
st.caption("GPS 또는 주소를 입력하면 위성사진을 불러오고 roof view 기준으로 데이터센터 여부를 판정합니다.")

js_key, rest_key, mode, map_type, wide_level, roof_level = render_shared_sidebar("Single Analysis")


with st.sidebar:

    # 1️⃣ 페이지 선택 (맨 위)
    st.page_link("app.py", label="Single Analysis", icon="🛰️")
    st.page_link("pages/2_Batch_Excel_Analysis.py", label="Batch Excel Analysis", icon="📄")

    st.markdown("---")

    # 2️⃣ 입력 방식
    input_mode = st.radio(
        "입력 방식",
        ["GPS 입력", "주소 입력"],
        index=0
    )

    st.markdown("---")

    # 3️⃣ API 키
    js_key = st.text_input("JavaScript Key", value=default_js_key, type="password")
    rest_key = st.text_input("REST API Key", value=default_rest_key, type="password")

    st.markdown("---")

    # 4️⃣ 모델 옵션
    mode = st.selectbox("분류 모드", ["zeroshot", "centroid", "linearprobe"])
    map_type = st.selectbox("지도 타입", ["SKYVIEW", "HYBRID"])

    wide_level = st.slider("wide level", 0, 6, 2)
    roof_level = st.slider("roof level", 0, 6, 1)

    st.markdown("---")

    st.write(f"linearprobe.joblib: {'있음' if LINEARPROBE_PATH.exists() else '없음'}")
    st.write(f"centroids.npz: {'있음' if CENTROIDS_PATH.exists() else '없음'}")

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
