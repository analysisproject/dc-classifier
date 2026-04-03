import pandas as pd
import streamlit as st

from utils.core import (
    render_shared_sidebar,
    read_batch_file,
    detect_lat_lng_name_columns,
    load_clip_model,
    load_artifacts,
    analyze_one_coordinate,
    dataframe_to_excel_bytes,
)

st.set_page_config(
    page_title="Batch Excel Analysis",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Batch Excel Analysis")
st.caption("엑셀 또는 CSV의 latitude / longitude 컬럼을 이용해 일괄 분석을 수행합니다.")

js_key, rest_key, mode, map_type, wide_level, roof_level = render_shared_sidebar("Batch Excel Analysis")

with st.sidebar:
    st.markdown("---")
    st.page_link("app.py", label="Single Analysis", icon="🛰️")
    st.page_link("pages/2_Batch_Excel_Analysis.py", label="Batch Excel Analysis", icon="📄")

    st.markdown("---")
    compute_wide = st.checkbox("wide view도 함께 계산", value=True)

uploaded_file = st.file_uploader(
    "엑셀 업로드 (.xlsx, .csv)",
    type=["xlsx", "csv"]
)

st.write("파일에는 `latitude`, `longitude` 컬럼이 있어야 하며, `name` 컬럼이 있으면 함께 결과에 포함됩니다.")

if uploaded_file is not None:
    try:
        batch_df = read_batch_file(uploaded_file)

        st.subheader("업로드된 데이터 미리보기")
        st.dataframe(batch_df.head(), use_container_width=True)

        lat_col, lng_col, name_col = detect_lat_lng_name_columns(batch_df)

        if lat_col is None or lng_col is None:
            st.error("파일에 `latitude`, `longitude` 또는 동등한 좌표 컬럼이 반드시 있어야 합니다.")
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
                                    compute_wide=compute_wide,
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
