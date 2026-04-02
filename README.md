# KakaoMap Data Center Classifier

주소 또는 GPS 좌표를 입력하면 카카오맵의 상세 위성지도를 보여주고, 같은 위치를 고해상도로 캡처해서 CLIP 기반 모델로 데이터센터 여부를 분류하는 Streamlit 대시보드입니다.

## 포함 파일

- `app.py` : Streamlit 대시보드 본체
- `prepare_assets.py` : pilot 이미지셋으로 `centroids.npz`, `linearprobe.joblib` 생성
- `requirements.txt` : Python 패키지 목록
- `packages.txt` : Streamlit Cloud / Debian 계열 런타임에서 필요한 시스템 패키지
- `.streamlit/config.toml` : Streamlit 설정
- `.streamlit/secrets.toml.example` : 로컬/배포용 시크릿 예시
- `artifacts/` : 학습 결과 파일을 넣는 폴더

## 1. 로컬 실행

```bash
pip install -r requirements.txt
python -m playwright install chromium
streamlit run app.py
```

## 2. 로컬 시크릿 설정

`.streamlit/secrets.toml.example`를 복사해 `.streamlit/secrets.toml`로 저장합니다.

```toml
KAKAO_JS_KEY = "..."
KAKAO_REST_KEY = "..."
```

## 3. pilot 데이터셋으로 배포용 분류기 만들기

데이터셋 구조 예시:

```text
my_dataset/
  pos/
    pos_001.png
  neg/
    neg_001.png
```

학습 실행:

```bash
python prepare_assets.py --data_root ./my_dataset --out_dir ./artifacts
```

생성 결과:

- `artifacts/centroids.npz`
- `artifacts/linearprobe.joblib`

## 4. GitHub에 올릴 파일

최소한 아래는 저장소에 있어야 합니다.

```text
app.py
prepare_assets.py
requirements.txt
packages.txt
README.md
.streamlit/config.toml
.streamlit/secrets.toml.example
```

`centroid`, `linearprobe` 모드를 실제로 쓰려면 아래도 추가해야 합니다.

```text
artifacts/centroids.npz
artifacts/linearprobe.joblib
```

## 5. Streamlit Community Cloud 배포

1. GitHub 저장소 생성
2. 위 파일 업로드
3. Streamlit Community Cloud에서 `app.py`를 메인 파일로 지정
4. App settings → Secrets에 아래 추가

```toml
KAKAO_JS_KEY = "..."
KAKAO_REST_KEY = "..."
```

## 6. 동작 방식

1. 주소 또는 GPS 입력
2. Kakao Local REST API로 주소를 좌표로 변환
3. Kakao Maps JavaScript API로 대시보드에 지도 표시
4. Playwright로 위성지도 영역을 캡처
5. CLIP 임베딩 추출
6. zero-shot / centroid / linearprobe 방식으로 데이터센터 여부 예측
7. 확률과 세부 점수 표시

## 7. 주의

- 공개 저장소에는 실제 시크릿 파일을 커밋하지 않는 편이 안전합니다.
- `zeroshot`은 바로 실행 가능하지만, `centroid`와 `linearprobe`는 `artifacts/` 파일이 있어야 합니다.
- Playwright가 처음 실행될 때 Chromium 설치 때문에 시간이 걸릴 수 있습니다.
