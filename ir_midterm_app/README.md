# IR 중간고사 검색 시스템

NLTK `Inaugural Address` 말뭉치를 대상으로 만든 정보검색 실기 과제용 Streamlit 앱입니다. 검색어를 입력하면 미국 대통령 취임사를 `cosine` TF-IDF 또는 `BM25(k1=1.2, b=0.9)`로 랭킹하고, 대통령 이름, 관련 문장, 원문 보기 링크, 질의어별 IDF, 결과 문서별 TF를 표시합니다.

## 실행 방법

```bash
cd /Users/benpark/Desktop/temp
source .venv/bin/activate
cd ir_midterm_app
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501`을 열면 됩니다.

## Render 배포

데모 사이트: `https://ir-midterm-search.onrender.com`

이 저장소 루트의 `render.yaml`은 Render Blueprint 배포용입니다. Render Starter 기준으로 설정되어 있고, 앱 루트는 `ir_midterm_app`입니다.

Render Dashboard에서 Blueprint를 쓰지 않고 Web Service로 직접 만들 경우:

- Root Directory: `ir_midterm_app`
- Build Command: `pip install -r requirements.txt && python -m nltk.downloader -d ./nltk_data inaugural`
- Start Command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- Environment Variable: `NLTK_DATA=./nltk_data`

## 테스트

```bash
cd /Users/benpark/Desktop/temp
source .venv/bin/activate
cd ir_midterm_app
pytest -q
```

## 파일 구성

- `ir_engine.py`: 토큰화, stemming, 사전, 역파일, TF, IDF, cosine, BM25, 중요 문장 추출을 담당합니다.
- `app.py`: Streamlit 검색 UI와 원문 보기 화면을 제공합니다.
- `test_ir_engine.py`: 엔진의 핵심 정보검색 동작을 검증하는 pytest 테스트입니다.

## 수동 검증 질의

- `freedom union`
- `war peace`
- `constitution people`

각 질의에서 `cosine`과 `BM25`를 번갈아 선택해 순위 차이와 TF/IDF 표시를 확인하면 됩니다.
