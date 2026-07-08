# STT 백엔드 실측 테스트 가이드 (TESTING)

문서 버전: 1.0 (2026-07-08)
성격: [SETUP.MD](SETUP.MD) 준비가 끝난 이후, 실제 모델 실측(SETUP.MD §4 "다음 단계") 진행 상황과 테스트 방법을 기록한다. 실측 결과가 쌓이면 이 문서의 §4에 계속 추가한다.

---

## 1. 현재까지 진행 상황 (2026-07-08 기준)

- llama.cpp 설치 완료 (사전빌드 바이너리, [SETUP.MD](SETUP.MD) §2.1)
- Qwen3-ASR-1.7B 모델 로딩 확인:
  ```powershell
  .\llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF
  ```
  `-hf` 플래그로 모델/mmproj 자동 다운로드 후 기본 호스트/포트(`127.0.0.1:8080`)에서 서버 기동 확인.
- Health check 통과:
  ```
  GET http://127.0.0.1:8080/health
  → 200 OK, {"status":"ok"}
  ```
- 전사(transcription) 테스트 스크립트 작성: [tools/test_transcribe.py](tools/test_transcribe.py)
  - 임의 오디오 파일(ffmpeg가 읽을 수 있는 포맷)을 16kHz mono WAV로 변환
  - base64 인코딩 후 [design.md](design.md) §10.1 포맷(OpenAI 호환 `input_audio`)으로 `/v1/chat/completions`에 POST
  - 응답 텍스트와 소요 시간(초) 출력
- 아직 실제 오디오 샘플로 전사는 시도하지 않음 — 테스트용 한국어 샘플 준비 대기 중.

---

## 2. 시험 방법

### 2.1 사전 조건

- llama-server 실행 중 (§1의 커맨드)
- Python 3.10+, ffmpeg, `requests` 패키지 설치 완료

### 2.2 실행

```bash
python tools/test_transcribe.py <오디오파일경로>
```

옵션:

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--server` | llama-server 주소 | `http://localhost:8080` |
| `--model` | 요청 body의 `model` 필드 | `qwen3-asr` (단일 모델 로딩 시 llama-server가 보통 무시) |
| `--prompt` | 오디오와 함께 보낼 텍스트 프롬프트 | `"Transcribe this audio."` |
| `--keep-wav` | 변환된 임시 WAV 파일을 삭제하지 않고 보존 | 삭제함 |

### 2.3 평가 기준 ([design.md](design.md) §5A.4)

- CJK(한국어 포함) 콘텐츠는 **CER(문자 오류율)** 기준으로 평가한다. WER은 사용하지 않는다.
- 무음/노이즈 구간에서 실제 발화와 무관한 텍스트를 출력하는 **환각(hallucination)** 여부를 반드시 확인한다. Whisper 계열의 고질적 문제이며, Qwen3-ASR도 긴 오디오에서 결과 누락 버그(GitHub #21847)가 보고된 바 있어 30초 Chunk 조건에서 재현되는지 확인이 필요하다.
- 가능하면 Qwen3-ASR 0.6B / 1.7B와 Whisper large-v2(기준선)를 같은 샘플로 비교한다 ([design.md](design.md) §5A.2, §5A.7 실측 우선순위).

---

## 3. 다음 단계

1. 한국어 샘플 오디오 확보 (콜센터/전화망 노이즈 등 실사용 조건에 가까운 것 우선)
2. `tools/test_transcribe.py`로 실측 진행, 모델별(0.6B/1.7B) 및 기준선(Whisper large-v2) 비교
3. 결과를 아래 §4에 기록
4. 결과에 따라 [design.md](design.md) §4.2 보컬 분리 결정 게이트, §5A.5 언어별 라우팅 여부 판단

---

## 4. 실측 결과 (TBD)

_실측 진행 후 이 섹션에 모델별 CER, 환각 발생 여부, 처리 시간(초/청크)을 기록한다._
