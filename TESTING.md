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
- VRAM 실측 및 튜닝: 플래그 없이 `-hf`만으로 띄우면 `/props` 확인 결과 `n_ctx=65536`, `total_slots=4`가 기본값으로 잡혀 VRAM을 10GB 이상 소모(`nvidia-smi` 실측, RTX 5070 Ti 16GB 중 12GB 사용). 30초 Chunk 정책에는 그런 큰 컨텍스트가 불필요하므로 `--ctx-size 4096 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0`로 줄여서 재기동 → VRAM 약 4GB로 감소 확인. 튜닝된 커맨드는 [design.md](design.md#L218) §6.1에 반영.
- 실제 녹음(`rec1.m4a`, "마이크 테스트" 발화)으로 첫 전사 성공 확인. 이 과정에서 언어 오판(한국어→일본어) 발견 → 프롬프트에 `language: ko` 힌트를 주면 정상화됨을 확인, [design.md](design.md#L185) §5A.8/§8/§9에 `language` 설정으로 반영.
- 라벨링된 CJK 샘플 확보 스크립트 작성: [tools/fetch_samples.py](tools/fetch_samples.py) — HF `datasets-server` rows API로 `google/fleurs`(`validation` split)에서 한국어/일본어/중국어 샘플(오디오+정답 전사)을 저장소에 바이너리를 커밋하지 않고 로컬(`samples/`, gitignore 처리)로 받는다.
- auto vs 언어 힌트 CER 비교 스크립트 작성: [tools/eval_language_hint.py](tools/eval_language_hint.py) — §4 결과 참고.

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

### 2.4 라벨링된 샘플 확보 + auto/언어힌트 비교

```bash
# 1. 정답 전사가 딸린 CJK 샘플을 로컬로 받는다 (samples/, gitignore 처리됨)
python tools/fetch_samples.py --lang ko ja zh --count 3

# 2. 언어 힌트 유무에 따른 CER 차이를 측정한다 (temperature=0으로 고정해 순수 효과만 비교)
python tools/eval_language_hint.py --temperature 0
```

`eval_language_hint.py`는 각 샘플에 대해 `"Transcribe this audio."`(auto)와 `"language: {code}"`(힌트) 두 프롬프트로 각각 요청을 보내고, `samples/manifest.jsonl`의 정답 전사와 비교해 CER과 언어 오판 횟수를 계산해 `samples/language_hint_comparison.json`에 저장한다.

---

## 3. 다음 단계

1. **노이즈/배경음악이 섞인 샘플로 재측정** — §4의 결과는 Fleurs 낭독체(클린 오디오) 기준이라 실사용 조건(콜센터 노이즈, 배경음악 등)에서는 CER이 크게 오를 것으로 예상됨. §5A.6 콜센터 벤치마크(CER 22.72%)와의 격차 원인 확인 필요
2. 모델별(0.6B/1.7B) 및 기준선(Whisper large-v2) 비교
3. 결과에 따라 [design.md](design.md) §4.2 보컬 분리 결정 게이트, §5A.5 언어별 라우팅 여부 판단

---

## 4. 실측 결과

### 4.1 auto vs 언어 힌트 CER 비교 (2026-07-08, Qwen3-ASR-1.7B, Fleurs validation 9샘플, temperature=0)

`tools/eval_language_hint.py` 실행 결과:

| 파일 | 언어 | auto CER | 힌트 CER | 비고 |
|---|---|---|---|---|
| ja_000 | ja | 0.065 | 0.065 | 동일 |
| ja_001 | ja | 0.086 | 0.086 | 동일 |
| ja_002 | ja | 0.036 | 0.036 | 동일 |
| zh_000 | zh | 0.045 | 0.045 | 동일 |
| zh_001 | zh | 0.143 | 0.143 | 동일 |
| zh_002 | zh | 0.077 | 0.077 | 동일 |
| ko_000 | ko | 0.117 | 0.117 | 동일 |
| ko_001 | ko | 0.136 | **0.045** | "부쩍"→"무척" 단어 선택 차이로 정정됨 |
| ko_002 | ko | 0.140 | 0.140 | 동일 |
| **평균** | | **0.094** | **0.084** | 약 11% 상대 개선 |

**언어 오판: auto 0/9, 힌트 0/9.**

해석:
- 9개 중 8개는 auto/힌트 출력이 완전히 동일 — 문맥이 충분한 **긴 자연문장**에서는 언어 힌트의 영향이 거의 없다. 세 언어 모두 힌트 없이도 100% 정확히 감지됨.
- 유일하게 달라진 `ko_001`도 언어 오판이 아니라 단어 선택 차이였다.
- 즉, `rec1.m4a`("마이크 테스트")에서 관찰됐던 한국어→일본어 오판은 **짧고 외래어 위주라 언어 자체가 모호한 발화**에서만 두드러지는 문제로 보이며, 이 배치(정상 길이 문장)에서는 재현되지 않았다.
- 힌트 추가 비용은 사실상 0이고 손해 사례가 없으므로, 드물게 발생하는 위 케이스에 대한 안전장치로 `language` 설정(§5A.8)을 유지하는 것이 합리적.
- CER 절대값(4~14%)은 [design.md](design.md#L165) §5A.6 콜센터 노이즈 데이터 벤치마크(22.72%)보다 훨씬 낮음 — Fleurs는 깨끗한 낭독체 음성이라 더 쉬운 조건. **노이즈/배경음악이 섞인 실사용 조건에서 격차가 얼마나 벌어지는지는 미검증** (§3.1 다음 단계).

원본 데이터: `samples/language_hint_comparison.json` (gitignore 대상, 로컬에만 존재).
