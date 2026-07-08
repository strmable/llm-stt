# Media Transcriber (SRT Generator) - Design v2

문서 버전: 2.0 (2026-07-08)
이전 버전 대비 변경 요약: [부록 A](#부록-a-v1--v2-변경-이력) 참고

## 1. 개요

Media Transcriber는 오디오 또는 동영상 파일을 입력받아 음성을 텍스트(STT)로 변환하고 SRT 자막 파일을 생성하는 데스크탑 GUI 애플리케이션이다.

생성된 자막은 처리 중 실시간으로 화면에 표시되며, 사용자는 진행 상태를 확인하고 언제든 작업을 중단할 수 있어야 한다.

프로그램은 대용량 미디어 파일 처리를 고려하여 스트리밍 기반으로 동작하며, 기본적으로 임시 오디오 파일을 생성하지 않는다.

### v1 대비 핵심 변경

v1 설계 시점(2026-06 초)에는 로컬 OpenAI 호환 서버의 오디오 입력(`input_audio`) 지원이 불확실하여 구현이 중단되었다. 이후 llama.cpp `PR #24118`(2026-06-04 머지) 및 GGUF 재업로드(2026-06-05)로 **llama.cpp `llama-server` + Gemma 4 12B 조합에서 오디오 입력이 정상 동작함이 검증**되었다.

이에 따라 v2에서는:

* **1차 타깃 Provider를 로컬 `llama-server`(Gemma 4 12B)로 확정**하고, 검증된 실행 환경/요청 포맷을 설계에 반영한다.
* Google Gemini API는 보조 Provider로 유지한다 (로컬 GPU가 없는 환경용).
* v1의 §23.4 "검증 메모"에 있던 불확실성 항목들은 해소되어 본문에 확정 사항으로 반영한다.

---

## 2. 목표

### 지원 기능

* 오디오 파일 자막 생성
* 동영상 파일 자막 생성
* SRT 생성
* 실시간 결과 표시
* 클립보드 복사
* STT Provider 선택 (Local llama-server / Google Gemini API)
* Prompt 커스터마이징
* 이전 인식 결과를 컨텍스트로 전달 (Context Carryover)
* 작업 중단
* 상세 디버깅 로그 출력

---

## 3. 실행 형태

GUI와 콘솔을 동시에 사용하는 Hybrid 구조.

* GUI: 파일 선택, 설정 관리, 진행률 표시, 결과 표시, 작업 제어
* Console: 디버깅 로그, 처리 상태, 오류, 성능 분석 출력

---

## 4. 지원 입력 형식

* Audio: wav, mp3, aac, m4a, flac, ogg
* Video: mp4, mkv, webm, mov, avi

사용자는 별도 변환 작업 없이 파일을 바로 입력할 수 있어야 한다.

---

## 5. STT Provider

### 5.1 Local llama-server (기본, 1차 타깃)

llama.cpp `llama-server`의 OpenAI 호환 `/v1/chat/completions` 엔드포인트를 사용한다.

검증된 구성 (2026-06-05 이후):

| 항목 | 값 |
|---|---|
| 모델 | Gemma 4 12B (encoder-free Unified 아키텍처, native audio input) |
| GGUF | `unsloth/gemma-4-12b-it-GGUF` Q8_0 (11.8GB) — **2026-06-05 06:36 이후 재업로드분** |
| 프로젝터 | `mmproj-F16.gguf` (117MB, `gemma4uv` 통합 vision+audio 포맷) |
| llama.cpp | `PR #24118` 포함 빌드 (b9890대 이상 권장) |
| 오디오 제약 | 최대 30초, 16kHz mono 권장 |

주의: 2026-06-05 이전에 다운로드한 GGUF는 메타데이터 오류(SIGFPE 크래시, 텍스트 변환 버그)가 있으므로 SHA256을 확인하고 재다운로드해야 한다.

대안 모델: `google/gemma-4-12B-it-qat-q4_0-gguf` (QAT, 6.98GB). 단, 6/5 수정 이후 재빌드 여부는 별도 확인 필요 — 오디오 용도로는 검증된 unsloth GGUF를 우선한다.

**LM Studio / Ollama는 지원 대상이 아니다.** LM Studio는 번들 llama.cpp가 upstream을 즉시 따라가지 않아 버전에 따라 동작이 불확실하고, Ollama는 OpenAI 호환 레이어에 오디오 필드 자체가 없다(관련 PR `#15243` 미병합). 자세한 내용은 [부록 B](#부록-b-비대상-서버-호환성-메모) 참고.

### 5.2 Google Gemini API (보조)

`generateContent` API의 `inlineData` 필드로 오디오를 전달한다.

사용 가능한 모델 예시: `gemini-3.1-flash-lite`(권장), `gemini-3-flash-preview`(시험용). 상세 모델 선택·요청 샘플은 [부록 D](#부록-d-gemini-api-이용-가이드-보조-provider) 참고.

로컬 GPU가 없거나 llama-server 준비가 어려운 환경에서 사용한다.

---

## 6. 로컬 서버 실행 환경 (llama-server)

프로그램 자체는 HTTP 클라이언트일 뿐이며 서버 구동은 사용자 책임이다. 다만 README와 설정 화면에 아래 검증된 실행 방법을 안내한다.

### 6.1 실행 커맨드

```bash
./llama-server \
  --model gemma-4-12b-it-Q8_0.gguf \
  --mmproj mmproj-F16.gguf \
  --n-gpu-layers 99 \
  --ctx-size 32768 \
  --flash-attn on \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --jinja \
  --host 0.0.0.0 --port 8088
```

* VRAM: Q8_0 기준 약 14GB
* 성능 참고: HF Transformers 대비 5~7배 빠른 디코딩 (64~131 t/s)

### 6.2 Windows + RTX 50시리즈(Blackwell) 주의사항

* llama.cpp 공식 릴리스는 CUDA 12.4 / CUDA 13.3 두 가지 사전빌드를 제공.
* 권장 절차: ① CUDA 13.3 빌드를 먼저 사용 (최신 드라이버와의 DLL 충돌 회피) → ② `llama-bench`로 pp512/tg128 확인, 비정상적으로 낮으면(cuBLAS 폴백 패턴) CUDA 12.4 빌드와 비교 → ③ NVIDIA 드라이버 최신 유지 (초기 Blackwell 드라이버의 `sharedMemPerBlockOptin` 버그가 원인이었음, llama.cpp `#23385`).

### 6.3 연결 확인 (Connection Test)

설정 창에 **[Test Connection]** 버튼을 제공한다 (v2 신규).

동작:

1. 1초 분량의 무음 WAV(16kHz mono)를 메모리에서 생성
2. 실제 STT 요청과 동일한 포맷(§10.1)으로 서버에 전송
3. 성공: HTTP 200 + `choices[0].message.content` 존재 → "OK" 표시
4. 실패: 상태 코드/오류 메시지를 표시 (예: `input_audio` 미지원 서버, 연결 거부, 타임아웃)

v1 검증 메모의 "구현 단계에서 런타임 검증이 필수" 권고를 기능으로 반영한 것이다.

---

## 7. 메인 UI

### 상단

```text
+---------------------------------------------------+
| File : [ Selected File ]           [Select File] |
+---------------------------------------------------+
```

* 파일 경로 표시, Drag & Drop 지원, 파일 선택

### 중앙

```text
+---------------------------------------------------+
|              SRT Output TextBox                   |
+---------------------------------------------------+
```

* Read Only, 자동 스크롤, 실시간 갱신

### 하단

```text
+---------------------------------------------------+
| Progress Bar                                      |
+---------------------------------------------------+

[Transcript] [Copy] [Settings]     (작업 중: [Stop] [Copy] [Settings])
```

---

## 8. 설정 창

Modal Dialog 형태.

### Provider 설정

```text
Provider

(o) Local llama-server (OpenAI-Compatible)
( ) Google Gemini API
```

기본값은 Local llama-server. 선택된 Provider에 따라 설정 영역 표시.

### Local llama-server 설정

```text
API URL   : http://localhost:8088/v1/chat/completions
Model Name: gemma-4-12b
[Test Connection]
```

안내 문구(설정 화면 하단 고정 표시):

> 검증 환경: llama.cpp llama-server(2026-06-05 이후 빌드) + Gemma 4 12B GGUF + gemma4uv mmproj.
> LM Studio / Ollama 등 기타 서버는 동작을 보증하지 않습니다.
> llama.cpp의 오디오 입력은 "experimental"로 표기되어 있어 인식 품질이 저하될 수 있습니다.

### Google Gemini API 설정

```text
API Key   : AIza...
Model Name: gemini-3.1-flash-lite
```

### Prompt 설정

멀티라인 텍스트 박스 + `[Load] [Save] [Save As]` 버튼.

#### Template Variables

| 변수 | 설명 |
|---|---|
| `{{context}}` | 직전에 처리된 1개 Chunk의 인식 결과 텍스트 (없으면 빈 문자열) |

기본 프롬프트:

```text
Transcribe the following audio chunk to text.
Use the previous context only to keep terminology, names and
sentence flow consistent. Do not repeat the previous context
in your output.

Previous context:
{{context}}

Output only the transcription of the current audio chunk.
```

* 사용자가 `{{context}}`를 제거하면 컨텍스트는 전달되지 않는다 (§16 참고).

### 모델 파라미터

```text
Temperature   Top-P   Top-K   Max Tokens
```

* Top-K는 v2 신규 (Gemma 4 권장 샘플링 값이 top-k 64를 포함하므로 노출).
* Gemini API 사용 시 Top-K는 `generationConfig.topK`로 전달.

### Debug 옵션

```text
[ ] Save Chunk WAV Files     (기본 OFF)
```

### 버튼

`[OK]` — config.json 저장 후 닫기 / `[Cancel]` — 변경사항 폐기

---

## 9. 설정 파일

config.json

```json
{
  "provider": "local_api",

  "local_api": {
    "url": "http://localhost:8088/v1/chat/completions",
    "model": "gemma-4-12b",
    "disable_thinking": true
  },

  "gemini": {
    "api_key": "",
    "model": "gemini-3.1-flash-lite"
  },

  "llm": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_tokens": 4096
  },

  "prompt": {
    "template": "Transcribe the following audio chunk to text.\n\nPrevious context:\n{{context}}\n\nOutput only the transcription of the current audio chunk."
  },

  "debug": {
    "save_chunk_wav": false
  },

  "logging": {
    "level": "INFO"
  }
}
```

v1 대비 변경:

* `provider` 기본값 `gemini` → `local_api`
* 기본 URL 포트 `8080` → `8088` (검증 커맨드 기준)
* `llm.top_k` 추가, `temperature` 기본값 0.2 → 1.0 (Gemma 4 권장값)
* `local_api.disable_thinking` 추가 — true 시 요청에 `chat_template_kwargs: {"enable_thinking": false}` 포함 (Gemma 4의 thinking 출력 억제, STT 용도에서는 항상 억제가 바람직)

---

## 10. STT API 요청 형식

Chunk WAV(BytesIO 메모리 버퍼)는 **Base64 인코딩 문자열**로 JSON Body에 담아 전송한다. 멀티파트 업로드는 사용하지 않는다. Base64의 `data`는 **prefix 없는 순수 base64 문자열**이다 (`data:audio/wav;base64,...` Data URL 형식 금지 — OpenAI 스펙 및 llama.cpp 모두 순수 base64 사용).

### 10.1 Local llama-server (OpenAI 호환)

`input_audio` content type 사용. **llama-server + Gemma 4 12B 조합에서 동작 검증 완료 (2026-06-05 이후).**

```json
POST {local_api.url}
Content-Type: application/json

{
  "model": "gemma-4-12b",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "<프롬프트 텍스트 (Context 치환 완료)>" },
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64 encoded WAV bytes>",
            "format": "wav"
          }
        }
      ]
    }
  ],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
  "max_tokens": 4096,
  "chat_template_kwargs": { "enable_thinking": false }
}
```

Python 예시:

```python
import base64

audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("ascii")

payload = {
    "model": cfg.local_api.model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "input_audio",
             "input_audio": {"data": audio_b64, "format": "wav"}},
        ],
    }],
    "temperature": cfg.llm.temperature,
    "top_p": cfg.llm.top_p,
    "top_k": cfg.llm.top_k,
    "max_tokens": cfg.llm.max_tokens,
}
if cfg.local_api.disable_thinking:
    payload["chat_template_kwargs"] = {"enable_thinking": False}
```

참고: `top_k`, `chat_template_kwargs`는 OpenAI 표준이 아닌 llama-server 확장 필드이나, llama-server는 미지원 필드를 무시하므로 안전하다.

### 10.2 Google Gemini API

`generateContent` API의 `inlineData` 필드 사용.

```json
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}
Content-Type: application/json

{
  "contents": [
    {
      "parts": [
        { "text": "<프롬프트 텍스트 (Context 치환 완료)>" },
        {
          "inline_data": {
            "mime_type": "audio/wav",
            "data": "<base64 encoded WAV bytes>"
          }
        }
      ]
    }
  ],
  "generationConfig": {
    "temperature": 1.0,
    "topP": 0.95,
    "topK": 64,
    "maxOutputTokens": 4096
  }
}
```

제약: inline 데이터 합산 요청 크기 20MB 미만. 30초 Chunk(16kHz mono WAV ≈ 1MB)는 충분히 만족. 모델 선택 및 무료 티어 특성은 [부록 D](#부록-d-gemini-api-이용-가이드-보조-provider) 참고.

### 10.3 공통 사항

* Base64 인코딩은 표준 라이브러리(`base64`) 사용.
* Base64 인코딩으로 인한 약 33% 크기 증가 및 직렬화 오버헤드는 DEBUG 로그에 Chunk별 처리 시간으로 기록.
* `{{context}}` 치환은 두 Provider 모두 텍스트 파트에서 동일하게 수행.
* 서버가 요청을 처리하지 못하는 경우 `[ERROR] API ...` 형태로 로깅하고 해당 Chunk를 실패 처리.

---

## 11. 오디오 처리 파이프라인

```text
Input Media File
        │
        ▼
FFmpeg Decode Stream
        │
        ▼
16kHz Mono PCM Stream
        │
        ▼
Silero VAD
        │
        ▼
Speech Chunk Extraction (최대 30초)
        │
        ▼
WAV Memory Buffer (BytesIO)
        │
        ▼
Prompt Build (Template + 직전 Chunk 결과 → {{context}})
        │
        ▼
Selected STT Provider (llama-server / Gemini)
        │
        ▼
Transcript Result
        │
        ▼
SRT Generation
        │
        ▼
Context 갱신 (현재 결과를 다음 Chunk의 {{context}}로 저장)
        │
        ▼
UI Update
```

### FFmpeg 처리 정책

모든 입력 파일은 FFmpeg로 처리하며, 동영상은 오디오 트랙을 자동 추출한다. 사용자가 사전 변환할 필요 없다.

출력 형식 통일: PCM / 16kHz / Mono

```bash
ffmpeg -i input_file -f s16le -ac 1 -ar 16000 -
```

이 형식은 Gemma 4의 권장 입력(16kHz mono)과 일치한다.

### VAD

Silero VAD 사용. 목적: 무음 제거, API 비용/부하 절감, 처리 속도 향상.

---

## 12. Chunk 생성 정책

분할 조건 (둘 중 먼저 만족 시 Chunk 종료):

1. **최대 30초** — Gemma 4의 오디오 입력 상한(30초)과 일치하는 **하드 리밋**. v1에서는 단순 권장 정책이었으나 v2에서는 모델 제약이므로 초과 금지.
2. 충분한 무음 구간 감지

---

## 13. 메모리 처리 정책

* 기본: 임시 오디오 파일 생성 금지. 모든 처리는 메모리 기반 (`BytesIO`, `numpy`).
* Debug Mode 활성화 시에만 `temp/chunk_NNNN.wav` 저장 가능.
* Cleanup: 프로그램 시작/종료/작업 취소 시 `temp/` 정리.

---

## 14. 실시간 출력

Chunk 처리 완료 시:

1. STT 수행 (직전 Chunk 결과를 `{{context}}`로 주입)
2. SRT 생성
3. 내부 버퍼 추가
4. 현재 결과를 다음 Chunk용 Context로 저장
5. TextBox 갱신
6. ProgressBar 갱신

사용자는 처리 중에도 결과를 확인할 수 있어야 한다.

---

## 15. 작업 제어

### 중단

* 작업 시작 시 `Transcript` 버튼이 `Stop`으로 변경.
* Stop 클릭 → `Cancel Requested` 상태 → 현재 Chunk 처리 완료 후 종료 (강제 종료 금지) → 버튼 복원.

### Progress 표시

```text
처리된 오디오 시간 / 전체 오디오 길이    예: 00:15:00 / 01:00:00 (25%)
```

### 클립보드 복사

Copy 버튼 클릭 시 TextBox의 전체 SRT 내용을 클립보드에 복사.

---

## 16. Context Carryover

### 목적

Chunk 경계에서의 문맥 단절(고유명사/용어 비일관, 잘린 문장, 어조 단절)을 완화하기 위해 직전 Chunk의 인식 결과 텍스트를 다음 요청 프롬프트에 전달한다.

### 동작 방식

1. Worker는 가장 최근 완료된 1개 Chunk의 인식 결과 텍스트(SRT가 아닌 순수 텍스트)를 메모리에 보관.
2. 첫 Chunk의 `{{context}}`는 빈 문자열.
3. Chunk N 처리 시 Chunk N-1의 결과를 `{{context}}`에 치환.
4. 완료 시 Context를 갱신 (항상 1블록만 유지).
5. SRT 출력에는 Context가 포함되지 않는다.

### 제약

* Context는 최대 1 Chunk 분량 (메모리/토큰 고려).
* 작업 취소 또는 새 파일 로드 시 초기화.
* Provider 공통 동작.
* 프롬프트에서 `{{context}}` 미사용 시 자연스럽게 비활성화 (별도 ON/OFF 불필요).

---

## 17. Console Logging

레벨: DEBUG / INFO / WARNING / ERROR

```text
[INFO] Application started
[INFO] Loading file: lecture.mp4
[INFO] Duration: 01:23:15
[INFO] Starting ffmpeg decoder
[INFO] Audio format: 16000Hz Mono PCM
[INFO] Starting VAD
[INFO] Chunk #1 Start=00:00:00 End=00:00:28
[INFO] Sending request
[INFO] Chunk #1 completed
[INFO] SRT updated
[INFO] Progress 2%

[DEBUG] Chunk #1 context: (empty)
[DEBUG] Chunk #2 context: "...이전 청크의 인식 결과..."
[DEBUG] Chunk #2 encode+request time: 3.42s

[ERROR] API timeout
[ERROR] Chunk #15 failed

[INFO] Cancel requested
[INFO] Waiting current chunk
[INFO] Worker stopped
```

---

## 18. 스레드 구조

* **GUI Thread**: UI, ProgressBar, Buttons, TextBox
* **Worker Thread**: FFmpeg, PCM Stream, VAD, Chunking, Prompt 조합(Context 치환), STT API, SRT 생성, Context 상태 관리
* **Signal**: `progressChanged`, `chunkCompleted`, `finished`, `error` — UI 직접 접근 금지, Signal/Slot 사용

---

## 19. 기술 스택

| 영역 | 기술 |
|---|---|
| GUI | PySide6 |
| 오디오 처리 | FFmpeg |
| VAD | Silero VAD |
| HTTP | httpx |
| 설정 관리 | json |
| 비동기 처리 | QThread |
| 오디오 버퍼 | numpy, soundfile, io.BytesIO |
| 로깅 | logging |

---

## 20. 오류 처리 정책

* Chunk 단위 실패 허용: 특정 Chunk의 API 오류 시 해당 Chunk만 실패 처리하고 다음 Chunk 계속 진행. 실패 Chunk는 SRT에 `[TRANSCRIPTION FAILED]` 플레이스홀더로 표기.
* 실패 Chunk의 Context는 갱신하지 않고 직전 성공 결과 유지.
* 재시도: 타임아웃/일시 오류 시 1회 재시도 후 실패 처리 (v2 신규).
* 연속 N회(기본 5회) 실패 시 작업 자동 중단 및 오류 안내 (서버 다운 등 환경 문제로 판단).

---

## 21. 향후 확장

* Save SRT / Save TXT
* OpenAI Audio API 지원
* Whisper Backend 지원
* 병렬 Chunk 처리 (`llama-server --parallel N` 활용)
* 화자 구분, 맞춤법 교정, 문장 정리
* 자막 포맷 개선, 번역 자막 생성, 다국어 지원
* Context Carryover Chunk 개수 설정 (현재 1개 고정)
* `local_api.request_format` 옵션 (비표준 포맷 서버 대응 — 현재는 불필요, 필요 시 추가)

---

## 부록 A. v1 → v2 변경 이력

| # | 변경 | 근거 |
|---|---|---|
| 1 | 기본 Provider를 Gemini → Local llama-server로 변경 | llama.cpp `PR #24118`(2026-06-04) 이후 오디오 입력 동작 검증 완료 |
| 2 | Local Provider를 "OpenAI-Compatible 일반"에서 "llama-server 특정"으로 구체화 | LM Studio/Ollama 미지원 확인, llama-server만 검증됨 |
| 3 | 검증된 모델/실행 커맨드 명시 (§5.1, §6.1) | unsloth GGUF Q8_0 + mmproj-F16 (6/5 이후 재업로드분) 동작 확인 |
| 4 | [Test Connection] 기능 추가 (§6.3) | v1 §23.4 "런타임 검증 필수" 권고의 기능화 |
| 5 | `top_k` 파라미터, `disable_thinking` 옵션 추가 | Gemma 4 권장 샘플링(temp 1.0, top-p 0.95, top-k 64), thinking 억제 필요 |
| 6 | 기본 temperature 0.2 → 1.0 | Gemma 4 권장값 |
| 7 | Chunk 최대 30초를 하드 리밋으로 격상 (§12) | Gemma 4 오디오 입력 상한 30초 |
| 8 | 기본 포트 8080 → 8088 | 검증 커맨드 기준 |
| 9 | 오류 처리 정책 신설 (§20) | Chunk 실패/서버 다운 시 동작 명세화 |
| 10 | v1 §23.4 검증 메모 삭제 | 불확실성 해소, 확정 사항은 본문 반영 |
| 11 | RTX 50시리즈 CUDA 빌드 주의사항 추가 (§6.2) | Blackwell 드라이버 버그(#23385) 및 CUDA 12/13 선택 이슈 |
| 12 | Gemini API 이용 가이드 부록 신설 (부록 D), 기본 Gemini 모델을 최신값으로 갱신 | 2.5-flash 등 구모델 대신 `gemini-3.1-flash-lite`(오디오 입력 지원, 무료 티어 넉넉) 확인 |

## 부록 B. 비대상 서버 호환성 메모 (2026-07 기준)

| 서버 | `input_audio` 지원 | 비고 |
|---|---|---|
| llama.cpp `llama-server` | ✅ 지원 (experimental) | Gemma 4 + `gemma4uv` mmproj, 2026-06-05 이후 빌드/GGUF 필요 |
| LM Studio | ⚠️ 버전 의존 | 번들 llama.cpp가 upstream을 즉시 따라가지 않음. 공식 문서에 `input_audio` 언급 없음. 지원 대상 아님 |
| Ollama | ❌ 미지원 | OpenAI 호환 레이어에 오디오 필드 없음. 자체 API용 PR `#15243` 미병합 |
| vLLM | ❌ 미지원 보고 | 이슈 #19977 |

## 부록 C. 참고 링크

* [llama.cpp PR #24118: Fix Gemma 4 Unified conversion](https://github.com/ggml-org/llama.cpp/pull/24118)
* [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
* [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
* [Running Gemma-4-12B Audio on llama.cpp (note.com)](https://note.com/unco3/n/n871e994d27b2?hl=en)
* [Blackwell CUDA Toolkit 이슈 (llama.cpp #23385)](https://github.com/ggml-org/llama.cpp/issues/23385)
* [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)

## 부록 D. Gemini API 이용 가이드 (보조 Provider)

로컬 GPU가 없거나 `llama-server` 준비가 어려운 환경에서 사용하는 보조 Provider(§5.2, §10.2)에 대한 상세 이용 가이드다. 오디오 입력(STT) 용도로 **실제 사용 가능한 최신 모델과 무료 티어 특성, 요청 샘플**을 정리한다.

### D.1 사용 가능 모델 (오디오 입력 STT 용도)

Google AI for Developers 문서(2026-07 기준) 확인 결과, 아래 두 모델이 오디오 입력을 정식 지원하며 본 프로그램의 STT 용도에 적합하다.

| 모델 코드 | 상태 | 오디오 입력 | 무료 티어 1일 한도(참고) | 비고 |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | **Stable (권장)** | ✅ 지원 | 넉넉함(약 300~500 RPD 수준) | 저지연·저비용, 고빈도 STT에 최적. 입력 Text/Image/Video/**Audio**/PDF, 출력 Text |
| `gemini-3-flash-preview` | Preview | ✅ 지원 | 약 20 RPD로 제한적 | 시험용으로만 권장. 한도가 낮아 대량 Chunk 처리에는 부적합 |

권장: **기본 모델은 `gemini-3.1-flash-lite`로 설정한다.** 무료 티어 한도가 넉넉해 다수 Chunk를 순차 전송하는 본 프로그램 구조와 잘 맞는다. `gemini-3-flash-preview`는 1일 요청 한도가 낮아(≈20) 기능 검증·소량 테스트 용도로만 사용한다.

`gemini-3.1-flash-lite` 주요 스펙:

* 입력 데이터 타입: Text, Image, Video, **Audio**, PDF / 출력: Text
* 입력 토큰 한도 1,048,576 / 출력 토큰 한도 65,536
* Knowledge cutoff: 2025-01
* **Audio generation·Live API는 미지원** — 파일 기반 전사/요약/분석만 가능(실시간 음성 대화 불가). 본 프로그램은 파일 Chunk 전사이므로 제약 없음.

> **주의 (Gemma vs Gemini):** Gemini API로 호스팅되는 **Gemma 4 모델(`gemma-4-31b-it`, `gemma-4-26b-a4b-it`)은 오디오 입력이 불가**하다("Audio input modality is not enabled" 오류). Gemini API 경로에서 오디오 STT가 필요하면 반드시 위 **Gemini 계열 모델**을 사용해야 한다. (오디오 학습된 Gemma 4 E2B/E4B/12B 변형은 Gemini API에 호스팅되지 않으며 로컬 실행이 필요 — 이 경우가 §5.1 llama-server 경로다.)

### D.2 오디오 입력 방식

Gemini API는 두 가지 오디오 입력 방식을 제공한다. 본 프로그램은 30초 Chunk(16kHz mono WAV ≈ 1MB)를 다루므로 **인라인(inline) 방식을 사용**한다(§10.2와 동일).

| 방식 | 사용 조건 | 본 프로그램 채택 |
|---|---|---|
| Inline data (`inline_data`) | 요청 총 크기 20MB 미만 | ✅ 채택 (Chunk가 항상 20MB 미만) |
| Files API 업로드 후 참조 | 총 크기 20MB 이상 또는 파일 재사용 | 미사용 |

지원 오디오 MIME 타입: `audio/wav`, `audio/mp3`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac`. 본 프로그램은 WAV(`audio/wav`)로 통일한다.

참고: Gemini는 오디오 1초를 32토큰으로 계산(1분 ≈ 1,920토큰), 단일 프롬프트 최대 9.5시간, 멀티채널은 모노로 병합. 30초 Chunk는 약 960토큰으로 매우 여유롭다.

### D.3 요청 샘플

#### D.3.1 REST (curl)

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key=${GEMINI_API_KEY}" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "contents": [{
      "parts": [
        { "text": "<프롬프트 텍스트 (Context 치환 완료)>" },
        {
          "inline_data": {
            "mime_type": "audio/wav",
            "data": "<base64 encoded WAV bytes>"
          }
        }
      ]
    }],
    "generationConfig": {
      "temperature": 1.0,
      "topP": 0.95,
      "topK": 64,
      "maxOutputTokens": 4096
    }
  }'
```

#### D.3.2 Python (httpx — 본 프로그램 기술 스택 기준)

§10.2의 요청 형식을 그대로 사용하며, Chunk WAV 버퍼를 순수 base64(prefix 없음)로 인코딩해 전송한다.

```python
import base64
import httpx

def transcribe_gemini(wav_buffer, prompt_text, cfg):
    audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("ascii")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.gemini.model}:generateContent?key={cfg.gemini.api_key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
            ],
        }],
        "generationConfig": {
            "temperature": cfg.llm.temperature,
            "topP": cfg.llm.top_p,
            "topK": cfg.llm.top_k,
            "maxOutputTokens": cfg.llm.max_tokens,
        },
    }

    resp = httpx.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
```

#### D.3.3 공식 SDK (google-genai) — 참고

프로토타이핑 시 공식 SDK를 쓰면 base64/엔드포인트 처리를 SDK가 대신한다. 본 프로그램은 httpx 직접 호출을 표준으로 하되, 아래는 검증용 참고 코드다.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=cfg.gemini.api_key)
resp = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=[
        prompt_text,
        types.Part.from_bytes(data=wav_buffer.getvalue(), mime_type="audio/wav"),
    ],
    config=types.GenerateContentConfig(
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        top_k=cfg.llm.top_k,
        max_output_tokens=cfg.llm.max_tokens,
    ),
)
print(resp.text)
```

### D.4 응답 파싱

성공 응답에서 전사 텍스트 경로는 다음과 같다.

```text
candidates[0].content.parts[0].text
```

* 안전 필터 등으로 `candidates`가 비거나 `finishReason`이 `STOP`이 아닐 수 있으므로 파싱 전 존재 여부를 확인하고, 실패 시 §20 오류 처리 정책(Chunk 단위 실패 허용, 1회 재시도)을 따른다.
* `{{context}}` 치환은 텍스트 파트에서 수행(§10.3 공통 사항과 동일).

### D.5 설정 반영

§9 config.json의 `gemini` 블록 기본 모델을 최신 권장값으로 갱신한다.

```json
"gemini": {
  "api_key": "",
  "model": "gemini-3.1-flash-lite"
}
```

> 참고: 본문 §5.2 / §8 설정 창의 Gemini 모델 예시도 본 부록의 최신 모델(`gemini-3.1-flash-lite` 기본, `gemini-3-flash-preview` 시험용)로 통일 적용한다.

### D.6 참고 링크

* [Gemini API — Audio understanding](https://ai.google.dev/gemini-api/docs/audio)
* [Gemini 3.1 Flash-Lite 모델 페이지](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
* [Gemini API 모델 목록](https://ai.google.dev/gemini-api/docs/models)
* [Run Gemma with the Gemini API (오디오 미지원 — 대조용)](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
