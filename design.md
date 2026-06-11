# Media Transcriber (SRT Generator) - Design

## 1. 개요

Media Transcriber는 오디오 또는 동영상 파일을 입력받아 음성을 텍스트(STT)로 변환하고 SRT 자막 파일을 생성하는 데스크탑 GUI 애플리케이션이다.

생성된 자막은 처리 중 실시간으로 화면에 표시되며, 사용자는 진행 상태를 확인하고 언제든 작업을 중단할 수 있어야 한다.

프로그램은 대용량 미디어 파일 처리를 고려하여 스트리밍 기반으로 동작하며, 기본적으로 임시 오디오 파일을 생성하지 않는다.

---

## 2. 목표

### 지원 기능

* 오디오 파일 자막 생성
* 동영상 파일 자막 생성
* SRT 생성
* 실시간 결과 표시
* 클립보드 복사
* STT Provider 선택
* Prompt 커스터마이징
* 이전 인식 결과를 컨텍스트로 전달 (Context Carryover)
* 작업 중단
* 상세 디버깅 로그 출력

---

## 3. 실행 형태

프로그램은 GUI와 콘솔을 동시에 사용하는 Hybrid 구조로 동작한다.

```text
GUI Window
+
Console Window
```

### GUI 역할

* 파일 선택
* 설정 관리
* 진행률 표시
* 결과 표시
* 작업 제어

### Console 역할

* 디버깅 로그 출력
* 처리 상태 출력
* 오류 출력
* 성능 분석

---

## 4. 지원 입력 형식

### Audio

* wav
* mp3
* aac
* m4a
* flac
* ogg

### Video

* mp4
* mkv
* webm
* mov
* avi

사용자는 별도 변환 작업 없이 파일을 바로 입력할 수 있어야 한다.

---

## 5. STT Provider

### 지원 Provider

#### Google Gemini API

사용 가능한 모델 예시

* gemma-3n-e2b-it
* gemma-3n-e4b-it
* gemini-2.5-flash
* gemini-2.5-pro

#### Local OpenAI-Compatible API

사용 가능한 모델 예시

* gemma-4-12b
* qwen-audio
* 기타 OpenAI 호환 모델

---

## 6. 메인 UI

### 상단

```text
+---------------------------------------------------+
| File : [ Selected File ]           [Select File] |
+---------------------------------------------------+
```

기능

* 파일 경로 표시
* Drag & Drop 지원
* 파일 선택

---

### 중앙

```text
+---------------------------------------------------+
|                                                   |
|              SRT Output TextBox                   |
|                                                   |
+---------------------------------------------------+
```

특징

* Read Only
* 자동 스크롤
* 실시간 갱신

---

### 하단

```text
+---------------------------------------------------+
| Progress Bar                                      |
+---------------------------------------------------+

[Transcript] [Copy] [Settings]
```

작업 중

```text
+---------------------------------------------------+
| Progress Bar                                      |
+---------------------------------------------------+

[Stop] [Copy] [Settings]
```

---

## 7. 설정 창

Modal Dialog 형태로 구현

---

### Provider 설정

```text
Provider

(o) Google Gemini API
( ) Local OpenAI-Compatible API
```

선택된 Provider에 따라 설정 영역 표시

---

### Google Gemini API 설정

```text
API Key

Model Name
```

예

```text
API Key
AIza...

Model
gemma-3n-e4b-it
```

---

### Local API 설정

```text
API URL

Model Name
```

예

```text
http://localhost:8080/v1/chat/completions

gemma-4-12b
```

---

### Prompt 설정

멀티라인 텍스트 박스 제공

버튼

```text
[Load]
[Save]
[Save As]
```

#### 지원 변수 (Template Variables)

프롬프트 텍스트 내에서 아래 변수를 사용할 수 있으며, 실행 시 실제 값으로 치환되어 STT Provider에 전달된다.

| 변수 | 설명 |
|---|---|
| `{{context}}` | 직전에 처리된 1개 Chunk의 인식 결과 텍스트 (없으면 빈 문자열로 치환) |

예시 프롬프트

```text
Transcribe the following audio chunk to text.
Use the previous context only to keep terminology, names and
sentence flow consistent. Do not repeat the previous context
in your output.

Previous context:
{{context}}

Output only the transcription of the current audio chunk.
```

* 기본 제공 프롬프트(Default Prompt)에는 `{{context}}` 변수가 포함되어 있다.
* 사용자가 프롬프트에서 `{{context}}` 변수를 제거하면 컨텍스트는 전달되지 않는다.
* 자세한 동작 방식은 [22. Context Carryover](#22-context-carryover) 참고.

---

### 모델 파라미터

```text
Temperature

Top-P

Max Tokens
```

---

### Debug 옵션

```text
[ ] Save Chunk WAV Files
```

기본값

```text
OFF
```

설명

* 기본적으로 임시 WAV 파일 생성 금지
* 디버깅 목적으로만 저장

---

### 설정 버튼

```text
[OK] [Cancel]
```

OK

* config.json 저장
* 창 닫기

Cancel

* 변경사항 폐기

---

## 8. 설정 파일

config.json

```json
{
  "provider": "gemini",

  "gemini": {
    "api_key": "",
    "model": "gemma-3n-e4b-it"
  },

  "local_api": {
    "url": "http://localhost:8080/v1/chat/completions",
    "model": "gemma-4-12b"
  },

  "llm": {
    "temperature": 0.2,
    "top_p": 0.95,
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

---

## 9. 오디오 처리 파이프라인

### 전체 흐름

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
Speech Chunk Extraction
        │
        ▼
WAV Memory Buffer (BytesIO)
        │
        ▼
Prompt Build
(Template + Previous Chunk Transcript → {{context}})
        │
        ▼
Selected STT Provider
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

---

## 10. FFmpeg 처리 정책

모든 입력 파일은 FFmpeg를 통해 처리한다.

사용자는 다음 작업을 수행할 필요가 없다.

* mp3 변환
* wav 변환
* aac 변환
* 오디오 추출

동영상 입력 시 FFmpeg가 자동으로 오디오 트랙을 추출한다.

---

### FFmpeg 출력 형식

모든 입력은 아래 형식으로 통일

```text
PCM
16kHz
Mono
```

예시

```bash
ffmpeg \
-i input_file \
-f s16le \
-ac 1 \
-ar 16000 \
-
```

---

## 11. VAD 처리

권장 라이브러리

```text
Silero VAD
```

목적

* 무음 제거
* API 비용 절감
* 처리 속도 향상

---

## 12. Chunk 생성 정책

분할 조건

### 조건 1

최대 30초

### 조건 2

충분한 무음 구간 감지

둘 중 먼저 만족 시 Chunk 종료

예

```text
0~18초

18~44초

44~72초

72~101초
```

---

## 13. 메모리 처리 정책

기본 정책

```text
No Temporary Audio Files
```

Chunk WAV 파일 생성 금지

모든 처리는 메모리 기반 수행

사용 기술

```text
BytesIO
numpy
```

---

### Debug Mode

활성화 시

```text
temp/
chunk_0001.wav
chunk_0002.wav
...
```

생성 가능

---

### Cleanup

프로그램 시작 시

```text
temp/
```

정리

프로그램 종료 시

```text
temp/
```

정리

작업 취소 시

```text
temp/
```

정리

---

## 14. 실시간 출력

Chunk 처리 완료 시

1. STT 수행 (이전 Chunk 인식 결과를 `{{context}}`로 프롬프트에 주입)
2. SRT 생성
3. 내부 버퍼 추가
4. 현재 Chunk 인식 결과를 다음 Chunk를 위한 Context로 저장
5. TextBox 갱신
6. ProgressBar 갱신

사용자는 처리 중에도 결과를 확인할 수 있어야 한다.

---

## 15. 중단 기능

작업 시작 시

```text
Transcript
```

버튼을

```text
Stop
```

으로 변경

사용자가 Stop 클릭 시

```text
Cancel Requested
```

상태로 변경

현재 Chunk 처리 완료 후 종료

강제 종료 금지

종료 후

```text
Stop
```

→

```text
Transcript
```

복원

---

## 16. Progress 표시

기준

```text
처리된 오디오 시간
/
전체 오디오 길이
```

예

```text
00:15:00 / 01:00:00

25%
```

---

## 17. 클립보드 복사

Copy 버튼 클릭 시

현재 TextBox의 전체 SRT 내용을 클립보드에 복사한다.

---

## 18. Console Logging

지원 레벨

```text
DEBUG
INFO
WARNING
ERROR
```

---

### 로그 예시

```text
[INFO] Application started

[INFO] Loading file:
lecture.mp4

[INFO] Duration:
01:23:15

[INFO] Starting ffmpeg decoder

[INFO] Audio format:
16000Hz Mono PCM

[INFO] Starting VAD

[INFO] Chunk #1
Start=00:00:00
End=00:00:28

[INFO] Sending request

[INFO] Chunk #1 completed

[INFO] SRT updated

[INFO] Progress 2%
```

---

### Context 관련 로그 예시

```text
[DEBUG] Chunk #1 context: (empty)

[DEBUG] Chunk #2 context: "...이전 청크의 인식 결과 텍스트..."
```

---

### 오류 로그

```text
[ERROR] API timeout

[ERROR] Chunk #15 failed
```

---

### 취소 로그

```text
[INFO] Cancel requested

[INFO] Waiting current chunk

[INFO] Worker stopped
```

---

## 19. 스레드 구조

### GUI Thread

담당

* UI
* ProgressBar
* Buttons
* TextBox

---

### Worker Thread

담당

* FFmpeg
* PCM Stream
* VAD
* Chunking
* Prompt 조합 (Context 치환)
* STT API
* SRT 생성
* Context 상태 관리

---

### Signal

```text
progressChanged

chunkCompleted

finished

error
```

UI 직접 접근 금지

Signal/Slot 사용

---

## 20. 기술 스택

GUI

* PySide6

오디오 처리

* FFmpeg

VAD

* Silero VAD

HTTP

* httpx

설정 관리

* json

비동기 처리

* QThread

오디오 버퍼 처리

* numpy
* soundfile
* io.BytesIO

로깅

* logging

---

## 21. 향후 확장

* Save SRT
* Save TXT
* OpenAI Audio API 지원
* Whisper Backend 지원
* 병렬 Chunk 처리
* 화자 구분
* 맞춤법 교정
* 문장 정리
* 자막 포맷 개선
* 번역 자막 생성
* 다국어 지원
* Context Carryover Chunk 개수 설정 (현재는 1개 고정)

---

## 22. Context Carryover

### 목적

Chunk 단위로 STT를 수행할 경우 Chunk 경계에서 문맥이 끊겨 다음과 같은 문제가 발생할 수 있다.

* 고유명사/용어의 일관성 부족
* 문장이 잘린 상태로 인식 시작
* 화자의 어조/맥락 단절

직전 Chunk의 인식 결과 텍스트를 다음 Chunk 요청 프롬프트에 함께 전달하여 위 문제를 완화한다.

### 동작 방식

1. Worker는 가장 최근에 완료된 1개 Chunk의 인식 결과 텍스트(SRT 텍스트가 아닌 순수 텍스트)를 메모리에 보관한다.
2. 첫 번째 Chunk 처리 시 `{{context}}`는 빈 문자열로 치환된다.
3. Chunk N 처리 시, 프롬프트의 `{{context}}` 자리에 Chunk N-1의 인식 결과 텍스트를 치환하여 STT Provider에 전달한다.
4. Chunk N 처리가 완료되면, Chunk N의 인식 결과 텍스트로 Context를 갱신한다 (이전 Context는 폐기, 항상 1블록만 유지).
5. SRT 출력 자체에는 Context 텍스트가 포함되지 않는다 (프롬프트 입력 용도로만 사용).

### 범위 / 제약

* 보관하는 Context는 최대 1개 Chunk 분량으로 제한한다 (메모리/토큰 사용량 고려).
* 작업 취소(Cancel) 또는 새 파일 로드 시 Context는 초기화된다.
* Provider(Gemini / Local API) 공통으로 동일하게 동작한다.
* 사용자가 프롬프트 템플릿에서 `{{context}}`를 사용하지 않으면 해당 기능은 자연스럽게 비활성화된다 (별도 ON/OFF 설정 불필요).

---

## 23. STT API 요청 형식 (Chunk WAV 전달 방식)

Chunk WAV (BytesIO 메모리 버퍼)는 Provider별로 아래와 같이 **Base64 인코딩된 문자열**로 변환하여 JSON Body에 담아 전송한다. 별도의 멀티파트(form-data) 업로드는 사용하지 않는다.

### 23.1 Google Gemini API

`generateContent` API의 `inlineData` 필드를 사용한다.

* `mimeType`: `audio/wav`
* `data`: WAV bytes를 Base64 인코딩한 ASCII 문자열

요청 예시 (REST)

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
    "temperature": 0.2,
    "topP": 0.95,
    "maxOutputTokens": 4096
  }
}
```

Python 예시

```python
import base64

audio_bytes = wav_buffer.getvalue()  # BytesIO -> bytes
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

contents = [
    {"text": prompt_text},
    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
]
```

#### 제약 사항

* inline 데이터(텍스트 프롬프트 + 오디오) 합산 요청 크기는 20MB 미만이어야 한다.
* Chunk 길이를 최대 30초로 제한하는 정책(§12)에 의해 16kHz Mono PCM WAV 1개 Chunk는 약 1MB 내외이므로 제약을 충분히 만족한다.

---

### 23.2 Local OpenAI-Compatible API

OpenAI Chat Completions 스펙의 `input_audio` content type을 사용한다.

* `type`: `"input_audio"`
* `input_audio.data`: WAV bytes를 Base64 인코딩한 ASCII 문자열
* `input_audio.format`: `"wav"`

요청 예시 (REST)

```json
POST {local_api.url}
Content-Type: application/json

{
  "model": "{local_api.model}",
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
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 4096
}
```

Python 예시

```python
import base64

audio_bytes = wav_buffer.getvalue()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

payload = {
    "model": local_model_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
            ],
        }
    ],
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 4096,
}
```

#### 서버 호환성 주의 사항 (중요)

OpenAI Chat Completions의 `input_audio` content type을 모든 "OpenAI 호환" 로컬 서버가 지원하는 것은 아니다.

* **LM Studio**: 2026년 6월 기준, 공식 OpenAI 호환 Chat Completions 문서에는 `input_audio`/`audio_url` 등 오디오 관련 필드가 전혀 명시되어 있지 않다. 관련 GitHub 이슈(`lmstudio-bug-tracker#1715`, 오디오 transcription/TTS/realtime 엔드포인트 추가 요청)도 2026-03-31 개설 후 아직 Open 상태로, 공식적으로는 미지원으로 보인다.
* **Ollama**: 2026년 6월 기준, OpenAI 호환 `/v1/chat/completions`는 `input_audio`를 지원하지 않는다. 오디오 입력 자체는 Ollama 자체 API(`/api/chat`, `/api/generate`)에 `images`와 유사한 `audio`/`audios` 필드를 추가하는 형태로 논의 중이며(이슈 `ollama#11798`, PR `#15243`), 해당 PR은 아직 **병합되지 않은 Open 상태**다. 즉 Gemma 4 12B를 Ollama로 구동하더라도 현재 시점에 OpenAI 호환 레이어를 통해 표준화된 방식으로 오디오를 전달할 수 있는지는 불확실하다.
* **vLLM**: OpenAI 호환 서버에서 오디오 입력을 지원하지 않는 이슈가 보고되어 있다 (모델/버전에 따라 다를 수 있음).
* 따라서 Local API Provider 사용 시, 사용자는 `input_audio` content type을 실제로 처리할 수 있는 서버(예: Qwen-Audio 계열을 직접 서빙하는 커스텀 OpenAI 호환 서버, 또는 향후 해당 기능을 지원하는 LM Studio/Ollama/vLLM 버전)를 준비해야 한다.
* 본 프로그램은 표준 `input_audio` 포맷(OpenAI 스펙: `data`는 prefix 없는 순수 base64 문자열, `format`은 `"wav"`)으로 요청을 전송하는 것까지만 책임지며, 서버가 이를 지원하지 않을 경우 `[ERROR] API ...` 형태로 오류를 로깅하고 해당 Chunk를 실패 처리한다 (§18).
* config.json의 `local_api.url`은 사용자가 호환 서버에 맞게 자유롭게 지정 가능하도록 한다 (예: `http://localhost:8080/v1/chat/completions`).

---

### 23.3 공통 사항

* Base64 인코딩/디코딩은 표준 라이브러리(`base64` 모듈)를 사용한다.
* Base64 인코딩 시 발생하는 약 33% 크기 증가 및 인코딩/직렬화 오버헤드는 DEBUG 로그에 Chunk별 처리 시간으로 기록하여 성능 분석에 활용한다 (§18 성능 분석).
* `{{context}}` 치환은 두 Provider 모두 텍스트 파트(`text` / `parts[0].text`)에서 동일하게 수행된다 (§22).

---

### 23.4 검증 메모 (Verification Notes, 2026-06 기준)

본 설계의 §23.2(Local OpenAI-Compatible API)는 OpenAI 공식 스펙을 기준으로 작성되었으나, **실제 로컬 추론 서버(LM Studio, Ollama, vLLM 등)가 이 스펙을 지원하는지는 공식 문서로 확인되지 않았다.**

| 항목 | 상태 | 근거 |
| --- | --- | --- |
| Gemma 4 12B의 native audio input 지원 | ✅ 확인됨 | Google 공식 블로그: encoder 제거, 오디오를 텍스트 토큰과 동일 차원으로 projection |
| LM Studio `/v1/chat/completions`의 `input_audio` 지원 | ❌ 미확인 (미지원 추정) | 공식 문서에 필드 자체가 없음, 관련 이슈 Open |
| Ollama `/v1/chat/completions`(OpenAI 호환)의 `input_audio` 지원 | ❌ 미확인 (미지원 추정) | 오디오 필드 추가 PR(#15243)이 아직 미병합 |
| `data:audio/wav;base64,...` 형태의 Data URL 사용 | ❌ 근거 없음 | OpenAI 공식 스펙은 prefix 없는 순수 base64를 요구. 인터넷에서 발견된 "LM Studio + Gemma 4 12B" 예시 코드의 Data URL 포맷은 출처 불명 |
| vLLM OpenAI 호환 서버의 오디오 입력 지원 | ❌ 미지원 보고됨 | vLLM GitHub 이슈 #19977 |

#### 설계상 결론 / 권고

1. 본 프로그램은 OpenAI 공식 스펙(prefix 없는 base64 + `format: "wav"`)을 1차 구현 대상으로 한다.
2. 다만 위 표에서 보듯 어떤 로컬 서버도 이 스펙으로 오디오 입력을 받는다는 것이 공식적으로 보장되지 않으므로, **구현 단계에서 사용자가 실제로 사용할 로컬 서버(LM Studio / Ollama / 커스텀 vLLM 등)에 대해 실제 HTTP 요청을 보내 응답을 확인하는 런타임 검증이 필수**다.
3. 만약 실제 서버가 다른 포맷(예: Data URL prefix, 별도 `audio`/`audios` 필드, 별도 엔드포인트)을 요구하는 것으로 확인되면, `local_api` 설정에 `request_format` 또는 유사한 옵션을 추가하여 포맷을 선택할 수 있도록 확장한다 (현재 §8 config.json에는 미반영, 향후 확장 항목으로 §21에 추가 권장).
4. Local API Provider는 "표준 스펙을 따르는 호환 서버가 준비되어 있다"는 전제하에 동작하는 기능이며, 특정 서버(LM Studio/Ollama 등)에 대한 동작을 보증하지 않는다는 점을 README/설정 화면 안내 문구에 명시한다.
khshim