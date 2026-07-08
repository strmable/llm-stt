# Media Transcriber - 사전 준비 가이드 (SETUP)

문서 버전: 1.0 (2026-07-08)
성격: 이 문서는 **설계 결정 문서(DESIGN_3.MD)와 별개**로, 개발/실행 전에 준비해야 할 도구와 파일을 다룬다. "왜 이렇게 설계했는가"는 DESIGN_3.MD를, "무엇을 설치해야 하는가"는 이 문서를 참고한다.

---

## 0. 준비물 개요

| 구성요소 | 용도 | 필수 여부 |
|---|---|---|
| Python 3.10+ | 프로그램 본체(PySide6, httpx 등) 실행 | 필수 |
| FFmpeg | 입력 미디어 디코딩(§11 오디오 처리 파이프라인) | 필수 |
| llama.cpp (`llama-server`) | 로컬 STT 백엔드 (Gemma4 / Qwen3-ASR) | Local Provider 사용 시 필수 |
| STT 모델 GGUF (Gemma4 또는 Qwen3-ASR) | 실제 전사 수행 | Local Provider 사용 시 필수 |
| Google Gemini API 키 | 보조 Provider | Gemini Provider 사용 시에만 |

이 문서는 **1~3번(Python 개발 환경, FFmpeg, llama.cpp+모델)** 중 FFmpeg와 llama.cpp+모델을 중심으로 다룬다. Python 패키지 설치(`requirements.txt` 등)는 프로젝트 코드가 확정된 이후 별도로 정리한다.

---

## 1. FFmpeg 설치

### Windows

1. [gyan.dev FFmpeg 빌드](https://www.gyan.dev/ffmpeg/builds/) 또는 공식 [ffmpeg.org](https://ffmpeg.org/download.html)에서 `ffmpeg-release-full.7z` 다운로드
2. 압축 해제 후 `bin` 폴더를 시스템 PATH에 추가, 또는 프로그램 실행 파일과 같은 디렉토리에 `ffmpeg.exe` 배치
3. 확인:
   ```powershell
   ffmpeg -version
   ```

### macOS

```bash
brew install ffmpeg
```

### Linux (Debian/Ubuntu 계열)

```bash
sudo apt update
sudo apt install ffmpeg
```

### 동작 확인

DESIGN_3.MD §11에 명시된 처리 형식과 동일하게 변환되는지 확인:

```bash
ffmpeg -i sample_input.mp4 -f s16le -ac 1 -ar 16000 -t 5 test_output.raw
```

에러 없이 파일이 생성되면 정상.

---

## 2. llama.cpp 설치

DESIGN_3.MD §4b.3 기준, 본 프로그램은 **A유형(llama-server Chat Completions)** 백엔드를 1순위로 사용한다. Gemma4와 Qwen3-ASR 모두 이 경로로 서빙 가능하다.

### 2.1 사전빌드 바이너리 다운로드 (권장)

소스 빌드 없이 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)에서 운영체제/GPU에 맞는 사전빌드 바이너리를 받는 것을 권장한다.

**Windows + NVIDIA GPU (RTX 50시리즈/Blackwell 포함)**:

GEMMA4_1.MD §6에서 확인된 주의사항 — CUDA 12.4 / CUDA 13.3 두 가지 사전빌드가 있으며, 초기 Blackwell 드라이버 버그(`sharedMemPerBlockOptin`, llama.cpp #23385) 이슈가 있었으나 2026년 7월 기준 후속 패치가 다수 반영된 상태다.

```text
권장 절차:
1. CUDA 13.3 사전빌드를 먼저 받는다 (최신 드라이버 스택과의 DLL 충돌 회피)
2. llama-bench로 pp512/tg128 수치 확인
   - 비정상적으로 낮으면(cuBLAS 폴백 패턴) CUDA 12.4 빌드로 교체 비교
3. NVIDIA 드라이버를 최신 상태로 유지
```

**macOS (Apple Silicon)**: Metal 백엔드가 포함된 사전빌드 사용, 별도 CUDA 관련 이슈 없음.

**Linux**: 배포판/GPU에 맞는 사전빌드 선택, 또는 필요 시 소스 빌드.

### 2.2 llama-bench로 정상 동작 확인

```bash
./llama-bench -m <임의의 작은 GGUF 모델>
```

pp512(prompt processing)/tg128(token generation) 수치가 나오면 GPU 가속이 정상 동작하는 것. 비정상적으로 느리면 §2.1의 CUDA 버전 교체 절차를 따른다.

### 2.3 STT 모델 다운로드

DESIGN_3.MD §4b.2 실측 대상 모델 기준으로, 아래 조합을 우선 준비한다.

**Gemma4 12B (검증된 조합, GEMMA4_1.MD §4 기준)**

```text
모델: unsloth/gemma-4-12b-it-GGUF (Q8_0, 11.8GB) — 2026-06-05 06:36 이후 재업로드분
프로젝터: mmproj-F16.gguf (117MB, gemma4uv 통합 포맷)
```

주의: 2026-06-05 이전 다운로드분은 SIGFPE 크래시/텍스트 변환 버그가 있으므로 재다운로드 필요 (SHA256 확인 권장).

```bash
huggingface-cli download unsloth/gemma-4-12b-it-GGUF \
  --include "*Q8_0*" "mmproj-F16.gguf" \
  --local-dir ./models/gemma-4-12b
```

**Gemma4 E4B (§4b.2 비교 대상, VRAM 부담 적음)**

```bash
huggingface-cli download unsloth/gemma-4-e4b-it-GGUF \
  --local-dir ./models/gemma-4-e4b
```

(정확한 리포지토리명은 실측 시점에 Hugging Face에서 재확인)

**Qwen3-ASR (§4b.2/§4b.6 1순위 검증 후보)**

`-hf` 플래그를 쓰면 모델과 mmproj를 한 번에 자동으로 받아온다(별도 다운로드 명령 불필요):

```bash
# 1.7B (정확도 우선)
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF

# 0.6B (경량, 실측 벤치마크에서도 Whisper large-v3-turbo를 상회 — §4b.6 참고)
llama-server -hf ggml-org/Qwen3-ASR-0.6B-GGUF
```

수동으로 파일을 받아두고 싶다면:

```bash
huggingface-cli download ggml-org/Qwen3-ASR-1.7B-GGUF --local-dir ./models/qwen3-asr-1.7b
huggingface-cli download ggml-org/Qwen3-ASR-0.6B-GGUF --local-dir ./models/qwen3-asr-0.6b
```

**Whisper large-v2 (§4b.4 CJK 비교 기준선 / 서구어용)**

whisper.cpp 별도 설치가 필요 (llama-server가 아닌 B유형 백엔드, §4b.3 참고):

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build && cmake --build build -j --config Release
./build/bin/whisper-cli --model models/ggml-large-v2.bin  # 모델은 별도 다운로드 스크립트 제공
```

### 2.4 실행 확인 (수동 테스트)

DESIGN_3.MD §6.3 "Test Connection" 기능이 구현되기 전까지, 아래로 수동 확인 가능:

```bash
# Gemma4 기준 (§6.1 커맨드)
./llama-server \
  --model ./models/gemma-4-12b/gemma-4-12b-it-Q8_0.gguf \
  --mmproj ./models/gemma-4-12b/mmproj-F16.gguf \
  --n-gpu-layers 99 --ctx-size 32768 --flash-attn on \
  --host 0.0.0.0 --port 8088

# 또는 Qwen3-ASR 기준
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF --host 0.0.0.0 --port 8088
```

서버가 뜬 뒤 별도 터미널에서:

```bash
curl http://localhost:8088/health
```

`{"status":"ok"}` 유사 응답이 오면 정상. 이후 §10.1 형식의 오디오 포함 요청으로 실제 전사도 확인한다.

---

## 3. 폴더 구조 예시 (권장)

DESIGN_3.MD §2(경로 정책)와 연동되는 로컬 준비 구조 예시:

```text
MediaTranscriber/                  ← 프로그램 실행 파일 위치
├── llama-server.exe (또는 해당 OS 바이너리)
├── models/
│   ├── gemma-4-12b/
│   │   ├── gemma-4-12b-it-Q8_0.gguf
│   │   └── mmproj-F16.gguf
│   ├── gemma-4-e4b/
│   ├── qwen3-asr-1.7b/
│   └── qwen3-asr-0.6b/
├── temp/                          ← §3 Job 임시 디렉토리 (프로그램이 자동 생성)
└── config.json
```

---

## 4. 다음 단계

이 문서의 준비가 끝나면:

1. DESIGN_3.MD §4b.2의 후보 모델들을 위 방식으로 각각 띄워 한국어 샘플 오디오로 실측 비교
2. 실측 결과에 따라 §4.2 결정 게이트(보컬 분리 필요 여부) 및 §4b.5(언어별 라우팅) 판단
3. Python 개발 환경(가상환경, `requirements.txt`)은 코드 구현 착수 시점에 별도 정리
