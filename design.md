# Media Transcriber (SRT Generator) - Design v3 (Final)

문서 버전: 3.1 Final (2026-07-08)
이 문서는 v2(DESIGN_2.MD)와 v3 델타 문서를 병합한 **자기완결적 최종본**이다. 이 문서 하나만으로 전체 설계를 파악할 수 있다.
설치/환경 준비는 별도 문서(SETUP.MD)를 참고한다.

---

## 0. 설계 원칙 — 리소스 관리 4원칙

전체 설계를 관통하는 기준. 모든 세부 결정은 이 원칙과의 정합성으로 판단한다.

1. **디스크는 자유롭게 사용한다.** 오디오는 용량이 크지 않으므로 각 단계 중간 산출물을 임시 파일로 적극적으로 남긴다.
2. **메모리/VRAM은 엄격하게 관리한다.** 파일 전체를 한 번에 메모리에 올리지 않고, Chunk 단위로만 처리한다. GPU 모델은 동시에 두 개 이상 상주시키지 않는다.
3. **중단/재시작이 가능해야 한다.** 어느 시점에 프로그램이 중단되더라도, 같은 입력 파일로 다시 실행하면 처리된 부분부터 이어서 진행한다.
4. **가능하면 PyTorch/TensorFlow 같은 무거운 ML 프레임워크 의존성 자체를 피한다.** RTX 50시리즈(Blackwell, sm_120)에서 CUDA/드라이버/프레임워크 버전 호환 문제로 설치 자체가 극도로 불안정한 사례가 다수 확인됨. 불가피하게 필요한 경우 메인 앱과 격리된 별도 프로세스로 구성한다(§22.3).

---

## 1. 개요

Media Transcriber는 오디오 또는 동영상 파일을 입력받아 음성을 텍스트(STT)로 변환하고 SRT 자막 파일을 생성하는 데스크탑 GUI 애플리케이션이다.

생성된 자막은 처리 중 실시간으로 화면에 표시되며, 사용자는 진행 상태를 확인하고 언제든 작업을 중단할 수 있어야 한다. 중단된 작업은 같은 입력 파일로 재실행 시 이어서 진행할 수 있다(원칙 3).

프로그램은 대용량 미디어 파일 처리를 고려하여 Chunk 단위 스트리밍 기반으로 동작하며, 중간 산출물은 Job 단위 임시 디렉토리에 적극적으로 저장한다(원칙 1).

### 버전 배경 요약

- **v1 (2026-06 초)**: 로컬 OpenAI 호환 서버의 오디오 입력(`input_audio`) 지원이 불확실하여 구현 중단.
- **v2 (2026-07-08)**: llama.cpp `PR #24118`(2026-06-04 머지) 및 GGUF 재업로드(2026-06-05)로 llama-server + Gemma 4 12B 조합의 오디오 입력이 검증됨. 1차 타깃 Provider를 로컬 llama-server로 확정.
- **v3 (2026-07-08, 본 문서)**: 실제 운용 중 발견된 리소스 문제(GPU 발열, RAM 32GB+ 폭증)에 대응해 처리 아키텍처를 Phase 분리 + Job 기반 재시작 구조로 재설계. 언어별 STT 모델 전략(§5A) 신설 — CJK는 Qwen3-ASR을 1순위 검증 후보로 확정.

---

## 2. 목표

### 지원 기능

* 오디오 파일 자막 생성
* 동영상 파일 자막 생성
* SRT 생성 (입력 파일과 동일 폴더에 자동 저장)
* 실시간 결과 표시
* 클립보드 복사
* STT Provider 선택 (Local llama-server / Google Gemini API)
* Prompt 커스터마이징
* 언어 힌트(§5A.8) 및 사용자 용어집 주입(Custom Vocabulary, §5B.2)
* 이전 인식 결과를 컨텍스트로 전달 (Context Carryover)
* 작업 중단 및 **재시작(Resume)**
* llama-server 자동 실행/종료 (Managed 모드)
* 상세 디버깅 로그 출력

---

## 3. 실행 형태

GUI와 콘솔을 동시에 사용하는 Hybrid 구조.

* GUI: 파일 선택, 설정 관리, 진행률 표시, 결과 표시, 작업 제어
* Console: 디버깅 로그, 처리 상태, 오류, 성능 분석 출력

---

## 4. 지원 입력 형식

* Audio: wav, mp3, aac, m4a, flac, ogg, wma, opus
* Video: mp4, mkv, webm, mov, avi, mpg, mpeg, wmv, ts, m2ts, 3gp

사용자는 별도 변환 작업 없이 파일을 바로 입력할 수 있어야 한다.

---

## 5. STT Provider

### 5.1 Local llama-server (기본, 1차 타깃)

llama.cpp `llama-server`의 OpenAI 호환 `/v1/chat/completions` 엔드포인트를 사용한다.

검증된 구성 (2026-06-05 이후):

| 항목 | 값 |
|---|---|
| 모델 | Gemma 4 12B (encoder-free Unified 아키텍처, native audio input) 또는 Qwen3-ASR (§5A 참고) |
| GGUF (Gemma4) | `unsloth/gemma-4-12b-it-GGUF` Q8_0 (11.8GB) — **2026-06-05 06:36 이후 재업로드분** |
| 프로젝터 (Gemma4) | `mmproj-F16.gguf` (117MB, `gemma4uv` 통합 vision+audio 포맷) |
| GGUF (Qwen3-ASR) | `ggml-org/Qwen3-ASR-{0.6B,1.7B}-GGUF` (`-hf` 플래그로 mmproj 자동 다운로드) |
| llama.cpp | `PR #24118` 포함 빌드 (b9890대 이상 권장) |
| 오디오 제약 | 최대 30초, 16kHz mono 권장 |

주의: 2026-06-05 이전에 다운로드한 Gemma4 GGUF는 메타데이터 오류(SIGFPE 크래시, 텍스트 변환 버그)가 있으므로 SHA256을 확인하고 재다운로드해야 한다.

**LM Studio / Ollama는 지원 대상이 아니다.** LM Studio는 번들 llama.cpp가 upstream을 즉시 따라가지 않아 버전에 따라 동작이 불확실하고, Ollama는 OpenAI 호환 레이어에 오디오 필드 자체가 없다(관련 PR `#15243` 미병합). 자세한 내용은 [부록 B](#부록-b-비대상-서버-호환성-메모) 참고.

### 5.2 Google Gemini API (보조)

`generateContent` API의 `inlineData` 필드로 오디오를 전달한다.

사용 가능한 모델 예시: `gemini-3.1-flash-lite`(권장), `gemini-3-flash-preview`(시험용). 상세 모델 선택·요청 샘플은 [부록 D](#부록-d-gemini-api-이용-가이드-보조-provider) 참고.

로컬 GPU가 없거나 llama-server 준비가 어려운 환경에서 사용한다.

---

## 5A. 언어별 STT 모델 전략

### 5A.1 기본 원칙 — 언어별로 다른 모델을 쓴다

**"하나의 모델로 모든 언어를 처리한다"는 전제 자체가 최적이 아니다.** 영어/서구어와 CJK(한국어·중국어·일본어)는 사용 가능한 검증된 모델의 성숙도가 크게 다르고, 요구되는 평가 지표(WER vs CER)도 다르다.

```text
영어/서구어(라틴 문자권) 콘텐츠
   → Whisper(v2 계열)로 충분. 이미 압도적으로 검증되고 성숙한 도구이므로
     굳이 Gemma4/Qwen3-ASR 등으로 대체할 이유가 약함.
   → 잡음이 섞인 환경에서도 영어는 학습 데이터량이 절대적으로 많아
     상대적으로 안정적 (단, 이는 "노이즈에 강함"이 아니라
     "언어 자체에 대한 확신이 강해 노이즈를 밀어붙이는 것"에 가까움 — §5A.4)

CJK(한국어/중국어/일본어) 콘텐츠
   → 별도 검증 필요. §5A.2의 후보군을 대상으로 직접 실측 후 채택
   → Whisper는 v3에서 CJK 환각(hallucination)이 구조적으로 심해지는
     문제가 다수 확인되어 기준 모델로 부적합. 비교 시에도 v3가 아닌
     v2를 기준선으로 사용한다 (§5A.4)
```

### 5A.2 CJK 후보 모델 비교

| 모델 | 크기 | 한국어 지원 | 백엔드 유형(§5A.3) / llama.cpp·GGUF | 비고 |
|---|---|---|---|---|
| **Qwen3-ASR-1.7B / 0.6B** | 0.6~1.7B | 명시적 지원 (30개 언어 + 22개 중국어 방언) | **A유형(llama-server)**, `ggml-org/Qwen3-ASR-{0.6B,1.7B}-GGUF` 공식 지원, `audio_chunk_len: 30`으로 우리 Chunk 정책과 일치 | ASR 전용 특화 모델. 긴 오디오 결과 누락 버그 보고 있음(GitHub #21847), experimental 단계. **실제 한국어 콜센터 데이터(1만 건) 기준 CER 22.72%(1.7B)/26.49%(0.6B)로 Whisper large-v3-turbo(27.70%)를 능가한 제3자 실측 보고 — §5A.6** |
| **Gemma4 12B** | 12B | 확인됨(제한적, 30초 초과 시 품질 저하) | **A유형(llama-server)**, 지원(experimental) | reasoning 모델이라 `<think>` 블록 이후에 전사 결과가 옴. 스케일링 역전으로 E4B보다 오히려 부정확할 수 있다는 벤치마크 존재 |
| **Gemma4 E4B** | 4B | 확인됨(실사용 사례 있음) | **A유형(llama-server)**, 지원(experimental) | 12B보다 VRAM 부담 적고, 일부 벤치마크에서 12B보다 정확도가 높게 나옴 |
| **Qwen3-Omni-30B-A3B (Thinking/Instruct)** | 30B(활성 3B) | Qwen3-ASR의 기반 옴니모달 모델 | **A유형(llama-server)**, `ggml-org/Qwen3-Omni-30B-A3B-{Thinking,Instruct}-GGUF` 공식 GGUF, `llama-server -hf` 한 줄 실행 확인 | VRAM 부담 큼(Q4_K_M 18.6GB). ASR 전용 파인튜닝이 아니라 순수 전사 정확도가 Qwen3-ASR보다 낫다는 보장 없음. §5A.7 판단에 따라 실측 후순위 |
| **SenseVoice-Small** | 0.23B | 명시적 지원(만다린·광둥어·영어·일본어·한국어) | **C유형(CLI 전용)**, `SenseVoiceSmall-GGUF`, 별도 프로젝트 `SenseVoice.cpp` | 매우 가벼움(1GB 미만 VRAM), 비자기회귀 구조로 Whisper-Large 대비 15배 빠름. AISHELL-1 CER 2.96% vs Whisper large-v3 5.14% 보고(낭독체 기준) |
| **Fun-ASR-Nano** | Nano급 | 지원(31개 언어, 동아시아 특화) | **C유형(CLI 전용)**, `llama-funasr-cli`(2026-06-20), VAD 내장, Python 불필요 | "환각 생성과 언어 혼동 문제 해결"을 공식적으로 내세움 |
| Whisper large-v2 (CJK 기준선) | 1.5B | 지원 | **B유형(Transcriptions 서버)**, whisper.cpp로 성숙 지원 | v3보다 CJK 환각이 적어 비교 기준선으로 채택 |

### 5A.3 서빙 방식 — 백엔드 유형 구분

ASR 전용 모델과 멀티모달 LLM은 흔히 서로 다른 서버/CLI 스택을 쓴다. 후보들을 실제 확인한 결과 최소 3가지 서빙 방식이 섞여 있다.

| 백엔드 유형 | 요청 형태 | 해당 모델 | §10 API 형식과의 관계 |
|---|---|---|---|
| **A. llama-server Chat Completions** | `POST /v1/chat/completions`, JSON body 내 `input_audio`(base64) | Gemma4, Qwen3-ASR(공식 GGUF), Qwen3-Omni | **§10.1 포맷 그대로 재사용 가능.** 모델 경로만 교체 |
| **B. Whisper 계열 Transcriptions 서버** | `POST /v1/audio/transcriptions`(또는 `/inference`), 멀티파트 파일 업로드 | whisper.cpp server, faster-whisper | 별도 요청 형식. **신규 Provider 어댑터 필요.** Context Carryover도 Whisper의 `prompt` 파라미터(직전 224토큰) 방식이라 §17 로직 그대로 못 씀 |
| **C. CLI 전용 도구 (서버 없음)** | 프로세스 실행 후 stdout 파싱 | SenseVoice.cpp, `llama-funasr-cli`, `qwen3-asr.cpp`/CrispASR, chatllm.cpp | HTTP가 아님. **subprocess+파일 I/O 패턴**(§22.3과 동일)으로 통합. A유형이 실측에서 막힐 경우의 폴백 후보군 |

**Qwen3-ASR 서빙 경로 확정**: `libmtmd`를 통해 `llama-cli`/`llama-server` 양쪽에서 지원되며, OpenAI Chat Completions 포맷으로 오디오 입력을 받으므로 §10.1 형식을 그대로 재사용한다. GitHub 이슈 #21847(긴 오디오 결과 누락)은 유효한 리스크이나 30초 Chunk 설계로 조건 자체를 피할 가능성이 높다(실측 시 확인). **vLLM 경로는 RTX 50시리즈(Blackwell)에서 사전빌드 wheel/Docker 이미지가 모두 실패하는 사례(GitHub #35432)가 확인되어 우선순위에서 제외** — llama.cpp 경로가 실측에서 막힐 경우에만 재검토한다.

### 5A.4 평가 지표 원칙

- CJK(중국어·광둥어·한국어·일본어)는 **CER(문자 오류율)** 기준으로 평가한다. WER은 띄어쓰기가 불명확하거나 조사가 결합되는 CJK 특성상 부적절하다. Qwen3-ASR, NVIDIA Nemotron 등 업계 공식 평가 방법론과 일치.
- 영어/서구어는 WER 기준.
- Whisper를 CJK 비교 기준선으로 쓸 때는 **반드시 large-v2**를 사용한다. large-v3는 서구어에서 10~20% 개선됐지만, CJK에서는 환각(존재하지 않는 텍스트 생성, 문장 반복)이 v2보다 오히려 심해지는 현상이 다수 독립 보고에서 확인됨. "Whisper가 좋다"는 평가는 어떤 언어로 검증됐는지 확인 없이 일반화하지 않는다.
- 노이즈/억양 환경에서 특정 언어(예: 영어)가 잘 인식되는 것은 "노이즈 내성"이 아니라 "학습 데이터량이 많아 모델이 억지로 해당 언어로 밀어붙이는 것"에 가깝다. CJK처럼 데이터가 적은 언어는 노이즈가 조금만 섞여도 환각으로 전환될 가능성이 크다.
- CER 계산 전 양쪽(ref, hyp)에 정규화를 적용한다: 공백 전부 제거(중국어 벤치마크 데이터의 글자별 공백 포함) + 유니코드 구두점 카테고리(`P*`) 전부 제거. FLEURS 등 벤치마크의 `ref`는 구두점이 없는 반면 모델은 자연스러운 구두점을 출력하므로, 정규화 없이는 실제 전사 오류가 아닌 표기 차이로 CER이 부풀려진다(§5A.6 "자체 실측"). 숫자 표기(이백만 vs 200만) 등 발음 동일·표기만 다른 경우는 규칙 기반 정규화가 어려워 수동 확인 대상으로 남긴다. 구현: `tools/eval_language_hint.py`의 `normalize()`.

### 5A.5 Provider/모델 언어별 분기 (차기 검토)

언어에 따라 다른 모델/Provider를 자동·수동 선택하는 옵션을 차기 버전에서 검토한다(§23 향후 확장 등재). 세부 스키마는 §5A.2 실측으로 채택 모델이 확정된 후 설계한다.

### 5A.6 외부 실측 벤치마크 근거 자료

**한국어 실전 데이터(콜센터 상담 녹취 1만 건) 비교** ([mz-moonzoo.tistory.com/133](https://mz-moonzoo.tistory.com/133), 2026-02-06) — 파인튜닝 없는 Base 모델, CER 기준:

| 모델 | CER |
|---|---|
| Qwen3-ASR-1.7B | **22.72%** |
| Qwen3-ASR-0.6B | 26.49% |
| Faster-whisper-large-v3-turbo | 27.70% |

- Whisper 계열은 무음/노이즈 구간에서 "감사합니다", "Thank you" 등 무관한 단어를 반복 출력하는 환각이 CER 상승의 주 원인 — §5A.4의 CJK 환각 문제가 실제 한국어 데이터에서 재현된 사례.
- 가장 작은 0.6B조차 Whisper large-v3-turbo를 상회. 파인튜닝 시 격차 확대(Qwen3-ASR-1.7B Full FT: CER 7.41% vs Whisper LoRA FT: 11.53%). 본 프로그램은 파인튜닝 비전제이므로 Base 수치가 직접 참고치.
- 데이터 특성(단답 발화, 화자 겹침, 전화망 노이즈)이 본 프로젝트 실사용 조건과 유사.

**Qwen3-ASR 기술 리포트 리뷰** ([mz-moonzoo.tistory.com/129](https://mz-moonzoo.tistory.com/129), 2026-02-02):

- AuT 인코더(8배 다운샘플링, 12.5Hz 토큰) + Qwen3 LLM 디코더. 0.6B 기준 TTFT 92ms.
- 복잡한 음향 환경(WenetSpeech)에서 Whisper-large-v3 대비 우위 큼. 단, Fleurs 전체 30개 언어 셋(롱테일 포함)에서는 Whisper-large-v3가 근소 우위 — 주요 20개 언어에서만 Qwen3-ASR 우위. 과신 경계.
- 단일 추론 최대 20분 지원 — 30초 Chunk 정책에 제약 없음.

**Qwen3-Omni vs Qwen3.5-Omni (혼동 주의)**:

- **Qwen3-Omni-30B-A3B(전작)**: Apache 2.0 오픈웨이트, `llama-server -hf ggml-org/Qwen3-Omni-30B-A3B-{Thinking,Instruct}-GGUF`로 즉시 실행 가능 확인.
- **Qwen3.5-Omni(2026-03-30, 후속작)**: Plus/Flash/Light variant. 출시 시점 기준 오픈웨이트 여부 미확인 + vLLM 권장 → 로컬 실행 후보에서 제외, 라이선스/GGUF 공개 여부를 주기적으로 재확인.
- 이름이 유사하므로 실측 시 리포지토리명으로 버전(3 vs 3.5)을 반드시 확인한다.

**자체 실측 (2026-07-09, Qwen3-ASR-1.7B, FLEURS validation 9샘플, temperature=0, `tools/eval_language_hint.py`)**:

CER 정규화(§5A.4, 구두점·공백 제거) 적용 전/후 언어별 평균 auto CER:

| 언어 | raw CER | 정규화 CER | 잔존 오류 |
|---|---|---|---|
| 일본어 | 6.2% | **1.6%** | 고유명사 이표기 1건 |
| 중국어 | 8.8% | **1.9%** | 고유명사 오류 1건 |
| 한국어 | 13.1% | **9.9%** | 음운 유사 오인식 3건 전체 |

- ja/zh는 정규화 후 오류가 거의 사라짐(6개 중 4개 0%) — raw CER의 대부분이 벤치마크 `ref`에 구두점이 없는 데서 온 착시였다. 낭독체 조건에서는 실질적으로 매우 우수.
- ko만 정규화 후에도 9.9%로 유의미하게 높게 남는다 — "몰아냈음→보란했음", "무척→부쩍", "족히→조기" 등 음운적으로 유사한 단어로의 실제 오인식 3건이 원인이며, 이는 표기 차이가 아닌 진짜 STT 오류. 한국어가 ja/zh보다 약하다는 신호.
- `language: {code}` 힌트(§5A.8)는 9건 중 1건(ko_001, 음운 오인식)을 정정하며 나머지엔 영향 없음 — 손해 사례 없이 가끔 이득이라는 §5A.8 판단을 재확인.
- 상세 표와 데이터 출처는 [TESTING.md](TESTING.md) §4.1 참고. 노이즈 섞인 실사용 오디오로는 아직 미검증(§5A.7 다음 단계).

### 5A.7 실측 우선순위

```text
1순위: Qwen3-ASR-1.7B / 0.6B
   - ASR 전용 모델이라 옴니모달(Qwen3-Omni-30B-A3B) 대비 크기·속도에서 유리
   - §5A.6 제3자 실측(한국어)에서 근거가 가장 탄탄함
   - A유형(llama-server)으로 아키텍처 재사용 부담 최소

2순위(보류): Gemma4 12B/E4B, Qwen3-Omni-30B-A3B, SenseVoice-Small, Fun-ASR-Nano
   - Qwen3-ASR 실측 결과가 기준치 미달일 경우에만 추가 비교
```

### 5A.8 언어 강제 지정 (프롬프트 힌트, 실측 확인)

§5A.5의 언어별 모델/Provider **라우팅**과는 별개로, 단일 모델 안에서 언어를 강제하는 훨씬 가벼운 해법이 실측으로 확인됐다 (2026-07-08, `rec1.m4a` 샘플, Qwen3-ASR-1.7B).

- Qwen3-ASR 응답은 `language {Lang}<asr_text>{내용}` 형태의 고정 구조로 나온다. 텍스트 프롬프트에 아무 언어 힌트가 없으면(`"Transcribe this audio."`) 짧고 외래어 위주인 한국어 발화("마이크 테스트")를 일본어로 오판해 가타카나로 옮기는 사례가 재현됐다 (§5A.4의 CJK 언어 혼동이 실제로 나타난 경우).
- 프롬프트에 `language: ko` 한 줄만 추가해도 응답이 `language Korean<asr_text>...`로 정상화된다. 문장 형태(영어/한국어 지시문)든 짧은 코드(`language: ko`)든 동일하게 작동 — 즉 **경량 프롬프트 주입만으로 언어 오판을 억제 가능**.
- 이 발견에 따라 config.json에 `language` 설정(기본 `auto`)을 추가하고, `auto`가 아니면 프롬프트에 `language: {code}` 힌트를 자동 주입하는 구조로 설계 반영 (§8, §9 참고). Qwen3-ASR 외 모델(Gemma4 등)에서도 동일하게 작동하는지는 추가 실측 필요.

---

## 5B. 텍스트 정확도 강화 — 용어집 주입 및 후처리

### 5B.1 배경 및 역할 구분

§5A.6 자체 실측과 §5A.8에서 경량 프롬프트 힌트(언어 지정)만으로 오인식이 해소되는 사례가 확인되었다. 이를 일반화하면 STT 파이프라인에 두 종류의 개입 지점이 있다.

```text
Audio
   ↓
[Pre-biasing] 언어 힌트({{language_hint}}, §5A.8) + Custom Vocabulary({{vocabulary}}, §5B.2)
   │            — 인식 전에 모델에게 힌트 제공
   ↓
STT (Qwen3-ASR 등)
   ↓
Transcript
   ↓
[Post-processing] Text Correction (§5B.3) — 인식 후 문맥 기반으로 재교정
   ↓
최종 결과 (SRT)
```

역할이 명확히 구분된다:

| 구분 | 시점 | 방식 | 예시 |
|---|---|---|---|
| **Pre-biasing (힌트 주입)** | 인식 *전* | 프롬프트에 힌트 주입, STT 모델이 후보 선택 시 우선 고려 | `language: ko` (§5A.8), "다음 용어가 등장할 수 있습니다: ヴェルサーチ, Machbase, OpenTelemetry" |
| **Post-processing (텍스트 후처리)** | 인식 *후* | 별도 LLM 호출로 이미 나온 전사 결과를 문맥 기반 재교정 | "ベルサージ" → 문맥(패션 브랜드) 보고 "ヴェルサーチ"로 정정 |

**Pre-biasing 장단점**: 모델이 처음부터 맞게 인식할 가능성이 높아지고 결과가 가장 자연스럽다. 다만 용어집에 없는 고유명사는 여전히 틀릴 수 있고, 너무 많은 용어를 넣으면 효과가 떨어질 수 있다(§23 튜닝 항목).

**Post-processing 장단점**: STT는 오디오 조각(Chunk) 하나만 보고 판단하지만, 후처리 LLM은 문장 전체(또는 여러 문장)를 보고 의미적으로 판단할 수 있다. 예를 들어 "克隆技术的发展"(복제 기술)과 "德国科隆大教堂"(쾰른 대성당)은 발음이 같은 "科隆/克隆"을 문맥으로 구분해야 하는데, 오디오 단독으로는 어렵고 텍스트 문맥이 있어야 가능하다 — §5A.6 자체 실측의 잔존 오류(고유명사 이표기, 음운 유사 오인식)가 정확히 이 유형이다. 반대로 후처리는 STT가 아예 잘못 들은 음성 자체는 교정하지 못한다(텍스트 교정 문제이지 재인식이 아님).

**결론**: 둘은 서로 다른 오류 유형을 잡아내므로 함께 쓰는 것이 이상적이다. 앞단(Pre-biasing)에서 오류를 줄이고, 뒷단(Post-processing)에서 남은 오류를 문맥으로 수정한다.

### 5B.2 Pre-biasing — Custom Vocabulary

§5A.8의 언어 힌트(`{{language_hint}}`)와 별개로, 설정 창(§8)에 사용자 용어집 입력란을 추가한다.

```text
Custom Vocabulary (선택, 줄바꿈으로 구분)
[ Machbase                                          ]
[ OpenTelemetry                                      ]
[ 홍길동                                              ]
```

프롬프트 템플릿(§8)에 신규 변수 추가:

| 변수 | 설명 |
|---|---|
| `{{vocabulary}}` | 용어집이 비어있지 않으면 `The following terms may appear in the audio:\n- 항목1\n- 항목2\n\n` 블록으로 치환, 비어있으면 빈 문자열 (빈 블록이 오히려 노이즈가 될 수 있으므로 블록째 생략) |

기본 프롬프트 템플릿 갱신:

```text
{{language_hint}}{{vocabulary}}Transcribe the following audio chunk to text.

Previous context:
{{context}}

Output only the transcription of the current audio chunk.
```

### 5B.3 Post-processing — Text Correction

**아키텍처 상 위치**: Phase B(STT) 완료 후 별도 Phase C로 추가한다. Chunk 단위가 아니라 **문서 전체(또는 인접 여러 Chunk)를 한 번에 보고 교정**하는 것이 이 기능의 핵심 가치이므로, 실시간 Chunk 처리 루프 안에 넣지 않는다.

```text
┌─────────────────────────────────────────────────────────┐
│ Phase C: 텍스트 후처리 (선택, 기본 OFF)                     │
├─────────────────────────────────────────────────────────┤
│ Phase B 완료 후 {입력파일명}.srt 전체(또는 설정된 윈도우 크기  │
│ 단위)를 텍스트 LLM에 전달                                    │
│   → 문맥 기반 재교정 요청                                    │
│   → 교정된 텍스트로 SRT 갱신 (원본은 {입력파일명}.raw.srt로    │
│     별도 보존 — 교정이 항상 개선을 보장하지 않으므로 원본      │
│     비교/롤백 가능해야 함)                                   │
│   → 콘솔 로그에 변경분(diff) 기록                             │
└─────────────────────────────────────────────────────────┘
```

**모델 선택**: Qwen3-ASR은 ASR 전용 모델이라 순수 텍스트 교정에 적합하지 않다. 후처리는 별도 텍스트 LLM(예: llama-server에 Qwen3 Instruct 계열을 띄우거나, Gemini API 텍스트 호출)을 쓴다. 원칙 2(VRAM 엄격 관리)에 따라 STT 모델과 텍스트 교정 모델을 동시에 상주시키지 않는다 — Phase B 종료(Managed 모드에서 STT 서버 종료, §6.3) 후 Phase C용 모델을 별도 구동하는 순서를 지킨다.

**긴 문서 처리**: 문서 전체를 한 번에 넣으면 컨텍스트 길이 제약에 걸릴 수 있으므로, 일정 크기(예: N개 Chunk)씩 슬라이딩 윈도우로 나눠 처리한다. 세부 청킹 전략은 구현 단계에서 결정.

**프롬프트 설계 방향**:

```text
다음은 음성 인식 결과입니다. 문맥을 보고 발음이 비슷해 혼동되었을
가능성이 있는 단어(고유명사, 전문용어, 동음이의어)를 자연스럽게 교정하세요.
원문의 의미나 문장 구조를 임의로 바꾸지 말고, 명백한 오인식만 수정하세요.

{{vocabulary}}  ← Pre-biasing과 동일한 용어집을 재사용해 후처리 정확도도 높임

[전사 결과]
{{transcript_window}}
```

### 5B.4 설정 창 옵션 (§8 확장)

```text
Custom Vocabulary (선택)
[                                                    ]

[ ] Text Correction (STT 결과를 문맥 기반으로 재교정, 기본 OFF)
    사용 시 추가 설정:
    Provider    : (o) 로컬(Qwen3 Instruct 등)  ( ) Gemini API
    윈도우 크기 : Chunk 단위 [   ] 개
```

세 기능은 책임이 분리되어 독립적으로 켜고 끌 수 있다: **출력 언어 설정(§8, §5A.8)**은 언어를 미리 알려주는 힌트, **Custom Vocabulary**는 등장 가능한 고유명사/전문용어를 미리 알려주는 힌트(이상 Pre-biasing), **Text Correction**은 STT 결과를 문맥에 맞게 다듬는 사후 단계(Post-processing).

기본값은 Custom Vocabulary=비어있음, Text Correction=OFF로 기본 파이프라인(§11)에 영향을 주지 않는다. Text Correction은 추가 리소스(별도 모델 구동, 처리 시간 증가)를 요구하므로 명시적 옵트인.

### 5B.5 config.json 반영

기존 최상위 `language` 필드(§9, §5A.8)는 그대로 두고, 아래 블록을 추가한다.

```json
"text_enhancement": {
  "custom_vocabulary": [],
  "text_correction": {
    "enabled": false,
    "provider": "local_api",
    "window_chunks": 5
  }
}
```

### 5B.6 버전 범위

- **Pre-biasing(언어 힌트 + Custom Vocabulary)**: 언어 힌트는 §5A.8 실측으로 효과 확인, 용어집은 구현 부담이 낮으므로(프롬프트 텍스트 조합) **이번 버전에 포함**한다.
- **Post-processing(Text Correction)**: 아키텍처 변경(Phase C, 별도 텍스트 모델)이 필요해 구현 부담이 크다. **Pre-biasing만으로 충분한지 먼저 확인한 뒤 도입 여부를 판단** — §23 향후 확장 등재.
- 검증 아이디어: §5A.6 자체 실측의 잔존 오류(ja_001 ヴェルサーチ, zh_001 科隆)가 Custom Vocabulary로 잡히는지 확인하면 §5B.2의 실효성을 바로 검증할 수 있다.

---

## 6. 로컬 서버 실행 환경 (llama-server)

### 6.1 실행 커맨드 (External 모드 기준)

```bash
# Gemma4 12B
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

# Qwen3-ASR (mmproj 자동 다운로드)
# 주의: 플래그 없이 그냥 -hf만 쓰면 llama-server가 모델의 최대 컨텍스트(65536)와
# 기본 병렬 슬롯(4)으로 KV 캐시를 잡아 VRAM을 10GB 이상 소모한다(실측 확인).
# 30초 Chunk 정책(§12)에는 큰 컨텍스트가 불필요하므로 아래처럼 명시적으로 줄인다.
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF \
  --ctx-size 4096 \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8088
```

* VRAM: Gemma4 Q8_0 기준 약 14GB / Qwen3-ASR은 위 튜닝 커맨드 기준 약 4GB (기본값 그대로 실행 시 10GB+, 2026-07-08 실측)
* 성능 참고: HF Transformers 대비 5~7배 빠른 디코딩 (Gemma4, 64~131 t/s)

### 6.2 Windows + RTX 50시리즈(Blackwell) 주의사항

* llama.cpp 공식 릴리스는 CUDA 12.4 / CUDA 13.3 두 가지 사전빌드를 제공.
* 권장 절차: ① CUDA 13.3 빌드를 먼저 사용 (최신 드라이버와의 DLL 충돌 회피) → ② `llama-bench`로 pp512/tg128 확인, 비정상적으로 낮으면(cuBLAS 폴백 패턴) CUDA 12.4 빌드와 비교 → ③ NVIDIA 드라이버 최신 유지 (초기 Blackwell 드라이버의 `sharedMemPerBlockOptin` 버그가 원인이었음, llama.cpp `#23385`).

### 6.3 Launch Mode — External / Managed

```text
Launch Mode
(o) External (기본 — 사용자가 사전에 llama-server를 직접 실행)
( ) Managed  (프로그램이 Phase B 직전 자동으로 llama-server를 실행하고, 완료 후 종료)
```

**Managed 모드 배경**: External 방식은 (1) 매번 별도 터미널에서 커맨드 실행이 번거롭고, (2) llama-server가 대기 상태에서도 모델을 VRAM에 상시 적재하는 문제가 있다. Managed 모드는 Phase B 직전에만 서버를 띄워 원칙 2(VRAM 엄격 관리)를 강화한다.

**실행 시점 및 생명주기**:

```text
Phase A 완료 (VAD/Chunking 종료)
   │
   ▼ [Managed 모드]
   1. 지정 포트가 이미 응답 중인지 확인
        - 응답 있음(사용자가 별도로 띄워둔 경우 등) → 재사용, 신규 실행하지 않음
        - 응답 없음 → subprocess로 llama-server 실행
   2. /health 엔드포인트를 폴링하며 모델 로딩 완료 대기 (기본 타임아웃 120초)
        - 로딩 중 UI에 "STT 서버 준비 중..." 표시
        - 타임아웃 시 §21 오류 처리 정책에 따라 오류 표시 후 작업 중단
   3. 준비 완료 → Phase B 시작
   │
   ▼
Phase B 완료 (모든 chunk transcribed)
   │
   ▼ [Managed 모드]
   프로그램이 직접 실행한 프로세스 → 종료(SIGTERM, 무응답 시 강제 종료)
   기존에 떠 있던 것을 재사용한 경우 → 종료시키지 않음
```

작업 중단(Stop) 시에도 프로그램이 직접 실행한 프로세스는 함께 정리한다. PID를 추적하여 이전 세션의 좀비 프로세스 확인 로직은 후속 구현에서 검토.

**트레이드오프**: Managed 모드의 모델 로딩은 **Job당 1회**(Phase A→B 전환 시점)만 발생한다. 긴 파일(수십 분~수 시간)에서는 전체 처리 시간 대비 오버헤드 비중이 미미하므로 Managed를 기본처럼 써도 무방하다. 짧은 파일을 여러 번 연속 처리하는 패턴에서는 External(사전 기동 후 여러 Job 재사용)이 유리하다.

### 6.4 연결 확인 (Test Connection)

설정 창에 **[Test Connection]** 버튼을 제공하며, **Provider에 따라 분기 동작**한다.

```text
provider == "local_api" →
  1. 1초 분량의 무음 WAV(16kHz mono)를 메모리에서 생성
  2. §10.1과 동일한 포맷으로 서버에 전송
  3. 성공: HTTP 200 + choices[0].message.content 존재 → "OK"
  4. 실패: 상태 코드/오류 메시지 표시 (input_audio 미지원, 연결 거부, 타임아웃 등)
  * Managed 모드에서는 경로(server_binary/model_path/mmproj_path) 존재 여부도 함께 검증

provider == "gemini" →
  동일한 무음 WAV를 inline_data로 전송 → candidates[0] 존재 확인
  * 키가 여러 개 등록된 경우(차기 버전) 기본은 첫 번째 키만 테스트
```

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

* 이전 작업(임시 파일)이 감지된 입력 파일 선택 시 "이어하기/새로 시작" 다이얼로그 표시 (§14.3)

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

Launch Mode
(o) External   ( ) Managed

llama-server 실행 파일 경로 : (예: C:\llama.cpp\llama-server.exe)
모델 파일 경로              : (예: C:\models\gemma-4-12b-it-Q8_0.gguf)
mmproj 파일 경로            : (예: C:\models\mmproj-F16.gguf)

[Managed 모드 선택 시에만 활성화]
포트                        : 8088
추가 인자 (선택)            : (예: --ctx-size 32768 --flash-attn on)

[Test Connection]
```

경로 3종(실행 파일/모델/mmproj)은 Launch Mode와 무관하게 항상 표시·저장한다(모드 전환 시 재입력 방지). 포트/추가 인자는 Managed 선택 시에만 활성화.

안내 문구(설정 화면 하단 고정 표시):

> 검증 환경: llama.cpp llama-server(2026-06-05 이후 빌드) + Gemma 4 12B GGUF + gemma4uv mmproj, 또는 ggml-org/Qwen3-ASR GGUF.
> LM Studio / Ollama 등 기타 서버는 동작을 보증하지 않습니다.
> llama.cpp의 오디오 입력은 "experimental"로 표기되어 있어 인식 품질이 저하될 수 있습니다.
> Managed 모드는 Job당 1회 모델 로딩 시간이 추가되나 일반적으로 비중이 낮으며, 짧은 파일을 반복 처리하는 경우 External 모드를 권장합니다.

### Google Gemini API 설정

```text
API Key   : AIza...
Model Name: gemini-3.1-flash-lite
[Test Connection]
```

API 키는 config.json에 평문으로 저장한다. 다중 키 로테이션은 차기 버전 검토 항목(§23).

### 출력 언어 설정

```text
출력 언어
(o) Auto (모델이 자동 감지)
( ) 강제 지정 : [ko ▾]   (한국어/日本語/中文/English/기타 코드 직접입력)
```

* 기본값은 `Auto`. Provider(local_api/gemini) 및 모델과 무관하게 공통 적용.
* `Auto`가 아닌 값을 선택하면 프롬프트의 `{{language_hint}}`가 `language: {code}\n\n` 형태로 채워진다. `Auto`면 `{{language_hint}}`는 빈 문자열.
* 실측 근거(§5A.8): 짧은 발화·외래어가 섞인 한국어를 모델이 일본어로 오판하는 사례가 확인됐고, `language: ko` 한 줄 힌트만으로 정상화됨을 Qwen3-ASR-1.7B로 검증 (2026-07-08). Gemma4 등 다른 모델에서의 동작은 추가 검증 필요.

### Custom Vocabulary / Text Correction (§5B)

```text
Custom Vocabulary (선택, 줄바꿈으로 구분 — 등장 가능한 고유명사/전문용어)
[                                                    ]

[ ] Text Correction (STT 결과를 문맥 기반으로 재교정, 기본 OFF — 차기 버전)
```

* Custom Vocabulary가 비어있지 않으면 프롬프트의 `{{vocabulary}}`가 용어 목록 블록으로 치환된다. 비어있으면 빈 문자열 (§5B.2).
* Text Correction은 차기 버전 검토 항목(§5B.6)이므로 이번 버전 UI에는 비활성 상태로만 표시하거나 노출하지 않는다.

### Prompt 설정

멀티라인 텍스트 박스 + `[Load] [Save] [Save As]` 버튼.

#### Template Variables

| 변수 | 설명 |
|---|---|
| `{{context}}` | 직전에 처리된 1개 Chunk의 인식 결과 텍스트 (없으면 빈 문자열) |
| `{{language_hint}}` | 출력 언어 설정이 `Auto`가 아닐 때 `language: {code}\n\n`, `Auto`일 때는 빈 문자열 |
| `{{vocabulary}}` | Custom Vocabulary(§5B.2)가 비어있지 않으면 용어 목록 블록, 비어있으면 빈 문자열 |

기본 프롬프트:

```text
{{language_hint}}{{vocabulary}}Transcribe the following audio chunk to text.
Use the previous context only to keep terminology, names and
sentence flow consistent. Do not repeat the previous context
in your output.

Previous context:
{{context}}

Output only the transcription of the current audio chunk.
```

* 사용자가 `{{context}}`를 제거하면 컨텍스트는 전달되지 않는다 (§17 참고).
* 사용자가 `{{language_hint}}`를 제거하면 출력 언어 설정과 무관하게 힌트가 전달되지 않는다 (Auto와 동일한 동작).

### 모델 파라미터

```text
Temperature   Top-P   Top-K   Max Tokens
```

* Gemma 4 권장 샘플링 값이 top-k 64를 포함하므로 Top-K 노출.
* Gemini API 사용 시 Top-K는 `generationConfig.topK`로 전달.

### 임시 파일 정리 옵션

```text
[✓] 완료 후 임시 파일 정리      (기본 ON)
```

정상 완료 시에만 `temp/{job_id}/`를 삭제한다. 중단/크래시 시에는 재시작(Resume)을 위해 항상 보존(§14.4). v2의 "Save Chunk WAV Files" Debug 옵션은 제거 — Chunk WAV가 항상 저장되는 구조로 바뀌어 의미가 없어짐.

### 버튼

`[OK]` — config.json 저장 후 닫기 / `[Cancel]` — 변경사항 폐기

---

## 9. 설정 파일

config.json

```json
{
  "provider": "local_api",
  "language": "auto",

  "local_api": {
    "url": "http://localhost:8088/v1/chat/completions",
    "model": "gemma-4-12b",
    "disable_thinking": true,
    "launch_mode": "external",
    "server_binary": "",
    "model_path": "",
    "mmproj_path": "",
    "managed": {
      "port": 8088,
      "extra_args": "",
      "startup_timeout_sec": 120
    }
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
    "template": "{{language_hint}}{{vocabulary}}Transcribe the following audio chunk to text.\n\nPrevious context:\n{{context}}\n\nOutput only the transcription of the current audio chunk."
  },

  "text_enhancement": {
    "custom_vocabulary": [],
    "text_correction": {
      "enabled": false,
      "provider": "local_api",
      "window_chunks": 5
    }
  },

  "cleanup": {
    "remove_temp_on_success": true
  },

  "logging": {
    "level": "INFO"
  }
}
```

주요 사항:

* `provider` 기본값: `local_api`
* `language` 기본값: `auto` (모델 자동 감지). `auto`가 아니면 프롬프트의 `{{language_hint}}`가 `language: {code}\n\n`로 치환되어 전송된다 (§5A.8 실측 근거, §8 "출력 언어 설정")
* `local_api.disable_thinking`: true 시 요청에 `chat_template_kwargs: {"enable_thinking": false}` 포함 (Gemma 4의 thinking 출력 억제, STT 용도에서는 항상 억제가 바람직)
* `local_api.launch_mode`: `"external"`(기본) / `"managed"`
* `server_binary`/`model_path`/`mmproj_path`는 `launch_mode`와 무관하게 최상위에 상시 저장 (모드 전환 시 재입력 방지). subprocess 실행에 실제 사용되는 것은 Managed 모드일 때뿐
* `cleanup.remove_temp_on_success`: 정상 완료 시 temp/{job_id}/ 삭제 여부 (기본 true)
* v2의 `debug.save_chunk_wav`는 제거 (Chunk WAV 상시 저장 구조로 대체)

---

## 10. STT API 요청 형식

Chunk WAV는 **Base64 인코딩 문자열**로 JSON Body에 담아 전송한다. 멀티파트 업로드는 사용하지 않는다. Base64의 `data`는 **prefix 없는 순수 base64 문자열**이다 (`data:audio/wav;base64,...` Data URL 형식 금지 — OpenAI 스펙 및 llama.cpp 모두 순수 base64 사용).

### 10.1 Local llama-server (OpenAI 호환)

`input_audio` content type 사용. **llama-server + Gemma 4 12B 조합에서 동작 검증 완료 (2026-06-05 이후). Qwen3-ASR 공식 GGUF도 동일 형식 사용(§5A.3 A유형).**

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

제약: inline 데이터 합산 요청 크기 20MB 미만. 30초 Chunk(16kHz mono WAV ≈ 1MB)는 충분히 만족. 상세는 [부록 D](#부록-d-gemini-api-이용-가이드-보조-provider).

### 10.3 공통 사항

* Base64 인코딩은 표준 라이브러리(`base64`) 사용.
* Base64 인코딩으로 인한 약 33% 크기 증가 및 직렬화 오버헤드는 DEBUG 로그에 Chunk별 처리 시간으로 기록.
* `{{context}}` 치환은 두 Provider 모두 텍스트 파트에서 동일하게 수행.
* 서버가 요청을 처리하지 못하는 경우 `[ERROR] API ...` 형태로 로깅하고 해당 Chunk를 실패 처리.

---

## 11. 처리 아키텍처 — Phase A/B 분리 파이프라인

```text
┌─────────────────────────────────────────────────────────┐
│ Phase A: 전처리 (LLM 서버 미구동)                          │
├─────────────────────────────────────────────────────────┤
│ Input Media File                                          │
│   → FFmpeg Decode → 16kHz Mono PCM Stream                  │
│   → VAD (음성 구간 검출) + 후처리(§12.2)                     │
│   → Chunk 분할 (최대 30초)                                  │
│   → temp/{job_id}/chunks/chunk_NNNN.wav 저장                │
│   → manifest.json 갱신 (각 chunk status)                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase B: STT (llama-server 구동, VRAM은 STT 모델 전용)       │
├─────────────────────────────────────────────────────────┤
│ [Managed 모드] llama-server 자동 실행 + /health 대기 (§6.3)  │
│ manifest.json에서 미완료 chunk 목록 로드                     │
│   → Chunk 순회:                                            │
│       Prompt Build (Context 치환)                          │
│       → Base64 인코딩 → STT API 호출 (llama-server/Gemini)   │
│       → 결과를 {입력파일명}.srt 에 즉시 append                │
│       → 결과를 chunk_NNNN.txt 로도 저장 (재시작용)            │
│       → manifest.json의 해당 chunk status = "transcribed"    │
│       → Context 갱신 (다음 chunk용, 메모리에는 1개 블록만 유지) │
│       → UI 갱신 (TextBox, ProgressBar)                      │
│ [Managed 모드] 완료 시 llama-server 종료                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                  모든 chunk 완료
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   [정리 옵션 ON, 기본값]      [정리 옵션 OFF]
   temp/{job_id}/ 삭제         temp/{job_id}/ 보존
```

> 차기 버전 검토 대상(§22 참고): STT 성능이 배경음 있는 파일에서 저하된다고 판단될 경우, Phase A에 Vocal Extraction 단계(Chunk 단위, External Process)를 삽입하는 방안을 검토한다. 이번 버전에는 포함하지 않는다.

**Provider가 Google Gemini API인 경우** Phase B에서 llama-server 구동 단계는 생략되고 바로 API 호출로 진행한다.

### FFmpeg 처리 정책

모든 입력 파일은 FFmpeg로 처리하며, 동영상은 오디오 트랙을 자동 추출한다. 사용자가 사전 변환할 필요 없다.

출력 형식 통일: PCM / 16kHz / Mono

```bash
ffmpeg -i input_file -f s16le -ac 1 -ar 16000 -
```

이 형식은 Gemma 4 / Qwen3-ASR의 권장 입력(16kHz mono)과 일치한다.

---

## 12. VAD 및 Chunk 생성 정책

### 12.1 VAD — Silero VAD (onnxruntime 직접 로드)

**채택**: Silero VAD (최신 버전 유지).

**알려진 한계**: ROC 벤치마크 기준 5% FPR에서 TPR 87.7%(음성 프레임 8개 중 1개 놓침), 1% FPR에서 TPR 80.4%. threshold를 낮추면 놓침은 줄지만 오탐이 늘고, 극단적으로 낮추면 전체 파일이 하나의 세그먼트로 뭉치는 부작용도 보고됨. threshold는 실측 단계 튜닝 대상.

**대안 검토 결과**: TEN VAD는 저지연 종료감지가 핵심 강점이나, 본 프로그램은 실시간 turn-taking이 아닌 오프라인 배치 SRT 생성이므로 효용이 낮음 → 우선순위 하향, Silero 최신 버전 유지를 우선(버전에 따라 정확도가 크게 달라짐 — v4 대비 이후 버전에서 개선 보고됨). Pyannote는 GPU 필요성이 높아 원칙 2와 상충. WebRTC VAD는 정확도 최하. 실제 교체 여부는 구현 단계에서 우리 데이터로 직접 비교 후 결정(§23 등재).

**의존성 최소화 (원칙 4)**: Silero VAD 공식 패키지는 오디오 I/O에 `torchaudio`를 사용해 PyTorch 생태계 전체가 딸려 들어올 수 있다. 본 프로그램은 FFmpeg로 16kHz mono PCM을 직접 추출하므로, Silero VAD의 `.onnx` 모델을 `onnxruntime.InferenceSession`으로 직접 로드하고 PCM 배열을 그대로 입력한다. 이렇게 하면 `torch`/`torchaudio` 없이 `onnxruntime`(CPU) 단일 의존성으로 VAD를 구동한다.

### 12.2 VAD 결과 후처리

VAD 원시 출력을 그대로 Chunk 경계로 쓰지 않는다. 실제로 VAD를 사용하면 0.2~1초의 아주 짧은 침묵도 다수 검출되는데("안녕하세요." (0.4초) "저는..." (0.5초) "오늘..."), 이걸 전부 분할 기준으로 삼으면 Chunk가 수백 개로 늘어나 비효율적이다(Chunk당 API 호출 오버헤드, Context Carryover 단절 빈도 증가).

**짧은 침묵/구간 병합**:

```text
- 0.5~1초 이하의 침묵은 무시하고 앞뒤 음성 구간을 이어 붙인다.
- 2~3초 이상의 침묵만 실제 분할 기준으로 사용한다.
- 너무 짧은 음성 구간(예: 1초 미만)은 앞뒤 구간과 병합한다.
```

임계값은 실측 단계 튜닝 대상으로 두고 위 범위를 초기값으로 삼는다.

**긴 음성 구간의 강제 분할**:

VAD가 검출한 하나의 구간이 30초를 초과할 수 있다(예: 쉼 없이 95초 발화). 이 경우 VAD만으로는 하드 리밋을 지킬 수 없으므로 강제 분할한다.

```text
예: 0~95초 무휴지 구간 → 0~30 / 30~60 / 60~90 / 90~95 로 강제 분할

이상적으로는 구간 내 에너지가 낮은 지점(숨쉬는 지점 등)을 절단선으로 선택하는 것이
좋으나, 이번 버전에서는 30초 지점에서 기계적으로 자르고 필요 시 개선한다.
```

### 12.3 Chunk 경계 결정 로직 (최종)

```text
1. VAD로 원시 음성/무음 구간 검출
2. 짧은 침묵(0.5~1초 이하) 병합 → 인접 음성 구간 연결
3. 짧은 음성 구간(1초 미만) 병합 → 인접 구간에 흡수
4. 병합 결과가 30초를 초과하면 강제 분할 (하드 리밋 — Gemma4 오디오 입력 상한과 일치)
5. 최종 Chunk 목록 확정 → manifest.json에 기록
```

30초는 모델 제약에 따른 **하드 리밋**이며 초과 금지.

---

## 13. 경로 정책

| 대상 | 위치 | 비고 |
|---|---|---|
| 입력 파일 | 사용자 지정 | — |
| 출력 SRT | **입력 파일과 동일 폴더**, `{입력파일명}.srt` | 고정값. 사용자 지정 옵션 없음. 결과 파일을 다시 찾아 헤매는 것을 방지하기 위한 의도적 설계 |
| 임시 파일 | **프로그램 실행 파일 디렉토리 기준**, `./temp/{job_id}/` | 입력 파일이 읽기 전용/네트워크 드라이브에 있어도 항상 쓰기 가능해야 하므로 입력 파일 위치와 분리 |

> 참고(구현 시 확인 필요): Windows에서 `Program Files` 하위 설치 시 일반 사용자 권한으로 쓰기가 제한될 수 있다. 이 경우 `%LOCALAPPDATA%\MediaTranscriber\temp\` 폴백이 필요할 수 있으나, 이번 버전에서는 실행 파일 디렉토리 기준으로 고정하고 문제 확인 시 후속 버전에서 추가한다(§23).

동명 SRT 파일이 이미 존재하는 경우의 처리(덮어쓰기 확인 vs 자동 `_1` 접미사)는 세부 구현 단계에서 결정한다(§24).

임시 디렉토리 구조:

```text
temp/{job_id}/
  manifest.json
  chunks/
    chunk_0001.wav        ← VAD 추출 Chunk
    chunk_0001.txt         ← STT 결과 텍스트 (재시작용)
    chunk_0002.wav
    ...
```

출력 SRT는 임시 파일이 아니라 Chunk 완료마다 최종 위치에 직접 append되는 결과물이므로, 정리 정책과 무관하게 항상 유지된다.

---

## 14. Job 기반 재시작 (Resume)

### 14.1 job_id 산정 기준

```text
job_id = f(입력 파일 절대경로, 수정시각(mtime), 파일 크기)
```

경로만으로 판단하면 동일 경로에 다른 파일이 놓였을 때 잘못된 이어하기가 발생할 수 있다. 파일 전체 해시는 큰 미디어 파일에서 비용이 크므로, **경로 + mtime + 크기** 조합을 절충안으로 채택한다.

### 14.2 manifest.json 스키마

```json
{
  "job_id": "a1b2c3d4",
  "source_file": "D:\\videos\\lecture.mp4",
  "source_mtime": "2026-07-08T09:12:00",
  "source_size": 1048576000,
  "provider": "local_api",
  "created_at": "2026-07-08T14:32:01",
  "output_srt": "D:\\videos\\lecture.srt",
  "chunks": [
    { "id": 1, "start": "00:00:00", "end": "00:00:28", "status": "transcribed" },
    { "id": 2, "start": "00:00:28", "end": "00:00:55", "status": "vad_extracted" },
    { "id": 3, "start": "00:00:55", "end": "00:01:20", "status": "pending" }
  ]
}
```

`status` 진행 순서: `pending → vad_extracted → transcribed`. 실패한 Chunk는 `failed`로 표기되며 §21 오류 처리 정책(1회 재시도 후 실패 처리)을 따른다.

### 14.3 시작 시 동작

```text
1. 입력 파일 선택
2. job_id 계산
3. temp/{job_id}/manifest.json 존재 확인
   - 없음 → 새 작업 시작 (temp/{job_id}/ 신규 생성)
   - 있음 → 다이얼로그: "이전 작업을 발견했습니다. 이어서 진행하시겠습니까?"
       [이어하기] → status가 transcribed가 아닌 chunk부터 Phase A/B 재개
       [새로 시작] → 기존 temp/{job_id}/ 삭제 후 처음부터 진행
```

### 14.4 정리(Cleanup) 정책

```text
정상 완료 (모든 chunk transcribed, output_srt 작성 완료)
   → [정리 옵션 ON, 기본값] temp/{job_id}/ 전체 삭제
   → [정리 옵션 OFF] temp/{job_id}/ 보존 (디버깅/재사용 목적)

중단(Stop 버튼) 또는 비정상 종료(크래시)
   → temp/{job_id}/ 는 정리 대상에서 제외, 항상 보존
   → 다음 실행 시 §14.3의 재시작 로직으로 이어짐
```

---

## 15. 실시간 출력

Chunk 처리 완료 시:

1. STT 수행 (직전 Chunk 결과를 `{{context}}`로 주입)
2. SRT 항목 생성 → `{입력파일명}.srt`에 즉시 append
3. `chunk_NNNN.txt` 저장, manifest.json status 갱신
4. 현재 결과를 다음 Chunk용 Context로 저장 (메모리에는 1개 블록만 유지)
5. TextBox 갱신
6. ProgressBar 갱신

사용자는 처리 중에도 결과를 확인할 수 있어야 한다. 결과가 Chunk마다 디스크에 flush되므로 어느 시점에 중단되어도 그때까지의 SRT는 보존된다(원칙 2, 3).

---

## 16. 작업 제어

### 중단

* 작업 시작 시 `Transcript` 버튼이 `Stop`으로 변경.
* Stop 클릭 → `Cancel Requested` 상태 → 현재 Chunk 처리 완료 후 종료 (강제 종료 금지) → 버튼 복원.
* Managed 모드에서는 프로그램이 직접 실행한 llama-server 프로세스도 함께 정리.
* 중단 시 temp/{job_id}/는 보존되어 재시작 대상이 된다.

### Progress 표시

```text
처리된 오디오 시간 / 전체 오디오 길이    예: 00:15:00 / 01:00:00 (25%)
```

Phase A(전처리)와 Phase B(STT)의 진행률은 구분하여 표시한다:

```text
[INFO] Phase A: Chunking 12/40 (30%)
[INFO] Phase A complete.
[INFO] Phase B: Transcribing chunk 1/40
```

### 클립보드 복사

Copy 버튼 클릭 시 TextBox의 전체 SRT 내용을 클립보드에 복사.

---

## 17. Context Carryover

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
* 작업 취소 또는 새 파일 로드 시 초기화. Resume 시에는 직전 성공 Chunk의 `chunk_NNNN.txt`에서 복원.
* Provider 공통 동작.
* 프롬프트에서 `{{context}}` 미사용 시 자연스럽게 비활성화 (별도 ON/OFF 불필요).

---

## 18. Console Logging

레벨: DEBUG / INFO / WARNING / ERROR

```text
[INFO] Application started
[INFO] Loading file: lecture.mp4
[INFO] Duration: 01:23:15
[INFO] Resume: previous job found (job_id=a1b2c3d4, 12/40 chunks done)
[INFO] Starting ffmpeg decoder
[INFO] Audio format: 16000Hz Mono PCM
[INFO] Starting VAD
[INFO] Phase A: Chunk #13 Start=00:05:32 End=00:05:58
[INFO] Phase A complete (40 chunks)
[INFO] Launch mode: managed
[INFO] Starting llama-server (PID 18420)
[INFO] Waiting for model load...
[INFO] llama-server ready (12.4s)
[INFO] Phase B: Sending chunk #13
[INFO] Chunk #13 completed
[INFO] SRT updated
[INFO] Progress 33%

[DEBUG] Chunk #13 context: "...이전 청크의 인식 결과..."
[DEBUG] Chunk #13 encode+request time: 3.42s

[ERROR] API timeout
[ERROR] Chunk #15 failed

[INFO] Cancel requested
[INFO] Waiting current chunk
[INFO] Stopping llama-server (PID 18420)
[INFO] Worker stopped
[INFO] Cleanup: temp/a1b2c3d4/ removed
```

---

## 19. 스레드 구조

* **GUI Thread**: UI, ProgressBar, Buttons, TextBox
* **Worker Thread**: FFmpeg, PCM Stream, VAD+후처리, Chunking, (Managed 모드) llama-server 생명주기 관리, Prompt 조합(Context 치환), STT API, SRT 생성/append, manifest 갱신, Context 상태 관리
* Phase A와 Phase B는 같은 Worker Thread 안에서 순차 실행하되, UI에는 두 단계 진행률을 구분해 표시
* **Signal**: `progressChanged`, `chunkCompleted`, `phaseChanged`, `finished`, `error` — UI 직접 접근 금지, Signal/Slot 사용

---

## 20. 기술 스택

| 영역 | 기술 |
|---|---|
| GUI | PySide6 |
| 오디오 처리 | FFmpeg |
| VAD | Silero VAD (.onnx) + **onnxruntime (CPU)** — torch/torchaudio 의존성 없음 (원칙 4, §12.1) |
| HTTP | httpx |
| 설정 관리 | json |
| 비동기 처리 | QThread |
| 오디오 버퍼 | numpy, soundfile, io.BytesIO |
| 로깅 | logging |

PyTorch/TensorFlow는 의존성에 포함하지 않는다 (원칙 4).

---

## 21. 오류 처리 정책

* Chunk 단위 실패 허용: 특정 Chunk의 API 오류 시 해당 Chunk만 실패 처리하고 다음 Chunk 계속 진행. 실패 Chunk는 SRT에 `[TRANSCRIPTION FAILED]` 플레이스홀더로 표기.
* 실패 Chunk의 Context는 갱신하지 않고 직전 성공 결과 유지.
* 재시도: 타임아웃/일시 오류 시 1회 재시도 후 실패 처리.
* 연속 N회(기본 5회) 실패 시 작업 자동 중단 및 오류 안내 (서버 다운 등 환경 문제로 판단). 중단 시에도 temp는 보존되어 재시작 가능.
* Managed 모드에서 llama-server 기동 타임아웃(기본 120초) 시 오류 표시 후 작업 중단.
* **반복 할루시네이션 필터 (opt-in, `text_enhancement.dedup_repeated_chunks`, 기본 OFF)**: 오디오가
  흐릿하거나 무음에 가까울 때 모델이 실제 내용 대신 직전 Chunk의 인식 결과를 그대로 반복 출력하는
  패턴이 관측됨. 활성화 시, Chunk N의 인식 결과가 Context로 전달된 Chunk N-1의 텍스트와 (공백/후행
  문장부호 무시하고) 동일하면 할루시네이션으로 간주해 해당 Chunk는 SRT에서 제외하고
  `manifest.json`에 `"possible_duplicate_hallucination"` 플래그만 남긴다 (Context는 갱신하지 않고
  직전 성공 텍스트 유지 — 흐릿한 구간이 연달아 나와도 매번 그 앞의 "진짜" 텍스트와 비교됨). 기본
  OFF인 이유: 화자가 실제로 같은 문장을 두 번 말하는 경우까지 지워버릴 수 있어 사용자 판단에 맡김.

---

## 22. Vocal Extraction (보컬 분리) — 이번 버전 범위 제외, 결정 게이트

### 22.1 결정 배경

GPU 기반 소스분리 모델(Demucs 등)을 검토했으나 다음 이유로 **이번 버전 범위에서 제외**한다.

- 실사용 중 GPU 발열 및 시스템 RAM 32GB 이상 소비 확인 (원인: 파일 전체를 통짜로 처리하는 구조 — 1시간 처리 시 약 7GB, 4시간 처리 시 34GB 초과 사례 보고)
- Demucs/Spleeter 등은 PyTorch/TensorFlow 기반이라 RTX 50시리즈(Blackwell) 환경에서 설치 자체가 까다로움 (원칙 4 위배)
- **더 근본적으로, STT 기본 성능이 아직 실측 검증되지 않은 상태**에서 보정 수단부터 설계하는 것은 순서가 맞지 않음

### 22.2 결정 게이트

```text
언어별·모델별 STT 실측 (§5A 절차, VAD만 적용, 보컬 분리 없음)
   │
   ├─ 충분히 양호 → 보컬 분리 불필요. 본 설계 그대로 진행
   │
   ├─ 배경음/음악이 섞인 파일에서만 저하 → 보컬 분리를 차기 버전 옵션으로 도입 검토
   │                                        (§22.3 External Process 패턴 적용)
   │
   └─ 전반적으로 부적합 → 로컬 LLM 기반 STT 파이프라인 자체를 재검토.
                          §5A의 대안 모델로 회귀할지 프로젝트 방향 재결정
```

### 22.3 (참고, 차기 버전용) 도입 시 적용할 설계 방향

- **메인 앱과 완전히 격리된 External Process로 구성.** 별도 venv/환경에 PyTorch+CUDA 버전을 고정해 패키징하고, 메인 앱은 로컬 HTTP API(§6.3 Managed 패턴과 동일 구조) 또는 subprocess+파일 I/O로 결과만 받는다. 메인 앱은 PyTorch/TensorFlow 의존성을 갖지 않는다.
- **반드시 Chunk(≤30초) 단위로 처리**해 리소스 폭증을 원천 차단. 모델은 1회 로드 후 Chunk 순차 투입.
- Phase A(전처리)와 Phase B(llama-server) 사이에 시간적으로 분리 배치해 VRAM 동시 점유를 피한다.
- 기존 오픈소스(UVR5, Demucs CLI 등)를 별도 패키징해 재사용하는 것을 우선 고려, 직접 ONNX 포팅은 후순위.

이번 버전 설정 창(§8)에는 Vocal Extraction 옵션을 노출하지 않는다.

---

## 23. 향후 확장

```text
* OpenAI Audio API 지원
* Whisper Backend 지원 (B유형 어댑터 — §5A.3)
* 병렬 Chunk 처리 (llama-server --parallel N 활용)
* 화자 구분, 맞춤법 교정, 문장 정리
* 자막 포맷 개선, 번역 자막 생성, 다국어 지원
* Context Carryover Chunk 개수 설정 (현재 1개 고정)
* local_api.request_format 옵션 (비표준 포맷 서버 대응)
* VAD 알고리즘 교체 검토: TEN VAD vs Silero VAD 실측 비교 (§12.1)
  - 단, 오프라인 배치 처리이므로 TEN VAD의 핵심 강점(저지연 종료감지)은 효용이 낮음.
    정확도 축 위주로 비교하고, 최신 Silero 버전 유지도 함께 검토
* Vocal Extraction(보컬 분리) 도입 여부 (§22 결정 게이트)
  - 전제조건: STT 기본 성능(VAD만 적용) 실측 후 필요성 판단
  - 도입 시: External Process 격리, Chunk 단위 처리, Phase A/B 시간 분리 유지
* 언어별 STT 모델/Provider 자동 라우팅 (§5A.5)
  - 전제조건: §5A.2 CJK 후보 모델 실측 완료, 채택 모델 확정
  - 영어/서구어는 Whisper 고정, CJK는 채택된 특화 모델로 자동 분기
* Gemini API 다중 키 로테이션
  - 전제조건: gemini-3.1-flash-lite 실사용 가용성(무료 티어 체감 한도, 안정성) 검증
  - 도입 시: api_key → api_keys 배열 전환, 429+RESOURCE_EXHAUSTED 감지 기반 로테이션,
    소진 키는 프로그램 재시작 전까지 스킵(단순 방식). 키 소진에 의한 로테이션은
    연속 실패 카운트(§21)에서 제외하고 "모든 키 소진" 시에만 포함
* Windows Program Files 설치 환경에서 temp 경로 쓰기 실패 시 %LOCALAPPDATA% 폴백
* 강제 분할(§12.2) 시 에너지 낮은 지점 우선 절단
* Qwen3.5-Omni 라이선스/GGUF 공개 여부 주기적 재확인 (§5A.6)
* Text Correction(후처리 LLM) 도입 (§5B.3, §5B.6)
  - 전제조건: Pre-biasing(언어 힌트+Custom Vocabulary)만으로 충분한지 먼저 확인
  - 도입 시: Phase C 아키텍처 추가, 텍스트 모델과 STT 모델의 VRAM 비동시 점유 보장,
    긴 문서 슬라이딩 윈도우 청킹 전략 확정
* Custom Vocabulary 최적 분량/포맷 튜닝 (§5B.2) — 과다 용어 주입 시 효과 저하 가능성 실측 필요
```

---

## 24. 미결정 사항 (구현 전 확정 필요)

- 동명 SRT 파일 존재 시 처리 방식 (덮어쓰기 확인 vs 자동 접미사)
- VAD threshold 및 §12.2 병합 임계값의 초기 튜닝
- Test Connection에서 Gemini API 키 검증 범위 (기본: 1개 키만 테스트)
- Managed 모드 좀비 프로세스 감지 로직의 구체 구현

---

## 부록 A. 버전별 변경 이력

### v1 → v2 (요약)

| # | 변경 | 근거 |
|---|---|---|
| 1 | 기본 Provider를 Gemini → Local llama-server로 변경 | llama.cpp `PR #24118`(2026-06-04) 이후 오디오 입력 동작 검증 완료 |
| 2 | Local Provider를 "OpenAI-Compatible 일반"에서 "llama-server 특정"으로 구체화 | LM Studio/Ollama 미지원 확인 |
| 3 | 검증된 모델/실행 커맨드 명시 | unsloth GGUF Q8_0 + mmproj-F16 (6/5 이후 재업로드분) 동작 확인 |
| 4 | [Test Connection] 기능 추가 | v1 "런타임 검증 필수" 권고의 기능화 |
| 5 | `top_k` 파라미터, `disable_thinking` 옵션 추가 | Gemma 4 권장 샘플링, thinking 억제 필요 |
| 6 | 기본 temperature 0.2 → 1.0 | Gemma 4 권장값 |
| 7 | Chunk 최대 30초를 하드 리밋으로 격상 | Gemma 4 오디오 입력 상한 |
| 8 | 기본 포트 8080 → 8088 | 검증 커맨드 기준 |
| 9 | 오류 처리 정책 신설 | Chunk 실패/서버 다운 시 동작 명세화 |
| 10 | RTX 50시리즈 CUDA 빌드 주의사항 추가 | Blackwell 드라이버 버그(#23385) |
| 11 | Gemini 기본 모델을 `gemini-3.1-flash-lite`로 갱신 | 오디오 입력 지원 + 무료 티어 넉넉 |

### v2 → v3 (본 문서)

| # | 변경 | 근거 |
|---|---|---|
| 1 | 리소스 관리 4원칙 신설 (§0) | 실운용 중 GPU 발열·RAM 32GB+ 폭증, RTX 50 PyTorch 설치 이슈 확인 |
| 2 | Vocal Extraction은 범위 제외, 결정 게이트로 전환 (§22) | STT 기본 성능 미검증 상태에서 보정 수단부터 설계하는 것은 순서 부적절 |
| 3 | 처리 흐름을 Phase A(전처리)/Phase B(STT)로 분리 (§11) | GPU 모델 동시 상주 방지 (원칙 2) |
| 4 | 임시 파일 정책을 "생성 금지" → "Job 디렉토리 기반 적극 생성"으로 전면 수정 (§13) | 원칙 1 |
| 5 | Job 기반 재시작(Resume) 구조 도입, manifest.json (§14) | 원칙 3 |
| 6 | 출력 SRT 경로를 입력 파일과 동일 폴더로 고정 (§13) | 결과 파일 위치 찾기 번거로움 방지 |
| 7 | 임시 파일 경로를 실행 파일 디렉토리 기준으로 고정 (§13) | 입력 위치와 무관하게 항상 쓰기 가능 |
| 8 | "완료 후 임시 파일 정리" 옵션 추가, 기본 ON (§8, §14.4) | 정상 완료 시에만 정리, 중단/크래시 시 보존 |
| 9 | "Save SRT / Save TXT" 향후 확장 항목 제거 | SRT가 Chunk마다 최종 위치에 직접 append되어 불필요 |
| 10 | VAD 실측 한계 문서화 + 후처리(짧은 침묵 병합, 강제 분할) 추가 (§12) | Silero miss율 실측 확인, 원시 출력 그대로 쓰면 Chunk 수백 개 발생 |
| 11 | Silero VAD를 onnxruntime 직접 로드로 구현 (§12.1) | torch/torchaudio 의존성 회피 (원칙 4) |
| 12 | Test Connection을 Provider별 분기 (§6.4) | Local/Gemini 각각 다른 엔드포인트·페이로드 |
| 13 | llama-server Managed 모드 도입 (§6.3) | 수동 실행 번거로움 + 대기 중 VRAM 상시 점유 해소 |
| 14 | 언어별 STT 모델 전략 신설 (§5A) — 영어/서구어는 Whisper(v2), CJK는 별도 후보군 실측 | Whisper large-v3의 CJK 환각 문제 다수 확인, Gemma4 12B 스케일링 역전 벤치마크 확인 |
| 15 | Qwen3-ASR을 CJK 1순위 검증 후보로 확정 (§5A.6, §5A.7) | 제3자 한국어 실측(콜센터 1만 건)에서 Base 모델로도 Whisper large-v3-turbo 상회 |
| 16 | 백엔드 유형(A/B/C) 구분 도입 (§5A.3) | ASR 전용 모델과 멀티모달 LLM의 서빙 스택이 다름을 확인 |
| 17 | Qwen3-ASR 기본 서빙 경로를 llama.cpp(A유형)로 확정, vLLM 제외 | vLLM은 RTX 50에서 사전빌드가 실패(#35432), llama.cpp가 더 현실적 |
| 18 | server_binary 등 경로 필드를 config 최상위로 이동 (§9) | 모드 전환 시 재입력 방지 |
| 19 | Qwen3-Omni-30B-A3B 후보 추가, Qwen3.5-Omni는 제외 (§5A.6) | 전작은 ggml-org GGUF로 즉시 실행 확인, 후속작은 오픈웨이트 미확인 |
| 20 | Gemini API 다중 키 로테이션은 차기 보류 (§23) | gemini-3.1-flash-lite 가용성 검증 선행 필요 |
| 21 | 언어 강제 지정({{language_hint}}) 및 CER 정규화·자체 실측 반영 (§5A.4, §5A.6, §5A.8, §8, §9) + Qwen3-ASR VRAM 튜닝 커맨드 (§6.1) | Claude Code 실측 세션(2026-07-08~09): 한국어→일본어 오판이 `language: ko` 한 줄로 정상화, FLEURS 9샘플 정규화 CER ja 1.6%/zh 1.9%/ko 9.9% 확정, `-hf` 기본 실행 시 VRAM 10GB+ 소모 문제를 --ctx-size 4096 등으로 약 4GB로 절감 |
| 22 | 텍스트 정확도 강화(§5B) 신설 — Custom Vocabulary({{vocabulary}})는 이번 버전 포함, Text Correction(Phase C)은 차기 검토 | 언어 힌트 실측 효과를 일반화: 용어집 주입(인식 전 Pre-biasing)과 LLM 후처리(인식 후 문맥 재교정)를 역할 분리. 후처리는 별도 텍스트 모델 및 Phase C 아키텍처가 필요해 구현 부담이 커 Pre-biasing 우선 채택. §5A.6 잔존 오류(고유명사 이표기)가 용어집으로 잡히는지가 1차 검증 포인트 |

---

## 부록 B. 비대상 서버 호환성 메모 (2026-07 기준)

| 서버 | `input_audio` 지원 | 비고 |
|---|---|---|
| llama.cpp `llama-server` | ✅ 지원 (experimental) | Gemma 4 + `gemma4uv` mmproj (2026-06-05 이후 빌드/GGUF), Qwen3-ASR (`ggml-org` GGUF, `libmtmd`) |
| LM Studio | ⚠️ 버전 의존 | 번들 llama.cpp가 upstream을 즉시 따라가지 않음. 공식 문서에 `input_audio` 언급 없음. 지원 대상 아님 |
| Ollama | ❌ 미지원 | OpenAI 호환 레이어에 오디오 필드 없음. 자체 API용 PR `#15243` 미병합 |
| vLLM | ⚠️ 지원하나 RTX 50 문제 | Qwen3-ASR 공식 권장 경로이나, Blackwell에서 사전빌드 wheel/Docker 모두 실패 사례(#35432). 로컬 경로로는 채택하지 않음 |
| whisper.cpp | (별도 형식) | `input_audio`가 아닌 `/inference` 또는 `/v1/audio/transcriptions`(멀티파트). B유형 백엔드 — 채택 시 별도 어댑터 필요 (§5A.3) |

---

## 부록 C. 참고 링크

### llama.cpp / Gemma 4
* [llama.cpp PR #24118: Fix Gemma 4 Unified conversion](https://github.com/ggml-org/llama.cpp/pull/24118)
* [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
* [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)
* [Running Gemma-4-12B Audio on llama.cpp (note.com)](https://note.com/unco3/n/n871e994d27b2?hl=en)
* [Blackwell CUDA Toolkit 이슈 (llama.cpp #23385)](https://github.com/ggml-org/llama.cpp/issues/23385)
* [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)

### Qwen3-ASR / Qwen3-Omni
* [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF)
* [ggml-org/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)
* [ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF)
* [Qwen3-ASR 한국어 실측 벤치마크 (mz-moonzoo.tistory.com/133)](https://mz-moonzoo.tistory.com/133)
* [Qwen3-ASR 기술 리포트 리뷰 (mz-moonzoo.tistory.com/129)](https://mz-moonzoo.tistory.com/129)
* [Qwen3.5-Omni 소개 (news.hada.io)](https://news.hada.io/topic?id=28027)
* [Qwen3-ASR 긴 오디오 버그 (llama.cpp #21847)](https://github.com/ggml-org/llama.cpp/issues/21847)

### 기타 CJK ASR 후보
* [SenseVoice.cpp](https://github.com/lovemefan/SenseVoice.cpp)
* [FunASR](https://github.com/modelscope/FunASR)

---

## 부록 D. Gemini API 이용 가이드 (보조 Provider)

로컬 GPU가 없거나 `llama-server` 준비가 어려운 환경에서 사용하는 보조 Provider(§5.2, §10.2)에 대한 상세 이용 가이드.

### D.1 사용 가능 모델 (오디오 입력 STT 용도)

| 모델 코드 | 상태 | 오디오 입력 | 무료 티어 1일 한도(참고) | 비고 |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | **Stable (권장)** | ✅ 지원 | 넉넉함(약 300~500 RPD 수준) | 저지연·저비용, 고빈도 STT에 최적. 입력 Text/Image/Video/**Audio**/PDF, 출력 Text |
| `gemini-3-flash-preview` | Preview | ✅ 지원 | 약 20 RPD로 제한적 | 시험용으로만 권장. 대량 Chunk 처리에 부적합 |

`gemini-3.1-flash-lite` 주요 스펙: 입력 토큰 한도 1,048,576 / 출력 65,536, Knowledge cutoff 2025-01. **Audio generation·Live API는 미지원** — 파일 기반 전사만 가능(본 프로그램에는 제약 없음).

> **주의 (Gemma vs Gemini):** Gemini API로 호스팅되는 **Gemma 4 모델(`gemma-4-31b-it` 등)은 오디오 입력이 불가**하다("Audio input modality is not enabled" 오류). Gemini API 경로에서는 반드시 **Gemini 계열 모델**을 사용한다. (오디오 학습된 Gemma 4 변형은 Gemini API에 호스팅되지 않으며 로컬 실행 필요 — §5.1 llama-server 경로)

### D.2 오디오 입력 방식

30초 Chunk(16kHz mono WAV ≈ 1MB)는 항상 20MB 미만이므로 **인라인(`inline_data`) 방식을 사용**한다. Files API는 사용하지 않는다.

지원 오디오 MIME: `audio/wav`, `audio/mp3`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac`. 본 프로그램은 WAV로 통일.

참고: 오디오 1초 = 32토큰(1분 ≈ 1,920토큰), 단일 프롬프트 최대 9.5시간. 30초 Chunk는 약 960토큰.

### D.3 요청 샘플 (Python / httpx)

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

### D.4 응답 파싱

성공 응답의 전사 텍스트 경로: `candidates[0].content.parts[0].text`

* 안전 필터 등으로 `candidates`가 비거나 `finishReason`이 `STOP`이 아닐 수 있으므로 파싱 전 존재 여부를 확인하고, 실패 시 §21 오류 처리 정책(Chunk 단위 실패 허용, 1회 재시도)을 따른다.

### D.5 참고 링크

* [Gemini API — Audio understanding](https://ai.google.dev/gemini-api/docs/audio)
* [Gemini API 모델 목록](https://ai.google.dev/gemini-api/docs/models)
* [Run Gemma with the Gemini API (오디오 미지원 — 대조용)](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
