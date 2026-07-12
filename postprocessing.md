# ASR 후처리 설계 v3

문서 버전: 3.0
이전 버전 대비 주요 변경: [부록 A](#부록-a-변경-이력) 참고

이 문서는 Phase C(텍스트 후처리)의 자기완결적 설계이다. 상위 파이프라인(Phase A/B, Provider, VAD 등)은 `design.md`를 참고한다.

---

## 0. v3 핵심 변경 요약 (먼저 읽을 것)

v2 대비 세 가지가 근본적으로 바뀌었다. 아래를 먼저 이해하면 나머지 문서가 쉽게 읽힌다.

1. **Sliding Window → Full-context 교정으로 전환 (§6).**
   v2는 "직전 교정 완료 N문장"만 context로 줬으나, v3는 **원본 전사(raw) 전체를 고정 context로 주입**하고 현재 문장만 교정한다. 실제 자막 텍스트는 수십 KB~100KB 수준으로 작아, 큰 context 한 번(또는 2~3개 대구간)으로 문서 전체를 담을 수 있다. 이 전환 하나로 세 가지가 동시에 해결된다.
   - 동음이의어 판별 문맥이 문서 전체로 넓어짐 (v2의 "1~2문장 앞으로는 애매" 문제 해소)
   - **오교정 전파(v2 §6 실패 모드)가 원천 소멸** — context가 raw라서 전파될 "교정 이력"이 없음
   - **별도 glossary 추출 pass 불필요** — 매 호출이 원문 전체를 보므로 고유명사가 문서 어디에 나오든 그 자리에서 참고 가능

2. **화자 전환 탐지 추가 (§10).**
   연속 대화가 한 자막으로 뭉쳐 한 화면에 통으로 표시되는 것을 막기 위해, 후처리 LLM이 텍스트 문맥만으로 화자 전환 지점을 탐지해 `[[SPEAKER:N]]` 마커를 삽입한다. **오디오 기반 diarization은 사용하지 않는다**(원칙 4 위배 + 화자 신원 추적은 텍스트로 신뢰도 확보 불가). 라벨은 **각 cue 내부에서만 유효한 상대 번호**이며, 문서 전역 일관성이나 성별/개인 프로필 추적은 하지 않는다.

3. **Cue Splitter 추가 (§11).**
   화자 마커 기준 분할 + CPS(초당 문자수) 초과 시 규칙 기반 추가 분할 + 시간 근사 배분을 담당하는 결정론적 후단 유틸리티. 이 단계는 **LLM을 쓰지 않는다**(요약/의역 위험 차단).

4. **설정 항목 정리 및 config.json 통합 (§12).**
   위 세 가지를 포함해 Phase C 전반의 configurable property(서버 구동값, 샘플링, full-context 분할, 화자 탐지, cue splitter)를 표로 정리하고 기본값을 확정했다. 별도 파일로 쪼개지 않고 기존 `config.json`의 `text_enhancement.text_correction` 아래 통합한다.

---

## 1. 목적

Qwen3-ASR로 생성된 전사 결과의 품질과 자막 가독성을 향상시키기 위한 후처리 단계이다.

후처리는 두 종류의 작업을 수행한다.

**(A) 텍스트 교정 — 최소한의 수정만으로:**

* 문맥에 맞지 않는 동음이의어
* 잘못 인식된 고유명사 (음차·표기 오류)
* 중복 단어 (예: `主 主`, `주 주요`)
* 조사 및 띄어쓰기 오류
* 명백한 오탈자

**(B) 자막 구조화 — 의미 변경 없이:**

* 뭉쳐진 연속 대화의 화자 전환 지점 분할 (§10, §11)
* 한 화면에 과도하게 긴 자막의 cue 분할 (§11)

반대로 다음 작업은 **수행하지 않는다.**

* 문장 재작성
* 의역
* 내용 추가
* 요약
* 문체 변경

즉, **ASR 결과를 최대한 보존하면서 필요한 부분만 수정하고, 표시 단위만 재구성하는 것**이 목표이며, 불확실한 경우 원문을 유지하는 것을 기본 원칙으로 한다.

---

## 2. 모델 선정

후처리 모델은 **Qwen3.5-9B**를 기본으로 사용한다. (대안: Qwen3.6-27B — VRAM 여유가 있고 품질을 우선할 경우)

### 선정 이유

* Instruction 준수 능력이 우수함 (모델 카드 IFEval 91.5)
* JSON/고정 포맷 출력 안정성이 높음
* 응답 속도가 빠름 (문장 단위 다수 호출에 유리)
* 최신 llama.cpp에서 동작 확인 (아키텍처 주의사항은 §3)
* 16GB VRAM 환경에서 Q6_K 운용 가능 (양자화·VRAM 상세는 §3)

ASR 후처리는 복잡한 추론보다 **원문 보존 · 최소 수정 · 높은 Instruction 준수**가 중요하다. 따라서 대형 추론 특화 모델보다 9B급 instruct 모델이 비용/속도 측면에서 적합하다.

> **모델 특성 주의:** `unsloth/Qwen3.5-9B-GGUF`는 순수 텍스트 모델이 아니라 **Hybrid Gated DeltaNet + sparse MoE + vision encoder를 가진 멀티모달(VL) 모델**이다. 후처리는 텍스트만 쓰므로 기능상 문제는 없으나, (1) 신규 아키텍처라 최신 llama.cpp 빌드가 필수이고, (2) vision encoder가 VRAM에 얹힐 수 있다. 상세와 실측 항목은 §3.

### ⚠️ Thinking 모드 처리 (중요)

Qwen3.5 / Qwen3.6 계열은 **기본적으로 thinking 모드로 동작**하며, `<think>...</think>` 형태의 추론 블록을 먼저 생성한 뒤 최종 응답을 출력한다. 본 후처리 용도에서는 이 동작이 문제가 된다.

* 후처리는 "최소 수정·과도한 reasoning 배제"가 목표인데, thinking은 이와 정면으로 배치된다.
* `n-predict`를 256~512로 제한한 상태에서 thinking이 켜져 있으면, **thinking 블록만 채우다 출력 토큰이 소진되어 실제 교정 결과가 비는 사고**가 발생한다.

따라서 **non-thinking(instruct) 모드를 명시적으로 활성화**해야 한다.

* llama.cpp: 채팅 템플릿에 `enable_thinking=false`를 전달하거나, 요청에 `chat_template_kwargs: {"enable_thinking": false}`를 포함한다.
* 모델별 non-thinking 비활성화 방법은 해당 모델 카드의 안내를 따른다.

### 샘플링 파라미터

Qwen3.5 **non-thinking(instruct) 모드 권장값**을 기준으로 하되, 후처리 목적에 맞게 결정성을 높인다. 구체적 채택값과 모델 공식 권장값 대비 조정 근거는 **§3의 "샘플링 파라미터" 표**에 정리한다(요약: temp 0.2~0.3, top_p 0.8, top_k 20, presence_penalty 0.5~1.5, 모두 실측 튜닝 대상).

> **핵심 원칙:** 이 계열은 temp=0에서 반복·붕괴 경향이 있어 0으로 내리지 않는다. 낮은 temp + 낮은 top_p 조합이 재현성과 안정성의 실용적 절충점이다.

> **표기 일관성 관련 주의(v3):** full-context 방식(§6)에서는 모델이 매 호출마다 원문 전체를 보지만, 이전 호출에서 특정 고유명사를 "어떻게 교정했는지"는 기억하지 못한다(context가 raw이므로). 따라서 같은 고유명사가 문서 앞뒤에서 샘플링 편차로 미묘하게 갈릴 수 있다(예: 앞 "베르사체" / 뒤 "베르사치"). temp를 낮게 유지하면 이 변동은 줄지만 완전히 없어지진 않는다. 관측 시 대응은 §7 참고.

---

## 3. llama.cpp 실행

### 검증 대상 모델 (v3)

후처리 모델로 `unsloth/Qwen3.5-9B-GGUF`를 1순위 검토한다. 확인된 사실과 주의점:

* **아키텍처는 Hybrid Gated DeltaNet + sparse MoE + vision encoder(멀티모달 VL).** 순수 dense 텍스트 모델이 아니다. 후처리는 텍스트만 쓰므로 기능상 문제는 없으나, 아래 두 가지가 파생된다.
  * **최신 llama.cpp 빌드 필수.** Gated DeltaNet의 신규 연산자(operator)를 지원하려면 반드시 최신 빌드여야 한다. 오래된 빌드는 GGUF 변환은 되어도 추론이 깨질 수 있다. Qwen3.5는 출시 당일부터 llama.cpp 지원이 확인됐고 4B GGUF 구동 사례가 문서화되어 있으나, 9B·특정 양자화·RTX 50 조합은 실측 확인이 필요하다(아래 실측 항목).
  * **vision encoder VRAM.** 텍스트 전용 로드로 vision 가중치를 스킵하는 옵션이 llama.cpp에 있는지 미확인. 없으면 vision 파트도 VRAM을 점유한다(실측 항목).
* **기본 thinking 모드.** 모델 카드가 명시적으로 확인. 반드시 비활성화한다(§2). MTP(speculative decoding) 버전은 속도 이점이 있으나 별도 PR 브랜치 빌드가 필요하고 이번 용도엔 불필요하므로 일반 GGUF를 쓴다.

### 실행 커맨드 (예시, 검증 대상)

```bash
llama-server \
  -m Qwen3.5-9B-Q6_K.gguf \
  --host 0.0.0.0 \
  --port 8081 \
  --ctx-size 32768 \
  --parallel 1 \
  --n-gpu-layers 999 \
  --threads 12 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --reasoning-budget 0 \
  --jinja \
  --metrics
```

커맨드 주요 사항:

* **포트 8081 (v3.1 정정).** design.md v3.2에서 STT 서버(Phase B)의 실제 기본 포트가 8080으로 확정되어, Phase C 서버는 8081을 쓴다(§12.2 config와 일치). Phase B/C는 시간적으로 분리 구동되므로(원칙 2) 런타임 충돌은 없으나, 설정·로그에서 두 서버를 혼동하지 않도록 포트를 구분한다.
* **양자화 Q8_0 → Q6_K로 하향(v3).** full-context(32K) 방식은 KV 캐시가 커, 16GB VRAM에서 Q8_0(9.53GB)은 빠듯하다. Q6_K(7.46GB)로 낮춰 KV 캐시 공간을 확보한다. 후처리는 최소 교정 용도라 Q8→Q6 품질 저하가 거의 체감되지 않을 것으로 예상. 그래도 부족하면 UD-Q4_K_XL(5.97GB)로 한 단계 더 내릴 여유가 있다.
* **`-fa on`**: 최신 llama.cpp는 flash attention을 명시적 on/off로 받는다(`--flash-attn` 단독 표기보다 안전).
* **`--reasoning-budget 0`**: thinking 억제(§2 필수 요건). Qwen3.5-4B llama-server 예시에서 실제 사용된 방식. **단, 이 플래그가 9B·해당 빌드에서 thinking을 확실히 끄는지는 첫 응답에 `<think>` 블록이 없는지로 검증할 것.** 안 되면 요청 body에 `chat_template_kwargs: {"enable_thinking": false}`(모델 카드 명시 방법)로 폴백한다.
* **`--jinja`**: 내장 챗 템플릿 사용. thinking 제어가 템플릿 경유이므로 필요.
* **`--cache-type-k/v q8_0`**: KV 캐시 양자화로 VRAM 절감(§Context 크기 참고).
* 출력 토큰 상한(`n_predict`)은 서버 기동 시 고정하기보다 **요청별 파라미터**로 넘기는 것을 권장. GPU offload 플래그는 `--n-gpu-layers`(축약 `-ngl`).

### 샘플링 파라미터 (모델 공식 권장 반영, v3)

모델 카드의 **non-thinking(instruct) 일반 태스크 권장값**은 `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1.0`이다. 후처리는 결정성을 높여야 하므로 temp를 낮추되, 이 계열이 temp=0 근처에서 붕괴 경향이 있으므로 0으로 내리지는 않는다. 커맨드에 고정하기보다 **요청별 파라미터**로 넘겨 실측 튜닝한다(§8.3).

| 항목 | 모델 공식 권장 | 후처리 채택(초기값) | 비고 |
|---|---|---|---|
| Temperature | 0.7 | 0.2 ~ 0.3 | 결정성↑ 목적으로 하향. **0.15는 다소 공격적** — 붕괴 관측 시 상향. 실측 튜닝 대상 |
| Top-p | 0.8 | 0.8 | 공식값 유지 |
| Top-k | 20 | 20 | 공식값 유지 |
| Presence penalty | 1.5 | 0.5 ~ 1.5 | 반복 억제용. 과하면 언어 혼합 유발 가능(모델 카드 경고), 낮게 시작 |
| Repetition penalty | 1.0 | 1.0 | 공식값 유지 |

> v2는 temp=0.15를 제시했으나, 모델 공식 non-thinking 권장이 0.7인 점을 감안하면 0.15는 공격적일 수 있다. 0.2~0.3에서 시작해 반복·붕괴가 없으면서 재현성이 확보되는 지점을 실측으로 찾는다.

### 권장 설정 요약

| 항목 | 권장값 |
|---|---|
| 모델 | `unsloth/Qwen3.5-9B-GGUF` (검증 대상) |
| 양자화 | **Q6_K (v3, 16GB+32K context 여유 확보)** — 부족 시 UD-Q4_K_XL |
| Context | **32768 (v3, full-context 수용)** |
| Max Predict (요청별) | 256 ~ 512 |
| Temperature | **0.2 ~ 0.3 (실측 튜닝)** |
| Top-p | 0.8 |
| Top-k | 20 |
| Presence penalty | 0.5 ~ 1.5 |
| Repetition penalty | 1.0 |
| Flash Attention | `-fa on` |
| KV cache type | q8_0 (k/v) |
| GPU Layers | 전체 Offload |
| Thinking | **비활성화 (필수, `--reasoning-budget 0` + 검증)** |

### 실측 확인 항목 (구동 전/직후 필수)

kh의 "실측 후 확정" 원칙에 따라 아래를 구동 시 반드시 확인한다. 하나라도 실패하면 모델 채택 자체를 재검토한다.

1. **아키텍처 추론 정상성** — 최신 llama.cpp 빌드로 `qwen35`(DeltaNet+MoE) 추론이 gibberish 없이 정상인지. **최우선.** 실패 시 이 모델 후보 탈락.
2. **thinking 억제 실동작** — 첫 응답에 `<think>` 블록이 없는지. `--reasoning-budget 0`이 안 먹으면 `chat_template_kwargs` 폴백.
3. **vision encoder VRAM 점유** — 텍스트 전용 로드가 되는지, 안 되면 Q6_K + 32K KV + vision 가중치 합산 VRAM이 16GB에 들어가는지. 넘치면 양자화 하향(Q4_K_XL).
4. **prefix 캐시 실동작** — 원문 전체 고정 prefix에 대해 KV 캐시가 재사용되는지(성능 결정적, 아래 별도 항목).
5. **RTX 50 CUDA 빌드** — CUDA 13.2 gibberish 이슈 회피(§CUDA 빌드 주의).

### Context 크기 (v3 변경)

v2는 Sliding Window라 8192로 충분했으나, v3는 원문 전체를 고정 context로 주입하므로 **더 큰 context가 필요**하다. Phase C는 Phase B(STT) 종료 후 GPU 전체를 텍스트 모델이 단독 사용하므로(design.md §5B.3), audio chunk 처리 때의 보수적 ctx-size 제약을 그대로 가져올 이유가 없다.

* **32768 권장.** Qwen 계열 토크나이저 기준 한국어 대략 1.5~2자/토큰이므로 32768 토큰 ≈ 5~6만 자(대략 50~60KB). 실제 자막 파일 크기(수십 KB~100KB 이하)의 대부분을 한 번에 담는다.
* 파일이 32768 토큰을 초과하면 문서를 2~3개 대구간으로 나눠 각 구간을 고정 context로 주입한다(§6 참고). 몇 시간짜리 초장편만 해당.
* 9B Q8_0 기준 32768 KV 캐시 증가분은 16GB VRAM에서 감당 가능한 수준. 부족 시 `--cache-type-k q8_0 --cache-type-v q8_0`로 완화.

### 출력 토큰 제한

후처리는 현재 문장 하나(+화자 마커)만 교정·구조화하므로 긴 출력이 필요 없다. `n_predict` 256~512 권장. 출력 제한의 이점은 다음과 같다.

* 원문 context를 통째로 다시 출력하는 실수 방지
* 장황한 설명·불필요한 reasoning 출력 방지
* 처리 시간 단축, GPU 메모리 절감
* 동일 입력에 대한 결과 안정화

**단, thinking을 비활성화한 상태에서만 이 토큰 예산이 유효하다** (§2 참고).

### Prefix 캐시 (v3 신규 — 성능 필수 확인 항목)

v3는 "원문 전체(고정 prefix) + 현재 교정 대상 문장(가변)" 구조로 매 요청을 보낸다. 원문 전체가 매번 동일한 prefix로 반복되므로, **llama-server의 prompt/KV prefix 캐시가 제대로 걸리면 원문 전체를 매 요청마다 다시 계산하지 않아도 된다.**

* 이것이 안 걸리면 문장마다 수만 토큰을 재계산하게 되어 비용이 폭증한다. **실측 단계에서 반드시 확인할 것.**
* 요청 간 prefix가 바이트 단위로 동일해야 캐시가 걸리므로, 원문 context 블록은 요청마다 **정확히 같은 문자열**을 유지한다(공백/줄바꿈 포함).
* 현재 교정 대상 문장은 prefix 뒤에 붙여, 캐시 경계가 prefix 끝에 형성되도록 프롬프트를 배치한다(§8.3).

### Stop Sequence

출력 형식을 `<OUTPUT>...</OUTPUT>`으로 고정하고 `</OUTPUT>`을 Stop Sequence로 지정한다. 모델이 교정 문장을 출력한 뒤 추가 설명이나 다음 문장을 생성하는 것을 방지한다.

### CUDA 빌드 주의 (RTX 50 / Blackwell)

* Qwen3.5/3.6을 llama.cpp에서 구동할 때 **CUDA 13.2 빌드는 gibberish 출력 이슈**가 보고되었다. CUDA 13.2 미만 또는 13.3을 사용한다.
* 출력이 gibberish이면 context length가 너무 낮게 잡혔거나, `--cache-type-k bf16 --cache-type-v bf16` 시도로 완화되는 경우가 있다.

---

## 4. 후처리 파이프라인 (Phase C 전체 흐름)

ASR을 **전체 완료한 후**(Phase B 종료, STT 서버 내림) 후처리를 수행한다. v3 전체 흐름은 다음과 같다.

```
Phase B 완료 (raw transcript 전체 확보)
   ↓
[문장 분리 저장] raw/000001.txt ... (원본 보존, §5)
   ↓
[Full-context 교정 + 화자 탐지] (§6, §10)
   - 원문 전체를 고정 context로 주입
   - 문장별로: 동음이의어·고유명사·오탈자 교정
   - 100자(튜닝) 초과 시 같은 호출에서 화자 전환 지점에 [[SPEAKER:N]] 마커 삽입
   - 결과: fixed/000001.txt ... (교정본, 마커 포함 가능)
   ↓
[Cue Splitter] (§11, 결정론적 — LLM 미사용)
   1차: [[SPEAKER:N]] 마커 기준 cue 분할
   2차: CPS 초과 cue를 문장부호/어절 경계로 규칙 기반 추가 분할
   3차: chunk duration을 조각별 문자 수 비율로 근사 배분
   ↓
SRT 최종 출력 ({입력파일명}.srt) / 원본은 {입력파일명}.raw.srt로 별도 보존
```

ASR 진행 중에는 후처리 LLM을 호출하지 않는다. ASR과 후처리를 분리하면 구현이 단순해지고, 후처리 실패 시 전사를 다시 할 필요가 없다.

> **v2의 "glossary 사전 구축 단계"는 v3에서 제거되었다.** full-context 방식이 그 역할을 흡수한다(§0, §7 참고).

---

## 5. 전사 저장

전사 완료 후 문장 단위로 분리 저장한다. **원본과 교정본을 모두 보존한다.**

```
raw/000001.txt      fixed/000001.txt
raw/000002.txt      fixed/000002.txt
...                 ...
raw/000845.txt      fixed/000845.txt
```

* 문장 번호는 후처리 시 순서 유지 및 원문↔교정본 대응에 사용한다.
* **원본을 별도 보존하는 이유**: 오교정을 나중에 발견해도 특정 문장만 원본에서 재교정할 수 있고, 원문 전사가 그대로 full-context 재료로 남는다. 추가 비용은 디스크 수 KB 수준이며, "실패 시 특정 문장만 재실행" 장점과 맞물린다.
* SRT 레벨에서도 교정 전 원본을 `{입력파일명}.raw.srt`로 보존해, 교정이 개선을 보장하지 않는 경우 비교/롤백이 가능하게 한다(design.md §5B.3과 일치).

---

## 6. Full-context 교정 (v3 핵심)

후처리는 **현재 문장 하나만** 수정하며, **원본 전사(raw) 전체**를 참고 context로 함께 제공한다.

```
[원본 전사 전체 — 고정 context, 수정 금지]
1  2  3  ...  844  845    (raw ASR, 변하지 않음)

[현재 교정 대상 — raw]
101
      ↓
[출력]
101'   (교정본, 필요 시 [[SPEAKER:N]] 마커 포함)
```

다음 문장으로 넘어가도 **context(원문 전체)는 그대로**이고, 교정 대상 번호만 102, 103...으로 이동한다.

즉,

* context는 항상 **원본 전사 전체** (교정본이 아님 → 전파 없음)
* 현재 문장만 수정
* 출력도 현재 문장 하나

### 왜 full-context인가 (v2 Sliding Window 대비)

v2는 "수백 문장 이전 내용이 현재 문장에 영향 줄 가능성은 낮다"고 보고 최근 N문장만 유지했다. 그러나 실제로는:

* **동음이의어 판별에 원거리 문맥이 필요한 경우가 있다.** 무척/부쩍, 족히/조기 류는 바로 앞뒤로 충분하지만, 고유명사·전문용어·화제 일관성은 문서 전체를 봐야 정확히 잡힌다. 1~2문장 앞만으로는 애매하다.
* **파일이 작다.** 자막 텍스트는 수십 KB~100KB 수준이라 원문 전체를 context에 담는 것이 물리적으로 가능하다(§3). v2가 full-context를 피한 것은 "긴 오디오"를 전제한 보수적 판단이었으나, 텍스트 단계에서는 그 전제가 성립하지 않는다.

```
A: 이번에 베르사체 신상 봤어?
   ...(중략 20문장)...
[ASR] 아까 그 벨사지 진짜 예쁘더라.
  → 20문장 떨어져 있어도 원문 전체 context 안에 "베르사체"가 있으므로 교정 가능
    (Sliding Window였다면 window 밖으로 밀려나 놓쳤을 케이스)
```

### 오교정 전파 문제의 소멸 (v2 §6 실패 모드 해결)

v2의 알려진 실패 모드는 "교정 완료본을 다음 window의 context로 재공급 → 한 번 잘못 교정된 표기가 이후로 전파"였다. v3는 **context가 교정본이 아니라 원본(raw)**이므로, 애초에 전파될 "교정 이력"이 존재하지 않는다. 각 문장은 오염되지 않은 원문 전체만 보고 독립적으로 교정된다. 이는 v2 부록 B의 "원문 window" 완화 옵션을, 파일이 작다는 사실 덕분에 window 단위가 아니라 **문서 전체 단위로 확장 적용**한 것에 해당한다.

### 문장별 병렬 처리 가능 (v3 이점)

각 문장 교정이 오직 "고정 원문 context + 해당 문장"에만 의존하고 이전 교정 결과에 의존하지 않으므로, **문장별 병렬 교정이 원리적으로 가능**하다. prefix 캐시(§3)가 걸린 상태라면 병렬 슬롯 간 prefix 공유 효과도 기대할 수 있다. 다만 초기 구현은 순차로 두고, 성능 필요 시 병렬화한다(§7).

### 긴 파일 처리 (문서가 context를 초과할 때)

원문이 ctx-size(예: 32768 토큰)를 초과하는 초장편은 문서를 **2~3개 대구간**으로 나눠, 각 구간을 그 구간 문장들의 고정 context로 사용한다. 구간 경계에서 고유명사 일관성이 약간 흔들릴 수 있으나, 대부분의 실사용 파일은 단일 구간에 들어가므로 예외 처리로 둔다.

---

## 7. 향후 개선

* **표기 일관성 보조 장치 (선택).** full-context가 대부분의 일관성을 해결하지만, §2에서 언급한 "샘플링 편차로 인한 고유명사 표기 흔들림"이 실측에서 관측되면, 가벼운 glossary(문서 1회 스캔으로 정규 표기 확정 후 프롬프트에 고정 주입)를 보조로 추가한다. **처음부터 넣지 않고, 흔들림이 실제로 보일 때만 추가**한다(Ship simpler first). v2에서 이 glossary는 "전파 차단"이 주목적이었으나 v3에서는 전파가 이미 없으므로, 목적이 "표기 통일"로 축소된다.
* Window/구간 크기 자동 조절 (초장편 대응)
* 교정 신뢰도 출력 / 수정 Diff 출력 (콘솔 로그에 변경분 기록)
* 사람이 검토할 문장 자동 표시
* 교정 결과 캐시
* 문장별 병렬 처리 정식 도입 (§6)
* **Word-level timestamp 확보 방안 검토** — §11 시간 근사 배분의 부정확성을 근본적으로 개선. STT 모델이 word timestamp를 지원하거나 별도 forced-alignment 도구를 붙이면, cue 분할 시 문자 수 비례가 아닌 실제 발화 시각으로 배분 가능.
* 화자 라벨 화면 노출 옵션 (§10 — 기본은 노출 안 함)

---

## 8. 교정 Prompt

### 8.1 System Prompt

보수적으로 작성한다. **원문 context 블록과 교정 대상을 XML 태그로 명시적으로 구획**하여, "원문 context는 수정 금지" · "현재 문장만 출력" 규칙이 안정적으로 지켜지도록 한다.

```text
You are an Automatic Speech Recognition (ASR) post-processing assistant.

Your task is to recover the speaker's intended utterance from an ASR transcript.
You are given the FULL raw ASR transcript inside <FULL_TRANSCRIPT> tags for
reference only, and the current segment to correct inside <CURRENT> tags.

The current segment may contain recognition errors such as:
- homophone substitutions
- incorrect proper nouns (transliteration / character errors)
- duplicated words or phrases
- obvious character mistakes
- minor punctuation or spacing errors

Rules:
1. Correct only clear ASR recognition errors in the <CURRENT> segment.
2. Use <FULL_TRANSCRIPT> only to disambiguate homophones and proper nouns and to
   keep terminology/proper-noun spelling consistent across the whole document.
3. Preserve the speaker's intended message, style, and tone.
4. Do NOT rewrite, summarize, paraphrase, or improve the sentence.
5. Do NOT add information that was not spoken.
6. If multiple valid interpretations exist and context cannot decide, keep the original text unchanged.
7. NEVER modify or output the transcript context. It is for understanding only.
8. Correct ONLY the current segment.
9. If (and only if) the current segment clearly contains an exchange between
   different speakers (e.g. a question immediately followed by its answer),
   insert the marker [[SPEAKER]] at each point where the speaker changes.
   Do NOT number or name speakers. Do NOT insert the marker when a single
   speaker is talking continuously. When in doubt, do NOT insert it.
10. Output ONLY the corrected current segment wrapped in <OUTPUT> tags. No
    explanations, no comments, no markdown, no quotation marks, no numbering.

The transcript may be in Japanese, Chinese, Korean, or a mixture of these.
Apply each language's normal writing conventions only when needed to fix an obvious ASR error.

Be conservative. When uncertain, leave the original text unchanged and insert no markers.
```

> **화자 마커 표기 결정(v3):** 프롬프트에서는 화자 번호를 매기지 않고 전환 지점에만 `[[SPEAKER]]`(경계 마커)를 삽입하게 한다. 번호(화자1/화자2)는 §11 Cue Splitter가 경계 순서대로 기계적으로 부여한다. LLM에게 번호 일관성까지 맡기면 부담이 커지고, 어차피 번호는 cue-local이라 경계 정보만 있으면 충분하기 때문이다.

### 8.2 Few-shot 예시

아래 예시를 System Prompt 뒤에 포함한다. 예시는 **실제 eval(`samples/language_hint_comparison.json`)에서 관측된 오류 패턴**을 기반으로 구성했으며, 언어별로 오류 성격이 다르다는 점, 그리고 화자 전환 마커 삽입/미삽입 기준을 함께 반영한다.

* **ja/zh**: 정규화 후 남는 오류는 대부분 고유명사 이표기(음차/한자 대체). → "문맥으로 정답 고유명사를 고르는" 유형.
* **ko**: 정규화해도 남는 오류가 표기 차이가 아니라 **사전에 존재하는 두 실제 단어 간 혼동**(몰아냈음↔보란했음, 무척↔부쩍, 족히↔조기). → 문법으로 판별 가능한 경우와, 문장 하나로는 판별 불가해 원문을 유지해야 하는 경우로 나뉜다.
* **화자 전환**: 질문-답변이 한 세그먼트에 뭉친 경우 마커 삽입, 한 사람의 연속 발화는 미삽입.

```text
Examples:

Example 1 (Japanese — proper noun corrected via document context, katakana transliteration error):
<CURRENT>
新しいベルサージのバッグを買いました。
</CURRENT>
<OUTPUT>
新しいヴェルサーチのバッグを買いました。
</OUTPUT>

Example 2 (Chinese — homophone-like proper noun corrected via context, hanzi substitution error):
<CURRENT>
第一站是克隆,然后去巴黎。
</CURRENT>
<OUTPUT>
第一站是科隆,然后去巴黎。
</OUTPUT>

Example 3 (Korean — grammatically implausible word replaced with the word that actually fits):
<CURRENT>
아무리 빨라도 3개월은 조기 걸린다고 하더라고요.
</CURRENT>
<OUTPUT>
아무리 빨라도 3개월은 족히 걸린다고 하더라고요.
</OUTPUT>

Example 4 (Korean — real word confused with another real word, resolved by sentence-level meaning):
<CURRENT>
저희가 경쟁사를 시장에서 보란했음을 강조했어요.
</CURRENT>
<OUTPUT>
저희가 경쟁사를 시장에서 몰아냈음을 강조했어요.
</OUTPUT>

Example 5 (Korean — two equally valid real words, context insufficient → leave unchanged):
<CURRENT>
네, 그런데 요즘 식욕이 무척 늘어서 걱정이에요.
</CURRENT>
<OUTPUT>
네, 그런데 요즘 식욕이 무척 늘어서 걱정이에요.
</OUTPUT>

Example 6 (Korean — a question and its answer merged into one segment → insert speaker-change marker):
<CURRENT>
이거 얼마예요 오천원입니다
</CURRENT>
<OUTPUT>
이거 얼마예요?[[SPEAKER]]오천원입니다.
</OUTPUT>

Example 7 (Korean — single speaker talking continuously, no speaker change → do NOT insert marker):
<CURRENT>
그래서 제가 어제 시장에 갔는데 사람이 정말 많더라고요 결국 아무것도 못 샀어요
</CURRENT>
<OUTPUT>
그래서 제가 어제 시장에 갔는데 사람이 정말 많더라고요. 결국 아무것도 못 샀어요.
</OUTPUT>

Do not copy these examples' content. They illustrate the correction and segmentation style only.
```

> **Example 5가 특히 중요하다.** 실제 eval에서 CER이 떨어지지 않은 케이스(무척/부쩍처럼 둘 다 정상 어휘라 판별 불가)와 같은 유형으로, 모델에게 "이 정도 애매함이면 건드리지 마라"는 기준선을 직접 보여준다. 예시가 전부 "교정하는" 사례뿐이면 모델이 과교정(over-correction)으로 편향된다.

> **Example 7도 같은 역할을 화자 분리 쪽에서 한다.** 마커 삽입 예시(6)만 있으면 모델이 한 사람의 긴 발화까지 과분할(over-split)하려 든다. "전환 없으면 마커 없음" 예시를 최소 1개 넣어 보수적 기준선을 잡는 것이 핵심이다. 미분할(놓침)이 과분할보다 안전한 실패 방향이다 — 안 쪼개면 최소 "기존과 동일"이지만, 잘못 쪼개면 자막이 어색해진다.

> 예시 언어·도메인은 실제 처리 콘텐츠 특성에 맞춰 교체하는 것을 권장한다(기술 용어가 많으면 기술 고유명사 예시로 대체 등).

### 8.3 실제 요청 시 Input 구조

prefix 캐시(§3)를 위해 **고정 블록(원문 전체)을 앞, 가변 블록(교정 대상)을 뒤**에 배치한다. `<FULL_TRANSCRIPT>` 블록은 요청마다 바이트 단위로 동일해야 한다.

```text
<FULL_TRANSCRIPT>
1 문장 텍스트 (raw)
2 문장 텍스트 (raw)
...
845 문장 텍스트 (raw)
</FULL_TRANSCRIPT>

<CURRENT>
101 문장 텍스트 (ASR 원문)
</CURRENT>
```

기대 출력:

```text
<OUTPUT>
101' 교정된 텍스트 (필요 시 [[SPEAKER]] 마커 포함)
</OUTPUT>
```

Stop Sequence로 `</OUTPUT>` 지정. 파싱은 `<OUTPUT>`~`</OUTPUT>` 사이 텍스트를 추출하고 앞뒤 공백을 제거한 뒤, `[[SPEAKER]]` 마커를 §11 Cue Splitter로 넘긴다.

---

## 9. 교정 단계의 장점

* 원문 전체 문맥 → 동음이의어·고유명사 판별 정확도 향상
* context가 raw → 오교정 전파 없음, 문장별 독립·병렬 처리 가능
* glossary 사전 구축 단계 불필요 → 파이프라인 단순화
* 실패 시 특정 문장만 재실행 가능 (원본 보존 전제, §5)
* 낮은 temp + 고정 출력 포맷으로 결과 안정성 확보

---

## 10. 화자 전환 탐지 (v3 신규)

### 10.1 목적과 범위

연속 대화(특히 짧은 질의-응답)가 VAD 단계에서 짧은 무음으로 인해 하나의 chunk로 병합되면(design.md §12.2), STT는 이를 한 덩어리 텍스트로 반환하고 **화자 경계 정보가 유실**된다. 그 결과 서로 다른 사람의 대사가 한 자막 화면에 통으로 표시된다. 이를 완화하는 것이 목적이다.

**명확한 범위 한정:**

* **하는 것**: 텍스트 문맥으로 "여기서 화자가 바뀐다"는 **전환 지점**만 탐지(`[[SPEAKER]]` 마커).
* **하지 않는 것**:
  * 오디오 기반 diarization (원칙 4 위배, 화자 신원 추적 신뢰도 확보 불가)
  * 화자 신원의 문서 전역 추적 (누가 화자A인지 문서 내내 일관 추적하지 않음)
  * 성별 판정 / 개인 프로필 (텍스트 기반 성별 추정은 편향·오류 위험이 커 하지 않음)
  * 라벨은 **각 cue 내부에서만 유효한 상대 순번**이며, 다음 cue에서 리셋되어도 무방

즉 목표는 "누가 말했는지 표시"가 아니라 **"뭉친 대사를 발화 단위로 쪼개 한 화면 통짜 표시를 막는 것"**이다.

### 10.2 처리 방식

§6의 full-context 교정 호출 안에서 함께 처리한다(별도 LLM 호출을 만들지 않는다 — 입력이 동일하므로 호출 2배는 낭비).

* 교정 대상 세그먼트 길이가 임계값(초기 **100자**, 튜닝 대상)을 넘고, 서로 다른 화자의 발화가 섞였다고 판단되면 전환 지점에 `[[SPEAKER]]` 삽입.
* 임계값 이하이거나 한 사람의 연속 발화면 마커를 넣지 않는다.
* 교정과 분할을 한 프롬프트에 합치는 것이 품질을 해치는지는 실측 확인 대상. 저하가 관측되면 분할 탐지만 별도 호출로 분리한다(초기엔 합쳐서 시도).

### 10.3 알려진 실패 모드

* **과분할(false split)**: 한 사람 발화를 여러 명처럼 쪼갬. → 프롬프트/few-shot을 보수적으로(Example 7) 잡아 빈도를 낮춘다.
* **미분할(false negative)**: 실제 전환을 놓침. → 더 안전한 실패 방향(결과가 "기존과 동일"). 보수적 설계가 이 방향으로 편향되게 한다.
* **마커 오삽입 전파**: full-context는 context가 raw라 §6대로 전파가 없으나, 마커는 교정 출력에만 나타나므로 애초에 context를 오염시키지 않는다. 전파 위험 낮음.

---

## 11. Cue Splitter (v3 신규, 결정론적 — LLM 미사용)

교정·마커 삽입이 끝난 문장을 실제 SRT cue로 변환하는 후단 유틸리티. **LLM을 쓰지 않는다**(요약/의역 위험 원천 차단 — design.md가 명시적으로 금지). 순수 규칙 기반으로 "같은 텍스트를 여러 cue로 나누고 시간을 배분"만 한다.

### 11.1 처리 순서

```
입력: 교정된 문장 텍스트(+[[SPEAKER]] 마커), 해당 chunk의 start/end 시각
   ↓
1차 분할 (화자 경계):
   [[SPEAKER]] 마커 위치에서 텍스트를 분리.
   각 조각에 순번(화자1, 화자2, ...)을 cue-local하게 부여(선택 — 화면 노출 여부는 §11.3).
   ↓
2차 분할 (길이/CPS 초과):
   각 조각의 CPS(초당 문자수) = 조각 글자수 / 배분된 조각 시간 을 계산.
   CPS가 임계값을 초과하면, 조각 안에서 문장부호(. , ? ! …) 또는
   어절 경계(공백) 기준으로 추가 분할. 의미 변경 없음(순수 자르기).
   ↓
3차 시간 배분 (근사):
   chunk duration을 최종 조각들의 "글자 수 비율"로 나눠 각 cue에 timestamp 부여.
   (균등 발화 속도 가정 — 부정확하지만 word-level timestamp 없이는 최선)
   ↓
출력: N개의 SRT cue
```

### 11.2 CPS 임계값

* 한국어/CJK 자막 통상 권장 범위는 대략 **초당 12~17자** 수준이나, 정확한 채택값은 구현 단계에서 자막 제작 관행 소스를 확인해 확정한다(여기서는 원칙만 세운다).
* 최대 노출 시간(한 cue가 화면에 머무는 상한)과 최소 노출 시간(너무 짧게 깜빡이지 않도록 하한)도 함께 둔다.
* 1차 분할(화자)과 2차 분할(CPS)은 독립적으로 적용된다. 화자 전환이 없어도 CPS만 초과하면 2차 분할이 걸린다.

### 11.3 화자 라벨 화면 노출 (기본: 노출 안 함)

* **기본값: 라벨을 SRT 텍스트에 넣지 않는다.** 마커는 분할 기준으로만 쓰고, 화면에는 순수 대사 텍스트만 표시한다.
* 이유: 원래 목적이 "누가 말했는지 표시"가 아니라 "가독성 좋게 짧게 쪼개기"이고, 라벨을 텍스트에 섞으면 이후 자막 활용(번역·검색 등)에 노이즈가 된다.
* 필요 시 옵션으로 "화자1:" 프리픽스 노출을 켤 수 있게 두되(§7 향후), 상대 순번임을 감안한다.

### 11.4 공통 한계 (문서에 명시)

* **시간 배분은 근사치다.** word-level alignment 없이 글자 수 비례로 나누므로, 실제 발화 속도와 다르면 자막-음성 싱크가 다소 어긋난다. 이 기능의 목적은 "정밀 동기화"가 아니라 "한 화면 통짜 표시 방지"임을 분명히 한다. 정밀화는 §7(word-level timestamp)로 이관.

---

## 12. 설정 항목 (config.json)

### 12.1 위치 — 통합이냐 분할이냐

Phase C 설정은 **별도 파일로 분할하지 않고 기존 `config.json`에 통합**한다(`text_enhancement.text_correction` 하위, design.md §5B.5와 동일 위치).

근거:

* design.md의 `config.json`은 이미 `local_api` / `gemini` / `llm` / `prompt`처럼 서로 다른 서브시스템을 최상위 섹션으로 중첩 관리하는 패턴을 쓰고 있고, `text_enhancement.text_correction`도 원래 이 구조의 일부로 자리 잡아 있었다. Phase C 설정이 늘었다고 이 패턴을 깰 이유가 약하다.
* 설정이 한 곳에 있어야 GUI(design.md §8 설정 창)가 config를 읽고 쓰는 로직이 단순해진다.
* **분할을 재검토할 조건**(지금은 해당 없음): GUI 없이 Phase C 파라미터만 독립적으로 자주 튜닝/실험해야 하는 워크플로우가 생기는 경우, 또는 비개발자가 config.json을 직접 열어보는 빈도가 높아 가독성이 중요해지는 경우. 필요해지면 그때 분리한다(Ship simpler first).

> **네임스페이스 공유 주의 (design.md v3.2 반영):** design.md §21에 `text_enhancement.dedup_repeated_chunks`, `text_enhancement.strip_infinite_repetition`라는 **Phase B(청크 단위 STT) 할루시네이션 필터**가 opt-in으로 있다 (구현 완료 — `gui/worker.py`). 이 둘은 §12.2의 `text_correction`(Phase C)과 **같은 `text_enhancement` 최상위 키를 형제로 공유**하지만 역할은 완전히 다르다 — dedup/strip 필터는 Phase B 도중 "직전 chunk 반복 출력"을 감지해 SRT에서 제외하는 것이고, `text_correction`은 Phase B 종료 후 별도 LLM으로 문맥 교정을 하는 것이다. 구현 시 이 둘을 같은 단계 로직으로 혼동하지 않도록 주의. (v3.1 시점에 지적했던 "design.md §9 config 예시에 두 필터 미반영" 문제는 design.md v3.2에서 해소됐고, design.md §21이 역으로 이 절을 참조하는 상호 링크도 정리됐다.)

### 12.2 스키마 (v3, 기본값 포함)

design.md §5B.5의 v2 스키마(`window_chunks: 5` 등, Sliding Window 전제)는 v3 방식과 맞지 않아 아래로 대체한다.

```json
"text_enhancement": {
  "custom_vocabulary": [],
  "text_correction": {
    "enabled": false,

    "server": {
      "url": "http://localhost:8081/v1/chat/completions",
      "launch_mode": "external",
      "server_binary": "",
      "model_path": "",
      "port": 8081,
      "extra_args": "--ctx-size 32768 --parallel 1 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --reasoning-budget 0 --jinja",
      "startup_timeout_sec": 120
    },

    "sampling": {
      "temperature": 0.25,
      "top_p": 0.8,
      "top_k": 20,
      "presence_penalty": 1.0,
      "repetition_penalty": 1.0,
      "max_tokens": 512
    },

    "full_context": {
      "max_segment_chars": 60000,
      "segment_split_count": 3,
      "glossary_assist": false
    },

    "speaker_detection": {
      "enabled": true,
      "trigger_length_chars": 100
    },

    "cue_splitter": {
      "cps_threshold": 15,
      "max_cue_duration_sec": null,
      "min_cue_duration_sec": null,
      "show_speaker_label": false
    }
  }
}
```

### 12.3 항목별 설명 및 기본값 근거

**`server` — Phase C 텍스트 모델 서빙**

| 필드 | 기본값 | 설명 |
|---|---|---|
| `url` | `http://localhost:8081/v1/chat/completions` | STT 서버(design.md `local_api`, **기본 8080** — design.md v3.2에서 확정)와 포트 충돌 방지 위해 8081 사용. Phase B/C가 시간적으로 분리 구동되므로(원칙 2) 동시 점유는 없으나, 포트 필드 자체는 구분해 혼동 방지 |
| `launch_mode` | `"external"` | design.md `local_api.launch_mode`와 동일 패턴(§6.3). **Managed 모드의 생명주기 로직은 이미 `pipeline/server_manager.py`의 `ensure_llama_server()`로 구현·실측 검증되어 있으므로**(design.md v3.2 §6.3), Phase C 구현 시 이 구현체를 그대로 재사용한다 — 신규 작성 불필요 |
| `server_binary` / `model_path` | `""` | 사용자 입력 필요. `launch_mode`와 무관하게 상시 저장(design.md §9와 동일 원칙 — 모드 전환 시 재입력 방지) |
| `port` | `8081` | Managed 모드 전용, External 시 무시 |
| `extra_args` | §3 검증 커맨드 문자열 | Q6_K·32K ctx·thinking 억제 등 §3에서 확정한 구동 옵션의 기본 반영. 모델 파일명 자체는 `model_path`에서 결정되므로 `-m` 제외 |
| `startup_timeout_sec` | `120` | design.md `local_api.managed.startup_timeout_sec`와 동일값 |

**`sampling` — §3 "샘플링 파라미터" 표와 동일**

| 필드 | 기본값 | 근거 |
|---|---|---|
| `temperature` | `0.25` | §3: 모델 공식 권장(0.7)보다 결정성 위해 하향, 0.15는 공격적이라 상향 조정한 중간값. **실측 튜닝 대상** |
| `top_p` | `0.8` | 모델 공식 권장값 유지 |
| `top_k` | `20` | 모델 공식 권장값 유지 |
| `presence_penalty` | `1.0` | 반복 억제. 0.5~1.5 범위 중간값에서 시작, 언어 혼합 부작용 관측 시 하향 |
| `repetition_penalty` | `1.0` | 공식값 유지(사실상 비활성) |
| `max_tokens` | `512` | §3: 현재 문장 하나(+마커) 교정용, 256~512 범위 상단 채택(화자 마커 삽입으로 출력이 약간 늘 수 있어 여유) |

**`full_context` — §6 Full-context 교정**

| 필드 | 기본값 | 근거 |
|---|---|---|
| `max_segment_chars` | `60000` | §3 Context 크기: 32768 토큰 ≈ 5~6만 자 추정치의 하한. 이 값을 넘으면 문서를 대구간으로 분할 |
| `segment_split_count` | `3` | §6 "긴 파일 처리": 초장편을 2~3개 대구간으로 분할한다는 원칙의 상한값을 기본으로 채택 |
| `glossary_assist` | `false` | §7: 표기 흔들림이 실측에서 관측되기 전까지는 비활성(Ship simpler first). 켜면 보조 glossary pass 추가 |

**`speaker_detection` — §10 화자 전환 탐지**

| 필드 | 기본값 | 근거 |
|---|---|---|
| `enabled` | `true` | 이번 설계 세션에서 신설된 기능. 다만 **원맨쇼 강의 등 화자 전환이 거의 없는 콘텐츠에서는 끄는 것이 안전**하므로 최상위 토글로 노출(이전엔 이 토글 자체가 문서에 없었던 문제를 이번에 보완) |
| `trigger_length_chars` | `100` | §10.2: "100자(튜닝 대상)" 그대로 기본값화 |

**`cue_splitter` — §11 Cue Splitter**

| 필드 | 기본값 | 근거 |
|---|---|---|
| `cps_threshold` | `15` | §11.2: 한국어/CJK 통상 권장 범위(12~17자/초)의 중간값. **정확한 채택값은 구현 단계 확정 대상**(§24 미결정 사항 참고) |
| `max_cue_duration_sec` | `null` | §11.2에서 "둔다"고만 하고 값 미정 상태였던 항목. **구현 전 확정 필요** — null은 "미설정 시 무제한"으로 해석 |
| `min_cue_duration_sec` | `null` | 위와 동일. 너무 짧은 cue 깜빡임 방지용이나 값 미정 |
| `show_speaker_label` | `false` | §11.3: 기본 미노출 원칙 그대로 |

### 12.4 구현 전 확정 필요 (design.md §24에 추가할 항목)

* `cue_splitter.cps_threshold`의 정확한 채택값 (12~17자/초 범위 내 확정)
* `cue_splitter.max_cue_duration_sec` / `min_cue_duration_sec`의 구체적 초 단위 값
* `full_context.max_segment_chars`의 실측 기반 보정 (토크나이저 실측치로 5~6만 자 추정을 검증)
* `speaker_detection.trigger_length_chars`(100자)의 실측 튜닝
* `sampling.temperature`(0.25) 등 샘플링 값 전반의 실측 튜닝 (§3)

---

## 13. 실측 결과 (2026-07-12)

### 13.1 환경

* GPU: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
* llama.cpp: `b9914 (931ca30be)`, Clang 20.1.8, Windows x86_64 (당시 최신 릴리스 `b9917` 대비 3빌드 차이 — 사실상 최신)
* 모델: `unsloth/Qwen3.5-9B-GGUF` — Q6_K, Q8_0 두 양자화 모두 실측

### 13.2 다운로드

```powershell
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q6_K.gguf
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf
```

`--local-dir` 미지정 → 기본 HF 캐시(`~/.cache/huggingface/hub`, Windows: `C:\Users\<user>\.cache\huggingface\hub`)에 저장됨. 실제 경로는 `hf cache scan`으로 확인.

### 13.3 실행 커맨드 (최종 확정)

```powershell
& "C:\ai\llama\llama-server.exe" `
  -m "<...>\Qwen3.5-9B-Q8_0.gguf" `
  --host 0.0.0.0 --port 8081 --ctx-size 32768 --parallel 1 `
  --n-gpu-layers 999 --threads 12 -fa on `
  --cache-type-k q8_0 --cache-type-v q8_0 `
  --reasoning-budget 0 --jinja --metrics
```

### 13.4 §3 "실측 확인 항목" 결과

1. **아키텍처 추론 정상성** — 정상 확인. (초기에 오탐 있었음 — §13.5 참고)
2. **thinking 억제** — `--reasoning-budget 0` + 요청별 `chat_template_kwargs: {"enable_thinking": false}` 조합으로 정상 억제 확인. `/v1/chat/completions` 응답에 `<think>` 블록 없음.
3. **vision encoder VRAM 점유** — 세 조합 모두 16GB 중 여유 있게 들어감(총점유량, 텍스트 전용 로드 옵션 확인은 안 됐으나 실사용상 문제 없는 수준):
   | 가중치 | KV캐시 타입 | VRAM 사용 |
   |---|---|---|
   | Q6_K | q8_0 | 9.1GB |
   | Q8_0 | bf16 | 11.0GB |
   | Q8_0 | q8_0 | 10.6GB |
4. **prefix 캐시 실동작** — 미검증. 실제 파이프라인(원문 전체 fixed prefix + 문장별 요청) 구현 시점에 확인 필요.
5. **RTX 50 CUDA gibberish 이슈** — 해당 없음으로 판명(§13.5). RTX 5070 Ti + 이 빌드 조합에서는 문제 재현 안 됨.

### 13.5 트러블슈팅: PowerShell 한글 인코딩 버그 (모델/GPU 오탐 주의)

초기 테스트에서 `"안녕하세요. 짧게 한 문장으로 답해주세요."`라는 질문에 전혀 무관한 영어 거절 응답("I cannot provide information related to illegal...")이 반복 관측됐다. Gated DeltaNet 신규 연산자 오류·CUDA 13.2 gibberish 이슈(RTX 50 계열)를 의심해 KV캐시 타입을 bf16으로 바꿔봤으나 재현. `/completion` raw 엔드포인트로 실제 전달된 prompt 문자열을 확인한 결과, **한글이 전부 `?`로 깨진 채 모델에 들어가고 있었다.**

원인은 `Invoke-RestMethod -Body $body`(PowerShell 문자열)가 UTF-8이 아닌 인코딩으로 전송된 것. 아래처럼 바이트로 명시 변환하자 정상화됨:

```powershell
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes
```

**결론: 모델·llama.cpp 빌드·GPU는 처음부터 문제가 없었다.** 유사한 "동문서답" 증상이 관측되면 아키텍처/드라이버를 의심하기 전에 요청 body 인코딩부터 확인할 것(특히 PowerShell 5.1의 `Invoke-RestMethod` + 비ASCII 문자 조합).

### 13.6 최종 권장 설정 (실측 반영)

Q8_0 + q8_0 KV캐시로 확정(VRAM 10.6GB/16GB, 여유 5.7GB). VRAM 여유가 충분히 확인됐으므로, §3 "권장 설정 요약" 표의 기본값을 Q6_K에서 **Q8_0로 상향해도 무방**하다(품질 우선 시). 32K 컨텍스트를 실제 자막으로 꽉 채웠을 때의 여유는 미검증(§13.4 항목 4와 함께 파이프라인 구현 시점에 재확인).

---

## 부록 A. 변경 이력

### v2 → v3

| # | 변경 | 근거 |
|---|---|---|
| 1 | **Sliding Window → Full-context 교정으로 전환** (§0, §6) | 자막 텍스트가 작아(수십 KB~100KB) 원문 전체를 context에 담을 수 있음. 동음이의어 원거리 문맥 확보 + 오교정 전파 소멸 + glossary 단계 불필요를 동시 달성 |
| 2 | Context 크기 8192 → 32768 권장 (§3) | full-context 수용. Phase C는 GPU 단독 사용이라 audio chunk용 보수적 제약 불필요 |
| 3 | Prefix 캐시 활용 명시 및 실측 필수 항목화 (§3) | 원문 전체가 고정 prefix로 반복되므로 KV 캐시 재사용이 성능에 결정적 |
| 4 | glossary 사전 구축 단계 제거 (§4, §7) | full-context가 전역 표기 참조를 흡수. glossary는 "표기 흔들림 관측 시 보조"로 격하 |
| 5 | 오교정 전파 실패 모드 "해결됨"으로 전환 (§6) | context가 raw → 전파될 교정 이력 자체가 없음. v2 부록B "원문 window"를 문서 전체로 확장 적용한 셈 |
| 6 | 문장별 병렬 교정 가능성 명시 (§6, §7) | 각 문장이 고정 원문 context에만 의존, 이전 교정 결과 비의존 |
| 7 | **화자 전환 탐지 신설** (§10), 교정 프롬프트에 `[[SPEAKER]]` 마커 규칙·few-shot 추가 (§8) | 뭉친 대사의 한 화면 통짜 표시 방지. 오디오 diarization·성별·전역 신원추적은 범위 제외 |
| 8 | **Cue Splitter 신설** (§11, 결정론적) | 화자 마커 분할 + CPS 초과 분할 + 시간 근사 배분. LLM 미사용으로 요약/의역 위험 차단 |
| 9 | Word-level timestamp 확보를 향후 개선에 등재 (§7, §11.4) | 시간 근사 배분의 부정확성을 근본 개선하는 경로 |
| 10 | 목적/저장에 "자막 구조화" 및 raw.srt 보존 반영 (§1, §5) | 교정(텍스트)과 구조화(표시단위) 두 축을 명시, 롤백 가능성 확보 |
| 11 | 후처리 모델을 `unsloth/Qwen3.5-9B-GGUF`로 구체화, 검증 대상·실행 커맨드·실측 항목 명시 (§2, §3) | 모델 카드/검색 확인 결과: 멀티모달(Gated DeltaNet+MoE+VL) 아키텍처라 최신 llama.cpp 필수, vision encoder VRAM 실측 필요. thinking 기본 → `--reasoning-budget 0` + 검증 |
| 12 | 양자화 Q8_0 → Q6_K 하향 (§3) | full-context(32K) KV 캐시 고려 시 16GB에서 Q8_0(9.53GB)은 빠듯. Q6_K(7.46GB)로 여유 확보, 부족 시 Q4_K_XL |
| 13 | 샘플링 파라미터를 모델 공식 non-thinking 권장(temp 0.7 등) 대비 재조정 (§3) | v2 temp=0.15는 공식 권장 대비 공격적. 0.2~0.3 시작 + presence_penalty 반영, 실측 튜닝 |
| 14 | **설정 항목(configurable property) 전체 정리 및 config.json 통합 스키마 신설** (§12) | Phase C 파라미터가 여러 절에 흩어져 있어 한곳에 정리 필요. 별도 파일 분할 대신 기존 config.json 패턴에 통합 결정. `speaker_detection.enabled` 등 이전에 누락됐던 토글 보완 |
| 15 | design.md §21 신설 Phase B 할루시네이션 필터(`dedup_repeated_chunks`, `strip_infinite_repetition`)와 `text_enhancement` 네임스페이스 공유 사실 명시 (§12.1) | design.md v3.1 갱신분 반영. Phase B 필터와 Phase C `text_correction`은 같은 상위 키를 공유하나 역할이 다름을 명확화, 혼동 방지 |
| 16 | design.md v3.2(코드 동기화판) 반영 (§3, §12.1, §12.3) — Phase C 서버 포트를 예시 커맨드까지 8081로 통일(STT 실제 기본 8080과 충돌 방지), §12.1의 "design.md §9 필터 미반영" 지적을 해소됨으로 갱신, Managed 생명주기는 기구현된 `server_manager.ensure_llama_server()` 재사용으로 명시 | design.md v3.2에서 STT 포트 8088→8080 확정, config 스키마 정리, server_manager 구현·실측 검증 완료 |

### v1 → v2 (요약, 참고 보존)

| # | 변경 | 근거 |
|---|---|---|
| 1 | Thinking 모드 비활성화를 필수 요건으로 명시 | Qwen3.5/3.6 기본 thinking + n-predict 제한 시 출력 비는 사고 |
| 2 | Temperature 0.0 → 0.1~0.2 | 해당 계열 temp=0에서 반복·붕괴 경향 |
| 3 | Top-p 0.9→0.8, Top-k 20 추가 | non-thinking 권장 샘플링값 |
| 4 | llama.cpp 플래그 정정, n-predict 요청별 이동 | 실제 플래그명/운용 반영 |
| 5 | CUDA 13.2 gibberish 주의 추가 | 알려진 이슈 |
| 6 | 원본·교정본 이중 저장 | 사후 재교정 대비 |
| 7 | (v2) Sliding Window 오교정 전파 실패 모드 명시 | → v3에서 방식 전환으로 해소 |
| 8 | Input/Output XML 태그 구획 도입 | 규칙 안정화, 파싱 용이 |
| 9 | Few-shot 예시 추가 | 과교정 경계 명확화 |
| 10 | Glossary 구축 시점 명확화 | → v3에서 단계 제거 |

## 부록 B. v2 오교정 전파 완화 옵션 (역사적 기록)

v2에서 검토했던 완화 옵션들. **v3는 full-context 전환(§6)으로 전파 문제 자체가 사라졌으므로, 아래는 역사적 맥락으로만 보존한다.** v3의 방식은 사실상 아래 "원문 window"를 문서 전체 규모로 극대화한 것에 해당하며, 표기 통일이 필요하면 "Glossary 고정 주입"만 §7 보조 장치로 선택적으로 되살릴 수 있다.

| 옵션 | 방식 | v3에서의 위치 |
|---|---|---|
| 원문 window | context에 교정본 대신 원본 ASR을 넣음 | **v3 기본 방식으로 채택·확장** (문서 전체) |
| Glossary 고정 주입 | 전사 스캔으로 정규표기 확정 후 주입 | §7 선택적 보조 (표기 흔들림 관측 시) |
| 원문/교정본 병기 | context에 원문+교정본 함께 제공 | 불필요 (context가 raw라 전파 없음) |
| Two-pass | 1차 교정 → glossary 확정 → 2차 재교정 | 불필요 (full-context 1-pass로 대체) |
