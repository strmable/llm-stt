# ASR 후처리 설계 v2

문서 버전: 2.0
이전 버전 대비 주요 변경: [부록 A](#부록-a-v1--v2-변경-이력) 참고

---

## 1. 목적

Qwen3-ASR로 생성된 전사 결과의 품질을 향상시키기 위한 후처리 단계이다.

후처리는 다음과 같은 오류를 **최소한의 수정만으로** 교정한다.

* 문맥에 맞지 않는 동음이의어
* 잘못 인식된 고유명사 (음차·표기 오류)
* 중복 단어 (예: `主 主`, `주 주요`)
* 조사 및 띄어쓰기 오류
* 명백한 오탈자

반대로 다음 작업은 **수행하지 않는다.**

* 문장 재작성
* 의역
* 내용 추가
* 요약
* 문체 변경

즉, **ASR 결과를 최대한 보존하면서 필요한 부분만 수정하는 것**이 목표이며, 불확실한 경우 원문을 유지하는 것을 기본 원칙으로 한다.

---

## 2. 모델 선정

후처리 모델은 **Qwen3.5-9B**를 기본으로 사용한다. (대안: Qwen3.6-27B — VRAM 여유가 있고 품질을 우선할 경우)

### 선정 이유

* Instruction 준수 능력이 우수함
* JSON/고정 포맷 출력 안정성이 높음
* 응답 속도가 빠름 (문장 단위 다수 호출에 유리)
* llama.cpp에서 안정적으로 동작
* 16GB VRAM 환경에서 Q8_0(8bit) 운용 가능

ASR 후처리는 복잡한 추론보다 **원문 보존 · 최소 수정 · 높은 Instruction 준수**가 중요하다. 따라서 대형 추론 특화 모델보다 9B급 instruct 모델이 비용/속도 측면에서 적합하다.

### ⚠️ Thinking 모드 처리 (중요)

Qwen3.5 / Qwen3.6 계열은 **기본적으로 thinking 모드로 동작**하며, `<think>...</think>` 형태의 추론 블록을 먼저 생성한 뒤 최종 응답을 출력한다. 본 후처리 용도에서는 이 동작이 문제가 된다.

* 후처리는 "최소 수정·과도한 reasoning 배제"가 목표인데, thinking은 이와 정면으로 배치된다.
* `n-predict`를 256~512로 제한한 상태에서 thinking이 켜져 있으면, **thinking 블록만 채우다 출력 토큰이 소진되어 실제 교정 결과가 비는 사고**가 발생한다.

따라서 **non-thinking(instruct) 모드를 명시적으로 활성화**해야 한다.

* llama.cpp: 채팅 템플릿에 `enable_thinking=false`를 전달하거나, 요청에 `chat_template_kwargs: {"enable_thinking": false}`를 포함한다.
* 모델별 non-thinking 비활성화 방법은 해당 모델 카드의 안내를 따른다.

### 샘플링 파라미터

Qwen3.5/3.6 계열 **non-thinking 모드 권장값**을 기준으로 하되, 후처리 목적에 맞게 결정성을 높인다.

| 항목 | 권장값 | 비고 |
|---|---|---|
| Temperature | 0.1 ~ 0.2 | **0.0 비권장** — 이 계열은 temp=0에서 반복·붕괴가 나타나는 경향이 있음 |
| Top-p | 0.8 | 모델 권장값 |
| Top-k | 20 | 모델 권장값 |
| Repeat penalty | 1.0 | penalty 사실상 비활성. 반복이 심하면 presence_penalty 0.5~1.5 검토 |

> 원 설계는 "재현성 확보"를 위해 temp=0을 제시했으나, Qwen3.5/3.6에서 temp=0은 오히려 결과 안정성을 해칠 수 있다. 낮은 temp(0.1~0.2) + 낮은 top_p 조합이 재현성과 안정성의 실용적 절충점이다.

---

## 3. llama.cpp 실행

### 실행 커맨드 (예시)

```bash
llama-server \
  -m qwen3.5-9b-q8_0.gguf \
  --ctx-size 8192 \
  --n-gpu-layers 999 \
  --temp 0.15 \
  --top-p 0.8 \
  --top-k 20 \
  --repeat-penalty 1.0 \
  --jinja \
  --host 0.0.0.0 \
  --port 8080
```

> 참고: 출력 토큰 상한(`n_predict`)은 서버 기동 시 고정하기보다 **요청별 파라미터**로 넘기는 것을 권장한다. GPU offload 플래그는 `--n-gpu-layers`(축약 `-ngl`)이다.

### 권장 설정 요약

| 항목 | 권장값 |
|---|---|
| 양자화 | Q8_0 (8bit) |
| Context | 8192 |
| Max Predict (요청별) | 256 ~ 512 |
| Temperature | 0.1 ~ 0.2 |
| Top-p | 0.8 |
| Top-k | 20 |
| Repeat penalty | 1.0 |
| GPU Layers | 전체 Offload |
| Thinking | **비활성화 (필수)** |

### Context 크기

16GB VRAM 기준 8192가 무난하다. 후처리는 긴 전사를 한 번에 처리하지 않고 Sliding Window로 수행하므로 큰 context가 필요 없다.

### 출력 토큰 제한

후처리는 현재 문장 하나만 교정하므로 긴 출력이 필요 없다. `n_predict` 256~512 권장. 출력 제한의 이점은 다음과 같다.

* 이전 문장을 다시 출력하는 실수 방지
* 장황한 설명·불필요한 reasoning 출력 방지
* 처리 시간 단축, GPU 메모리 절감
* 동일 입력에 대한 결과 안정화

**단, thinking을 비활성화한 상태에서만 이 토큰 예산이 유효하다** (§2 참고).

### Stop Sequence

출력 형식을 `<OUTPUT>...</OUTPUT>`으로 고정하고 `</OUTPUT>`을 Stop Sequence로 지정한다. 모델이 교정 문장을 출력한 뒤 추가 설명이나 다음 문장을 생성하는 것을 방지한다.

### CUDA 빌드 주의 (RTX 50 / Blackwell)

* Qwen3.5/3.6을 llama.cpp에서 구동할 때 **CUDA 13.2 빌드는 gibberish 출력 이슈**가 보고되었다. CUDA 13.2 미만 또는 13.3을 사용한다.
* 출력이 gibberish이면 context length가 너무 낮게 잡혔거나, `--cache-type-k bf16 --cache-type-v bf16` 시도로 완화되는 경우가 있다.

---

## 4. 후처리 방식 (ASR과 분리)

ASR을 **전체 완료한 후** 후처리를 수행한다.

```
Audio
  ↓
ASR (전체 전사 완료)      ← 원본 전사 전량 확보
  ↓
[선택] 원본 전사 스캔 → 고유명사 정규화 테이블(glossary) 구축   (§7, 향후 개선)
  ↓
Sliding Window 후처리 (문장별 교정)
```

ASR 진행 중에는 LLM을 호출하지 않는다. ASR과 후처리를 분리하면 구현이 단순해지고, 후처리 실패 시 전사를 다시 할 필요가 없다.

---

## 5. 전사 저장

전사 완료 후 문장 단위로 분리 저장한다. **원본과 교정본을 모두 보존한다.**

```
raw/000001.txt      fixed/000001.txt
raw/000002.txt      fixed/000002.txt
...                 ...
raw/000845.txt      fixed/000845.txt
```

* 문장 번호는 후처리 시 순서 유지에 사용한다.
* **원본을 별도 보존하는 이유**: 오교정을 나중에 발견해도 특정 문장만 원본에서 재교정할 수 있고, 향후 glossary 방식으로 업그레이드할 때 원본 전사가 그대로 재료로 남는다. 추가 비용은 디스크 수 KB 수준이며, "실패 시 특정 문장만 재실행" 장점과 맞물린다.

---

## 6. Sliding Window 교정

후처리는 **현재 문장 하나만** 수정하며, 최근 교정 완료된 문장들을 context로 함께 제공한다.

Window 크기가 10일 때:

```
[이전 교정 완료본 — context]
92' 93' 94' 95' 96' 97' 98' 99' 100'
[현재 ASR 원문 — 교정 대상]
101
      ↓
[출력]
101'
```

다음 단계는 window가 한 칸 이동한다.

```
[context] 93' 94' ... 100' 101'
[현재]    102
      ↓
[출력]    102'
```

즉,

* 이전 문장은 모두 **교정 완료본**
* 현재 문장만 수정
* 출력도 현재 문장 하나

교정 완료 문장이 다음 문장의 문맥으로 이어져, 고유명사 표기·용어 일관성이 점진적으로 향상되는 효과를 노린다.

### Sliding Window를 쓰는 이유

수백 문장 이전 내용이 현재 문장의 동음이의어 선택에 영향을 줄 가능성은 낮다. 반면 최근 문맥은 매우 중요하다.

```
A: 이번에 베르사체 신상 봤어?
B: 응.
[ASR] 벨사지 예쁘더라.
  → 최근 문맥만으로도  벨사지 → 베르사체  교정 가능
```

긴 context 전체보다 최근 문장만 유지하는 편이 효율적이다. Window 크기는 초기 **10문장**으로 시작하고 테스트 결과에 따라 조정한다.

### ⚠️ 알려진 실패 모드: 오교정 전파 (Error Propagation)

교정 완료본을 다음 window의 context로 재공급하는 구조에는 **한 번 잘못 교정된 표기가 이후 문장에 "확정된 문맥"으로 전파**될 수 있는 약점이 있다. 예를 들어 어떤 고유명사가 첫 등장에서 오교정되면, window가 그것을 정답으로 간주해 뒤 문장에도 같은 오답을 유도할 수 있다.

**초기 버전에서는 이 실패 모드를 인지한 상태로 그대로 진행한다.** 실사용상 순효과는 대체로 플러스이며, 이유는 다음과 같다.

* 프롬프트가 보수적("불확실하면 원문 유지")이라 오교정 발생 빈도 자체가 낮다.
* 잘못된 교정본은 window 크기(10문장)만큼만 유지되고 이후 밖으로 밀려나므로, 영향이 **국소적**이다. 전체 문서에 영구 고착되지 않는다.
* feedback이 오답을 전파하는 경우보다, 옳은 표기를 전파해 일관성을 높이는 경우가 훨씬 많다.

전파를 근본적으로 차단하려는 개선안(glossary, 원문 병기, feedback 제거 등)은 §7 및 [부록 B](#부록-b-오교정-전파-완화-옵션-향후)에 정리한다.

---

## 7. 향후 개선

* **고유명사 일관성 (glossary)**: 전사 **완료 후, 후처리 전** 단계에서 원본 전사 전체를 스캔해 고유명사 정규 표기 테이블을 구축하고, 이를 모든 문장 교정 시 고정 주입한다. 이렇게 하면 sliding window를 타지 않고 전역적으로 표기를 확정할 수 있어 오교정 전파를 차단한다.
  * 주의: 원본 전사 자체가 ASR 오류를 포함하므로, **단순 빈도 집계는 위험**하다(오답이 다수결로 채택될 수 있음). LLM에게 전사 전체/큰 청크를 주고 "변이형을 한 엔티티로 묶어 정규 표기 선정"을 시키는 방식이 안전하며, 그래도 애매한 고유명사는 강제 교정하지 않는다.
* Window 크기 자동 조절
* 교정 신뢰도 출력 / 수정 Diff 출력
* 사람이 검토할 문장 자동 표시
* 교정 결과 캐시
* 병렬 처리: feedback loop를 제거하고 glossary로 대체하면(부록 B) 문장별 병렬 교정이 가능해진다.

초기 버전은 단순 Sliding Window만으로도 대부분의 ASR 오류를 안정적으로 교정할 수 있을 것으로 예상한다.

---

## 8. Prompt

### 8.1 System Prompt

보수적으로 작성한다. **Input/Output을 XML 태그로 명시적으로 구획**하여, "이전 context는 수정 금지" · "현재 문장만 출력" 규칙이 안정적으로 지켜지도록 한다.

```text
You are an Automatic Speech Recognition (ASR) post-processing assistant.

Your task is to recover the speaker's intended utterance from an ASR transcript.
You will be given previous, already-corrected context inside <PREVIOUS_CONTEXT>
tags and the current ASR segment to correct inside <CURRENT> tags.

The current segment may contain recognition errors such as:
- homophone substitutions
- incorrect proper nouns (transliteration / character errors)
- duplicated words or phrases
- obvious character mistakes
- minor punctuation or spacing errors

Rules:
1. Correct only clear ASR recognition errors in the <CURRENT> segment.
2. Use <PREVIOUS_CONTEXT> only to disambiguate homophones and proper nouns.
3. Preserve the speaker's intended message, style, and tone.
4. Do NOT rewrite, summarize, paraphrase, or improve the sentence.
5. Do NOT add information that was not spoken.
6. If multiple valid interpretations exist and context cannot decide, keep the original text unchanged.
7. NEVER modify or output the previous context. It is for understanding only.
8. Correct ONLY the current segment.
9. Output ONLY the corrected current segment wrapped in <OUTPUT> tags. No explanations, no comments, no markdown, no quotation marks, no numbering.

The transcript may be in Japanese, Chinese, Korean, or a mixture of these.
Apply each language's normal writing conventions only when needed to fix an obvious ASR error.

Be conservative. When uncertain, leave the original text unchanged.
```

### 8.2 Few-shot 예시

아래 예시를 System Prompt 뒤에 포함한다. 예시는 **실제 eval(`samples/language_hint_comparison.json`)에서 관측된 오류 패턴**을 기반으로 구성했으며, 언어별로 오류 성격이 다르다는 점을 반영한다.

* **ja/zh**: 정규화 후 남는 오류는 대부분 고유명사 이표기(음차/한자 대체). → "문맥으로 정답 고유명사를 고르는" 유형.
* **ko**: 정규화해도 남는 오류가 표기 차이가 아니라 **사전에 존재하는 두 실제 단어 간 혼동**(몰아냈음↔보란했음, 무척↔부쩍, 족히↔조기). → 문법으로 판별 가능한 경우와, 문장 하나로는 판별 불가해 원문을 유지해야 하는 경우로 나뉜다.

```text
Examples:

Example 1 (Japanese — proper noun corrected via topic context, katakana transliteration error):
<PREVIOUS_CONTEXT>
今日は買い物に行ってきました。
</PREVIOUS_CONTEXT>
<CURRENT>
新しいベルサージのバッグを買いました。
</CURRENT>
<OUTPUT>
新しいヴェルサーチのバッグを買いました。
</OUTPUT>

Example 2 (Chinese — homophone-like proper noun corrected via context, hanzi substitution error):
<PREVIOUS_CONTEXT>
我们打算这个月去欧洲旅行。
</PREVIOUS_CONTEXT>
<CURRENT>
第一站是克隆,然后去巴黎。
</CURRENT>
<OUTPUT>
第一站是科隆,然后去巴黎。
</OUTPUT>

Example 3 (Korean — grammatically implausible word replaced with the word that actually fits):
<PREVIOUS_CONTEXT>
공사가 언제 끝날지 물어봤어요.
</PREVIOUS_CONTEXT>
<CURRENT>
아무리 빨라도 3개월은 조기 걸린다고 하더라고요.
</CURRENT>
<OUTPUT>
아무리 빨라도 3개월은 족히 걸린다고 하더라고요.
</OUTPUT>

Example 4 (Korean — real word confused with another real word, resolved by sentence-level meaning):
<PREVIOUS_CONTEXT>
이번 분기 실적 발표에서 있었던 일이에요.
</PREVIOUS_CONTEXT>
<CURRENT>
저희가 경쟁사를 시장에서 보란했음을 강조했어요.
</CURRENT>
<OUTPUT>
저희가 경쟁사를 시장에서 몰아냈음을 강조했어요.
</OUTPUT>

Example 5 (Korean — two equally valid real words, context insufficient → leave unchanged):
<PREVIOUS_CONTEXT>
다이어트 시작한 지 얼마 안 됐다고 들었는데요.
</PREVIOUS_CONTEXT>
<CURRENT>
네, 그런데 요즘 식욕이 무척 늘어서 걱정이에요.
</CURRENT>
<OUTPUT>
네, 그런데 요즘 식욕이 무척 늘어서 걱정이에요.
</OUTPUT>

Do not copy these examples' content. They illustrate the correction style only.
```

> **Example 5가 특히 중요하다.** 실제 eval에서 CER이 떨어지지 않은 케이스(무척/부쩍처럼 둘 다 정상 어휘라 판별 불가)와 같은 유형으로, 모델에게 "이 정도 애매함이면 건드리지 마라"는 기준선을 직접 보여준다. 예시가 전부 "교정하는" 사례뿐이면 모델이 과교정(over-correction)으로 편향되므로, "고치지 않는" 사례를 최소 1개 포함하는 것이 핵심이다.

> 예시 언어·도메인은 실제 처리 콘텐츠 특성에 맞춰 교체하는 것을 권장한다(기술 용어가 많으면 기술 고유명사 예시로 대체 등).

### 8.3 실제 요청 시 Input 구조

```text
<PREVIOUS_CONTEXT>
92' 문장 텍스트
93' 문장 텍스트
...
100' 문장 텍스트
</PREVIOUS_CONTEXT>

<CURRENT>
101 문장 텍스트 (ASR 원문)
</CURRENT>
```

기대 출력:

```text
<OUTPUT>
101' 교정된 텍스트
</OUTPUT>
```

Stop Sequence로 `</OUTPUT>` 지정. 파싱은 `<OUTPUT>`~`</OUTPUT>` 사이 텍스트를 추출하고 앞뒤 공백을 제거한다.

---

## 9. 장점

* VRAM 사용량 일정 / Context 크기 일정 → 긴 영상도 처리 가능
* 교정 결과가 다음 문장의 문맥으로 사용되어 용어 일관성 향상
* 실패 시 특정 문장만 재실행 가능 (원본 보존 전제, §5)
* 낮은 temp + 고정 출력 포맷으로 결과 안정성 확보

---

## 부록 A. v1 → v2 변경 이력

| # | 변경 | 근거 |
|---|---|---|
| 1 | Thinking 모드 비활성화를 필수 요건으로 명시 (§2) | Qwen3.5/3.6은 기본 thinking. n-predict 제한과 결합 시 출력이 비는 사고 발생 |
| 2 | Temperature 0.0 → 0.1~0.2 권장으로 변경 (§2, §3) | 해당 계열 temp=0에서 반복·붕괴 경향. 안정성/재현성 절충 |
| 3 | Top-p 0.9→0.8, Top-k 20 추가 | Qwen3.5/3.6 non-thinking 권장 샘플링값 반영 |
| 4 | llama.cpp 플래그 정정 (`--gpu-layers`→`--n-gpu-layers`), n-predict를 요청별 파라미터로 이동 (§3) | 실제 플래그명/운용 방식 반영 |
| 5 | CUDA 13.2 gibberish 주의사항 추가 (§3) | Qwen3.5/3.6 + llama.cpp 알려진 이슈 |
| 6 | 원본·교정본 이중 저장으로 변경 (§5) | 오교정 사후 재교정, glossary 업그레이드 대비 |
| 7 | Sliding Window 오교정 전파 실패 모드 명시 및 대응 방침 정리 (§6) | feedback 구조의 본질적 약점. as-is 진행 근거와 한계 명확화 |
| 8 | Input/Output XML 태그 구획 도입 (§8.1, §8.3) | context 수정 금지·현재 문장만 출력 규칙 안정화, 파싱 용이 |
| 9 | Few-shot 예시 추가 (§8.2) | 규칙만으로는 과교정 경계가 모호. eval 실측 오류 패턴 기반 |
| 10 | Glossary 구축 시점을 "전사 완료 후·후처리 전"으로 명확화 (§4, §7) | 전사 이전엔 용어집을 알 수 없음. 사후 구축이 유일하게 성립 |

## 부록 B. 오교정 전파 완화 옵션 (향후)

초기 버전은 as-is(§6)로 진행하되, 품질 이슈가 관측되면 아래를 단계적으로 도입한다.

| 옵션 | 방식 | 효과 | 비용 |
|---|---|---|---|
| 원문 window | context에 교정본 대신 **원본 ASR** 문장을 넣음 | 오교정 전파 원천 차단 | 고유명사 일관성 향상 효과 상실 (glossary로 대체 필요) |
| Glossary 고정 주입 | 전사 전체 스캔으로 고유명사 정규표기 확정 후 전 문장에 주입 (§7) | 전역적 표기 일관성, 전파 차단 | glossary 구축용 LLM 호출 추가 |
| 원문/교정본 병기 | context 각 문장에 `원문 / 교정본` 함께 제공 | 이전 교정을 "확정"이 아닌 "참고"로 취급 | context 2배, 프롬프트 복잡도↑ |
| Two-pass | 1차 독립 교정 → glossary 확정 → 2차 재교정 | 품질 최상 | LLM 호출 2배 |

**권장 조합**: "원문 window + glossary 고정 주입". feedback loop를 제거해 전파를 차단하면서 일관성은 glossary로 확보하며, 문장별 병렬 처리까지 가능해진다.
