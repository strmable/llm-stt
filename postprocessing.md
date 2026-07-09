# ASR 후처리 설계

## 목적

Qwen-ASR로 생성된 전사 결과의 품질을 향상시키기 위한 후처리 단계이다.

후처리에서는 다음과 같은 오류를 최소한의 수정만으로 교정한다.

* 문맥에 맞지 않는 동음이의어
* 잘못 인식된 고유명사
* 중복 단어 (예: `主 主`)
* 조사 및 띄어쓰기 오류
* 명백한 오탈자

반대로 다음 작업은 수행하지 않는다.

* 문장 재작성
* 의역
* 내용 추가
* 요약
* 문체 변경

즉, **ASR 결과를 최대한 유지하면서 필요한 부분만 수정하는 것이 목표**이다.

---

# 모델 선정

후처리 모델은 **Qwen3.5-9B**를 사용한다.

선정 이유

* Instruct를 잘 따름
* Tool Calling 지원
* JSON 출력 안정성 우수
* 과도한 Reasoning이 적음
* 응답 속도가 빠름
* llama.cpp에서 안정적으로 동작
* 16GB VRAM 환경에서 8bit(Q8_0) 운용 가능

ASR 후처리는 복잡한 추론보다

* 원문 유지
* 최소 수정
* 높은 Instruction 준수

가 더 중요하므로 Qwen3 계열보다 Qwen3.5-9B가 적합하다.

---

# llama.cpp 실행

예시

```bash
llama-server \
  -m qwen3.5-9b-q8_0.gguf \
  --ctx-size 8192 \
  --n-predict 256 \
  --gpu-layers 999 \
  --temp 0.0 \
  --top-p 0.9 \
  --repeat-penalty 1.0 \
  --host 0.0.0.0 \
  --port 8080
```

권장 설정

| 항목             | 권장값         |
| -------------- | ----------- |
| 양자화            | Q8_0 (8bit) |
| Context        | 8192        |
| Max Predict    | 256~512     |
| Temperature    | 0.0         |
| Top-p          | 0.9         |
| Repeat penalty | 1.0         |
| GPU Layers     | 전체 Offload  |

### Context 크기

16GB VRAM 기준에서는 8192 정도가 가장 무난하다.

후처리는 긴 전사를 한 번에 처리하지 않고 Sliding Window 방식으로 수행하므로, 지나치게 큰 Context는 필요하지 않다.

### 출력 토큰 제한

후처리는 현재 문장 하나만 교정하는 작업이므로 긴 출력이 필요하지 않다.

따라서 `--n-predict`는 **256~512 정도**를 권장한다.

출력 토큰을 제한하면 다음과 같은 장점이 있다.

* 프롬프트를 잘못 이해하여 이전 문장을 다시 출력하는 것을 방지
* 장황한 설명이나 불필요한 Reasoning 출력 방지
* 처리 시간 단축
* GPU 메모리 사용량 감소
* 동일한 입력에 대해 보다 안정적인 결과 생성

현재 설계에서는 Sliding Window의 마지막 문장 하나만 출력하므로 256~512 토큰이면 충분하다.

### Stop Sequence 사용

가능하다면 API 또는 llama.cpp의 Stop Sequence 기능도 함께 사용하는 것이 좋다.

예를 들어 출력 형식을

```
<OUTPUT>
...
</OUTPUT>
```

처럼 정의한 경우 `</OUTPUT>`을 Stop Sequence로 지정하면 된다.

이렇게 하면 모델이 마지막 문장을 출력한 뒤 추가 설명이나 다음 문장을 생성하는 것을 방지할 수 있다.

---

# 후처리 방식

ASR 완료 후 후처리를 수행한다.

즉,

```
Audio

↓

ASR

↓

전체 전사 완료

↓

후처리
```

ASR 진행 중에는 LLM을 호출하지 않는다.

ASR과 후처리를 분리하면 구현이 단순해지고, 후처리 실패 시에도 전사를 다시 수행할 필요가 없다.

---

# 전사 저장

전사가 완료되면 문장 단위로 분리하여 저장한다.

예)

```
000001.txt
000002.txt
000003.txt
...
000845.txt
```

문장 번호는 이후 후처리 시 순서를 유지하기 위해 사용한다.

---

# Sliding Window 교정

후처리는 현재 문장만 수정한다.

단, 최근 교정 완료된 문장을 함께 Context에 제공한다.

예를 들어 Window 크기가 10이라면

```
교정 완료

91
92
93
94
95
96
97
98
99
100

현재 ASR

101
```

LLM은

```
101만 수정
```

하여 출력한다.

다음 단계는

```
92'
93'
94'
95'
96'
97'
98'
99'
100'
101'

현재

102
```

와 같이 수행한다.

즉,

* 이전 문장은 모두 교정 완료본
* 현재 문장만 수정
* 출력도 현재 문장 하나

이다.

교정이 완료된 문장이 다음 문장의 문맥(Context)으로 계속 사용되므로, 고유명사 표기나 용어의 일관성이 점차 향상되는 효과를 기대할 수 있다.

---

# Sliding Window를 사용하는 이유

수백 문장 이전의 내용이 현재 문장의 동음이의어 선택에 영향을 줄 가능성은 낮다.

반대로 최근 대화는 매우 중요한 문맥이 된다.

예)

```
A:
이번에 베르사체 신상 봤어?

B:
응.

ASR

벨사지 예쁘더라.
```

최근 문맥만 있어도

```
벨사지

↓

베르사체
```

로 교정할 가능성이 높아진다.

따라서 긴 Context 전체를 사용하는 것보다 최근 문장만 유지하는 것이 효율적이다.

Window 크기는 초기에는 **최근 10문장 정도**로 시작하고, 실제 테스트 결과에 따라 조정하는 것을 권장한다.

---

# Prompt 원칙

Prompt는 최대한 보수적으로 작성한다.

예시


```text
You are an Automatic Speech Recognition (ASR) post-processing assistant.

Your task is to recover the speaker's intended utterance from an ASR transcript.

The transcript may contain recognition errors such as:
- homophone substitutions
- incorrect proper nouns
- duplicated words or phrases
- obvious character mistakes
- minor punctuation or formatting errors

Rules:

1. Correct only clear ASR recognition errors.
2. Use the previous context to disambiguate homophones and proper nouns.
3. Preserve the speaker's intended message, style, and tone.
4. Do NOT rewrite, summarize, paraphrase, or improve the sentence.
5. Do NOT add information that was not spoken.
6. If multiple interpretations are possible, keep the original text.
7. Previous context is provided only for understanding. Never modify it.
8. Correct ONLY the last transcript segment.
9. Output ONLY the corrected last segment. Do not output explanations, comments, or any other text.

The transcript may be in Japanese, Chinese, Korean, or a mixture of these languages.
Apply the normal writing conventions of each language only when necessary to correct an obvious ASR error.

Be conservative.
When uncertain, leave the original transcript unchanged.

Output format:

Return exactly one corrected transcript segment.

Do not repeat previous context.
Do not include explanations.
Do not include markdown.
Do not include quotation marks.
Do not include numbering.

```

Temperature는 0으로 설정하여 항상 동일한 결과가 나오도록 한다.

가능하면 출력 형식도 고정하여 항상 동일한 형태의 결과만 반환하도록 한다.

---

# 장점

* VRAM 사용량 일정
* Context 크기 일정
* 긴 영상도 처리 가능
* 교정 결과가 다음 문장의 문맥으로 사용됨
* 고유명사 및 용어의 일관성 향상
* 실패 시 특정 문장만 재실행 가능
* 동일한 입력에 대해 동일한 결과 생성

---

# 향후 개선

추후에는 다음 기능을 추가할 수 있다.

* Window 크기 자동 조절
* 고유명사 자동 일관성 검사
* 교정 신뢰도 출력
* 수정 Diff 출력
* 사람이 검토해야 할 문장 자동 표시
* 교정 결과 캐시

초기 버전에서는 단순한 Sliding Window 방식만으로도 대부분의 ASR 오류를 안정적으로 교정할 수 있을 것으로 예상된다.
