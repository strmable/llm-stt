# Pipeline 실행 방법

`phase_a_roadmap.md`의 Stage 1~4를 구현한 독립 실행 CLI 스크립트 모음. 각 스크립트는 단독으로
실행 가능하며, 뒷 단계로 갈수록 앞 단계의 산출물(`temp/{job_id}/` 아래)을 그대로 재사용한다.

4단계를 한 번에 실행하고 싶으면 저장소 루트의 [`run_transcript.py`](../run_transcript.py)를 쓴다:
`python run_transcript.py path/to/source_video.mp4` — 성공하면 `{source}.srt`를 원본 옆에 만들고
`temp/{job_id}/`를 삭제한다.

## 0. 사전 준비

- `ffmpeg`/`ffprobe`가 PATH에 있어야 함
- Python 패키지: `soundfile`, `onnxruntime`, `numpy`, `matplotlib`, `requests`
- Stage 3(전사)만 llama-server(Qwen3-ASR)가 필요함 — `config.json`의 `local_api.launch_mode`가
  `"external"`(기본)이면 [SETUP.MD](../SETUP.MD)/[TESTING.md](../TESTING.md)대로 미리 직접 띄워둬야
  하고, `"managed"`면 Stage 3가 시작될 때 알아서 실행하고 끝나면 종료한다 (아래 "Managed 모드" 참고)
- Silero VAD `.onnx` 모델은 Stage 2a/2b/2c 최초 실행 시 `models/silero_vad.onnx`로 자동 다운로드됨 (gitignore 대상)

모든 명령은 저장소 루트에서 실행한다 (`temp/`, `models/`가 실행 위치 기준으로 생성되므로).

---

## Stage 1 — `extract_audio.py`

임의 미디어 파일 → `temp/{job_id}/audio_16k_mono.wav` (16kHz mono WAV).

```bash
python pipeline/extract_audio.py path/to/input.mp4
python pipeline/extract_audio.py path/to/input.mp4 --job-dir temp/custom   # 출력 위치 강제 지정
```

- `job_id`는 입력 파일의 절대경로+mtime+크기로 결정되므로(design.md §14.1), 같은 파일을 다시 실행하면 같은 `temp/{job_id}/`를 재사용한다.
- 성공 시 `temp/{job_id}/source_info.json`에 원본 파일의 경로/mtime/크기를 기록한다. 이후 단계(2a~2c)를
  원본 파일이 아니라 이미 추출된 `audio_16k_mono.wav`에 직접 대고 돌려도(아래 참고) 이 파일을 통해
  원본 경로를 추적한다 — 이게 없으면 Stage 4가 만드는 SRT가 원본 영상 옆이 아니라 `temp/` 안에
  엉뚱한 이름으로 만들어진다.
- 자동 검증: 출력 WAV를 다시 열어 `sample_rate==16000`, `channels==1`, `duration≈원본`(±0.1s) assert.
  실패 시 조용히 넘어가지 않고 예외 발생.
- **사람이 직접 해야 하는 것**: 출력 WAV를 실제로 들어서 원본과 내용이 같은지 확인 (자동화 불가).

---

## Stage 2a — `vad_raw_test.py`

Stage 1 산출물에 Silero VAD(onnxruntime 직접 로드, design.md §12.1)를 프레임 단위로 돌려 **후처리 없는
원시 결과**를 확인한다.

```bash
python pipeline/vad_raw_test.py path/to/input.mp4
python pipeline/vad_raw_test.py path/to/input.mp4 --threshold 0.3   # 0.3/0.5/0.7 비교 권장
```

- 이미 추출된 WAV를 아는 경우 원본 미디어 대신 그 WAV 경로를 직접 넘겨도 된다(재추출 생략):
  `python pipeline/vad_raw_test.py temp/{job_id}/audio_16k_mono.wav`
- 출력: raw segment 목록(콘솔) + `{stem}_vad_raw_t{threshold}.png` (파형+확률곡선+구간 하이라이트,
  30초 단위로 여러 줄에 나눠 그리고, 10줄 넘으면 `_p1.png`/`_p2.png`로 페이지 분할)
- **사람이 직접 해야 하는 것**: PNG를 보면서 "말하는 구간에 하이라이트가 실제로 걸쳐 있는가",
  "짧은 감탄사가 통째로 누락되지 않는가" 확인 (통과 기준: 문장 단위 발화 누락 0건).

---

## Stage 2b — `vad_merge.py`

Stage 2a의 raw segment에 병합 정책(design.md §12.3, 순서 고정)을 적용:
짧은 침묵 병합 → 짧은 발화 흡수(단, 거리 제한 있음) → 30초 초과 시 강제 분할.

```bash
python pipeline/vad_merge.py path/to/input.mp4
python pipeline/vad_merge.py path/to/input.mp4 \
    --threshold 0.5 --min-silence 0.7 --min-speech 1.0 --max-absorb-gap 3.0 --max-chunk 30
```

| 옵션 | 기본값(`config.json`/`config.example.json`의 `vad` 섹션) | 의미 |
|---|---|---|
| `--threshold` | 0.5 | Stage 2a와 동일, VAD 확률 임계값 |
| `--min-silence` | 0.7 | 이 값 이하 침묵은 앞뒤 발화를 이어붙임 |
| `--min-speech` | 1.0 | 이보다 짧은 발화 구간은 이웃 구간에 흡수 |
| `--max-absorb-gap` | 3.0 | 흡수 시 건널 수 있는 침묵 거리의 상한. 초과하면 흡수하지 않고 독립 chunk로 남김 (음수를 주면 무제한 — 예전 동작) |
| `--max-chunk` | 30.0 | 이보다 긴 구간은 이 길이로 기계적 강제 분할 (모델 입력 상한) |

다섯 값 모두 CLI 인자를 안 주면 `config.json`(없으면 `config.example.json`)의 `vad` 섹션에서 읽어온다
(`common.vad_defaults()`). `config.json`을 고쳐서 재실행하면 바로 반영되고, CLI 인자를 명시하면 그 값이
항상 우선한다. `vad_raw_test.py --threshold`, `chunk_export.py`도 동일한 소스를 공유한다.

- 출력 PNG(`{stem}_vad_merged_..._t.._s.._m.._g.._c...png`)는 raw(해치 무늬 테두리)와 merged(옅은
  배경색)를 겹쳐 그려서, 병합이 실제로는 분리돼야 할 구간을 뭉개지 않았는지 눈으로 비교할 수 있게 한다.
- 자동 검증: 모든 최종 구간이 30초 이하인지 assert.
- **사람이 직접 해야 하는 것**: raw/merged PNG를 비교하며 병합이 과한지(문장이 섞임) 부족한지(chunk 수가
  너무 많음, 체감 기준 분당 2~4개) 판단.

---

## Stage 2c — `chunk_export.py`

Stage 2b의 최종 구간을 실제 `chunk_NNNN.wav` 파일로 잘라 저장하고 `manifest.json`을 생성한다
(design.md §14.2 스키마).

```bash
python pipeline/chunk_export.py path/to/input.mp4
python pipeline/chunk_export.py path/to/input.mp4 --min-silence 0.7 --min-speech 1.0   # 2b와 동일 옵션 공유
```

- Chunk는 `audio_16k_mono.wav`에서 **샘플 인덱스로 직접 슬라이싱**한다 (chunk마다 ffmpeg를 다시 호출하지
  않음 — 부동소수점 seek 오차가 누적될 여지를 원천 차단). 각 chunk의 offset은 manifest에 세 가지 형태로
  중복 기록: `start_sec`/`end_sec`(연산용), `start`/`end`(`HH:MM:SS.mmm`, 사람이 보기용/SRT 재사용),
  `start_sample`/`end_sample`(정확한 슬라이싱 인덱스, 재현/디버깅용).
- `manifest.json`의 `source_file`/`source_mtime`/`source_size`/`output_srt`는 **Stage 1이 남긴
  `source_info.json`이 있으면 그것을 최우선으로 사용**한다. 없으면(예: 원본 미디어 없이 추출된 WAV만
  갖고 이 스크립트를 바로 돌린 경우) WAV 자체 경로로 대체하고 콘솔에 WARNING을 출력 — 이 경우
  `output_srt`가 원본 영상 옆을 가리키지 않으므로 그대로 믿지 말 것.
- 이미 Stage 3을 거쳐 `status: "transcribed"`인 chunk가 있는 상태에서 이 스크립트를 다시 돌리면(같은
  VAD 파라미터로 재추출), `chunk_NNNN.txt`가 실제로 남아있는 chunk는 `transcribed` 상태를 그대로
  보존한다(재전사 낭비 방지).
- 자동 검증: chunk 목록이 시간순으로 겹치지 않는지 assert, 각 chunk wav를 다시 열어 의도한 길이와
  일치하는지 assert.
- **사람이 직접 해야 하는 것**: 로그 마지막에 무작위로 뽑아주는 chunk 5개를 직접 들어서 문장이 중간에
  잘리지 않았는지 확인.

---

## Stage 3 — `transcribe_chunks.py`

`manifest.json` + chunk WAV들을 순회하며 llama-server(Qwen3-ASR)에 전사 요청.

```bash
python pipeline/transcribe_chunks.py temp/{job_id}
python pipeline/transcribe_chunks.py temp/{job_id}/manifest.json --language ko
python pipeline/transcribe_chunks.py temp/{job_id} --server http://localhost:8080 --force
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--server` | `http://localhost:8080` | llama-server 주소 |
| `--model` | `qwen3-asr` | 요청 body의 `model` 필드 |
| `--language` | `auto` | `auto`면 힌트 없이 "Transcribe this audio.", 언어코드(`ko`/`ja`/`zh` 등)면 "language: {code}" (design.md §5A.8) |
| `--temperature` | 1.0 | design.md §6.1 기본값 |
| `--timeout` | 120 | 요청당 타임아웃(초) |
| `--force` | off | 이미 `transcribed` 상태인 chunk도 재전사 (기본은 건너뜀 — resume 방식) |

- 전사를 시작하기 전 `pipeline/server_manager.py`가 서버 상태를 확인한다 (design.md §6.3, 아래
  "Managed 모드" 참고). 이미 응답 중이면 그대로 재사용하고, 응답이 없는데 `launch_mode`가
  `"external"`이면 즉시 종료.
- 결과는 `chunk_NNNN.txt`로 저장(가공된 최종 텍스트), 콘솔에는 원문 그대로 로그.
- 실패 시 1회 재시도 후 `status: "failed"`로 표시하고 다음 chunk로 계속 진행(design.md §21).
- 같은 2-20자 패턴이 5회 이상 연속 반복되면 `[WARNING: possible infinite repetition ...]` 표시 —
  참고용 휴리스틱이지, 할루시네이션 검사를 대체하지 않는다.
- **아직 구현 안 한 것(의도적)**: Context Carryover, Custom Vocabulary 프롬프트 주입(design.md §17,
  §5B.2). Qwen3-ASR의 고정 출력 형식(`language {Lang}<asr_text>{text}`)에 그 프롬프트 문구가 실제로
  어떻게 반응하는지 검증되지 않아서 이번 단계 범위에서 제외.
- **사람이 직접 해야 하는 것**: Stage 0 정답 전사가 있는 클립이면 CER 계산(현재 스크립트에 없음,
  참고: `tools/eval_language_hint.py`), 그리고 `chunk_NNNN.txt`를 `chunk_NNNN.wav`와 나란히 놓고
  할루시네이션/무한반복/언어오판 육안 확인.

### Managed 모드 — llama-server 자동 실행/종료 (design.md §6.3)

`config.json`(없으면 `config.example.json`)의 `local_api` 섹션으로 제어한다:

```json
"local_api": {
  "launch_mode": "managed",
  "server_binary": "C:\\ai\\llama\\llama-server.exe",
  "model_path": "",
  "mmproj_path": "",
  "hf_repo": "ggml-org/Qwen3-ASR-1.7B-GGUF",
  "managed": {
    "port": 8080,
    "extra_args": "--ctx-size 4096 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0",
    "startup_timeout_sec": 120
  }
}
```

- `launch_mode`가 `"external"`(기본)이면 이 섹션은 무시되고, 미리 떠 있는 서버가 반드시 있어야 한다.
- `"managed"`면 `pipeline/server_manager.py`가 다음 순서로 동작한다 (`transcribe_chunks.py`가 자동으로
  이 로직을 거치므로 `run_transcript.py`/`transcribe_chunks.py` 어느 쪽으로 실행해도 동일하게 적용됨):
  1. 지정된 `--server` 주소가 이미 응답하는지 확인 → **응답이 있으면 그대로 재사용하고, 이 프로그램이
     직접 켠 게 아니므로 끝나고 나서도 절대 종료시키지 않는다** (design.md §6.3 규칙 그대로).
  2. 응답이 없으면 `server_binary`로 subprocess 실행. `model_path`가 채워져 있으면
     `--model {model_path}` (+`--mmproj {mmproj_path}`), 비어 있으면 `-hf {hf_repo}`로 자동 다운로드
     (TESTING.md에서 검증된 방식). `managed.extra_args`를 그대로 뒤에 붙인다.
  3. `/health`를 `managed.startup_timeout_sec`(기본 120초)까지 폴링, 준비되면 전사 시작.
  4. Stage 3가 끝나면(성공/실패 무관, `finally`) 이 프로그램이 직접 켠 프로세스만 종료(SIGTERM, 15초
     내 미종료 시 강제 종료).
- 서버의 stdout/stderr는 `temp/{job_id}/llama-server.log`에 기록된다 (콘솔에 그대로 흘리지 않음).
- 실측 확인(2026-07-09, `rec1.m4a`): 서버 없는 상태 → managed 기동 2.0s 후 healthy → 전사 → 자동 종료
  → `/health` 재확인 결과 완전히 내려감. 반대로 서버를 미리 직접 띄워둔 상태에서 같은 명령을 실행하면
  "이미 응답 중" 메시지와 함께 재사용하고, 끝난 뒤에도 그 서버는 그대로 살아있음을 확인.

---

## Stage 4 — `build_srt.py`

`manifest.json`의 chunk 시각 + `chunk_NNNN.txt`를 조합해 SRT를 만든다.

```bash
python pipeline/build_srt.py temp/{job_id}
python pipeline/build_srt.py temp/{job_id}/manifest.json --output custom.srt
```

- 기본 출력 경로는 `manifest["output_srt"]` (design.md §13: 원본 파일과 같은 폴더, `{원본파일명}.srt`).
  Stage 2c가 `source_info.json`을 못 찾아 WARNING을 띄운 job이면 이 경로가 의미 없을 수 있음.
- `status != "transcribed"`이거나 텍스트가 빈 chunk는 건너뛰고 개수를 로그로 남긴다.
- 자동 검증: 타임스탬프가 시간순이고 겹치지 않는지 assert.
- **사람이 직접 해야 하는 것**: 실제 비디오 플레이어(VLC 등)에 SRT를 얹어 재생 — 자막 타이밍이
  체감상 맞는지 확인 (통과 기준: 육안상 0.5초 이상 밀리는 구간 0건). 이건 원본 영상 파일이 있어야만
  가능하다.

---

## 진행 상황 (2026-07-09 기준)

Stage 1~4 스크립트 전부 작성 완료, 실제 파일 4개로 end-to-end 실행 확인:

| 파일 | 길이 | 상태 |
|---|---|---|
| `rec1.m4a` | 3.7s | 1~4 전부 통과, SRT 생성 확인 |
| `temp/8bc2b06c` (원본 영상 없음) | 133s | 1~4 전부 통과. `source_info.json` 없어서 SRT가 `temp/` 안에 생성됨 |
| `temp/e3a3ac10` (원본 영상 없음) | 526s | 동일 |
| `F:\tmp\jp-sam2.mkv` | 392s | 1~4 전부 통과, SRT가 원본 영상 옆(`F:\tmp\jp-sam2.srt`)에 정상 생성 확인 |

미완료/알려진 갭:

- **Stage 0 정식 테스트 셋 없음** — `testset/clip_01~04.*` + 정답 전사 파일을 아직 안 만듦. 위 4개
  파일로 대체 실측했지만 "깨끗함/배경음악/짧은 감탄사/화자 2명 겹침" 4가지 대표성은 의도적으로 확보된
  게 아님.
- **CER 정량 평가 없음** — 정답 전사가 있는 클립이 없어서 Stage 3에 CER 계산을 아직 안 붙임.
- **사람이 직접 들어야 하는 검증들**(Stage 1 청취, 2a/2b 육안 PNG, 2c 무작위 청취, 3 할루시네이션 대조,
  4 VLC 재생)은 전부 위 표에 적힌 자동 검증까지만 대신했고, 실제 청취/재생 확인은 아직 아무도 안 함.
- ~~config.json의 VAD 기본값이 스크립트에 연결 안 됨~~ → 해결. `vad_raw_test.py`/`vad_merge.py`/
  `chunk_export.py`의 CLI 기본값은 이제 `common.vad_defaults()`를 통해 `config.json`(없으면
  `config.example.json`)의 `vad` 섹션에서 읽어온다. CLI 인자를 명시하면 그 값이 항상 우선한다.
  `config.json`의 `provider`/`local_api`/`gemini`/`prompt` 등 나머지 섹션은 아직 어떤 스크립트도
  읽지 않는다 (아직 GUI/main.py가 없어서 그 값들을 쓸 곳 자체가 없음).
