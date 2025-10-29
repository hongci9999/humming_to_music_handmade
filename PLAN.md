# 허밍 기반 반주 생성 서비스 구현 계획서

버전: 0.1 (초안)
작성일: 2025-10-26

---

## 1) 목표와 범위

- **목표**: 사용자 허밍을 입력받아 조성/길이를 분석하고, 지정한 악기·분위기·장르에 맞는 2마디 루프를 3–4개 생성한 뒤, 이를 반복·전환하여 허밍 길이의 반주를 자동 편곡·출력.
- **초기 범위(MVP)**:
  - 허밍 분석: 키(조성)·BPM·길이(초) 추정 및 구간별 키 변화 감지(선택).
  - 루프 생성: 라벨된 루프에서 조건부 검색→ 전조·타임스트레치로 정합, 필요시 경량 변주 생성.
  - 장기 편곡: 2마디 단위로 시퀀싱, 전환 크로스페이드, 템포·키 일치 보장.

## 2) 사용자 흐름

1. 사용자가 허밍 오디오 업로드(또는 마이크 녹음) + 옵션(악기/분위기/장르) 입력
2. 시스템이 허밍을 분석(키/BPM/길이/구간키) → 조건 쿼리 생성
3. 루프 후보 3–4개 생성(조건부 검색+전조/타임스트레치/변주)
4. 허밍 길이에 맞게 루프 시퀀싱 → 전환 처리 → 최종 오디오 반환(미리듣기 + 다운로드)

### 2.1 옵션 미선택(Zero-input) 정책

- 기본 추론: 허밍의 `global_key`, `bpm`만으로 작동
- 기본값
  - instrument: 허밍 톤 중심으로 자동 추천(멜로디성 높음→Piano_01, 리듬성 높음→Drums_01, 중립→Guitar_02)
  - genre: tempo 기반 근사(≤90: Ballad, 90–120: Pop/Dance, ≥120: EDM/Rock)
  - mood: 크로마 분산·F0 안정성 기반(안정/밝음→bright, 불안/어둠→dark, 중립→neutral)
  - form: isMain 우선(없으면 Intro/Verse 선호)
- 제안 루프 수: 3개(경량) 또는 4개(기기 성능 여유 시)
- UI 피드백: 생성 결과 카드에 자동 선택된 기본값과 근거(키/BPM/분류)를 함께 표시

## 3) 데이터셋 개요(보유 데이터)

- WAV 37,668개, 라벨 JSON 37,668개, MIDI 37,668개(대다수 2마디, 48kHz, 4/4, 3.5–5.5초)
- 정합성: 현재 TL(라벨)–TS(소스) 카테고리별 1:1 매핑 확인됨
- 카테고리(예시): Drums_01(Steel Drums), Drums_02(Synth Drum), Piano_01(Acoustic Grand), Piano_02(Bright), Guitar_02(Acoustic Steel), Guitar_04(Electric Clean), Bass_02(Electric Finger), Strings_01(Violin), Wind_06(Flute), Lead_01(Square)

## 4) 시스템 구성(아키텍처)

- Humming Analyzer: 피치 트래킹→키·BPM 검출→길이 산출→구간 키 변화(optional)
- Loop Indexer & Retriever: 라벨/메타 기반 인덱싱, 키·BPM 검증, 임베딩 생성(검색 가속)
- Loop Generator/Adapter: 전조(+/– n semitones), 타임스트레치(±5%), 라우드니스 정규화, 경량 변주(VAE/Transformer-MIDI 옵션)
- Arranger: 2마디 블록 배치, 전환 크로스페이드/드럼 필/리버브 테일 관리, 템포 스냅
- API: 업로드/옵션/미리듣기/결과 다운로드
- Frontend(후순위): 업로드 위젯, 옵션 선택, 파형/미디 시각화, 미리듣기 플레이어

## 5) 기능 상세 설계

### 5.1 허밍 분석

- Pitch: CREPE/pyin(librosa) → 프레임별 F0
- Key: 크룸한슬(K-K) 또는 Essentia key extractor, Tonal centroid(tonnetz) 기반 강건화
- BPM/Tempo: madmom 또는 librosa beat tracker, 2마디 기반 바/박 추정
- Segmentation: 키 신뢰도 변화에 따른 세그먼트 분할(키 전환 탐지, MVP는 전체 1키 가정 가능)
- Output: {global_key, bpm, duration_sec, segments: [{start, end, key}]}

#### 5.1.1 파이프라인 상세

- 입력 전처리: 모노 변환 → 리샘플(48k→16k) → 라우드니스 정규화(–23 LUFS 근사) → (옵션)VAD로 무음 제거
- 피치 추정: CREPE(small, hop ~10ms)로 f0_Hz·voicing 추정 → 저신뢰 프레임 마스킹 → 메디안 스무딩
- 템포/비트: onset envelope → beat_track로 BPM/비트 그리드 산출 → 2마디 스냅 포인트 계산
- 키 추정: chroma_cqt → K-K 템플릿 매칭(major/minor) + 대체 추정(essentia 가능) → 앙상블 점수
- 구간 분할: 슬라이딩 윈도우 크로마 상관·키 신뢰도 변화 감지 → HMM/메디안으로 과분할 방지
- 품질 플래그: low_snr, unstable_key, low_beat_confidence 등 기록

#### 5.1.2 사용 라이브러리

- 오디오 I/O: soundfile, ffmpeg-python(백엔드 ffmpeg)
- 전처리: pyloudnorm, webrtcvad(옵션)
- 피치: crepe(권장) / librosa.pyin(대안)
- 템포/비트: librosa(기본) / madmom(옵션)
- 키: librosa(chroma+K-K) / essentia(KeyExtractor, 옵션)
- 스무딩/모델: numpy/scipy, hmmlearn(옵션)

#### 5.1.3 출력 스키마 예시

{
"global_key": "C major",
"key_confidence": 0.86,
"key_alternatives": [{"key": "A minor", "score": 0.78}],
"bpm": 106.0,
"beat_grid": [0.00, 0.57, 1.13, "..."],
"duration_sec": 12.04,
"segments": [
{"start": 0.00, "end": 8.02, "key": "C major", "conf": 0.84},
{"start": 8.02, "end": 12.04, "key": "D major", "conf": 0.71}
],
"quality_flags": {"low_snr": false, "unstable_key": false}
}

### 5.2 루프 생성/선택

- Retrieval-first: 인덱스에서 (악기·분위기·장르·폼·isMain·BPM버킷·키) 조건으로 후보 검색
- Adaptation: 후보를 허밍 키/BPM에 정합(피치シ프트, 타임스트레치)
- Variation(옵션):
  - MIDI 루프가 있는 경우: 경량 Transformer로 2마디 변주 생성 → 리렌더
  - 오디오만 있는 경우: stoch. time-varying EQ/필터, 드럼필 샘플러 삽입
- 출력: 3–4개 상이한 캐릭터의 루프(이름, 키/BPM, 프리뷰 wav)

### 5.3 장기 편곡(시퀀싱)

- 규칙: 2마디 스냅 정렬, 허밍 키 타임라인에 맞춘 전조, 동일/유사 악기 간 루프 A→B 전환
- 전환: 50–150ms 크로스페이드, 드럼 필/라이저로 부자연스러움 완화
- 길이 맞춤: 허밍 길이까지 반복/전환 패턴 생성(예: A-A-B | A-A-C …)

## 6) 모델링 전략

- Phase 1(MVP): 검색+적응(Retrieval+Transform)만으로 출시 → 빠른 성과/안정성
- Phase 2: MIDI 2마디 조건부 생성기(20–60M 파라미터) 학습 → 변주 강화
- Phase 3: 오디오 생성기(멜-디퓨전 소형) 미세조정 → 텍스처 다양화

## 7) 데이터 파이프라인

- Index CSV/Parquet 생성: `label_json, wav_path, wav_exists, instrument_type, instrument_name, genre, style, form, bpm, scale, bar_count, beat_count, play_time_ms, loop_index, is_main`
- 검증: 라벨과 실제 추정 키/BPM 비교하여 outlier 플래그
- 임베딩: CLAP/AudioCLIP 또는 멜-스펙트럼統계 + UMAP 인덱스(FAISS)

### 7.1 라벨 JSON 스키마와 활용

- 스키마 예시

```
{
    "dataSet": {
        "version": "1.0",
        "loopIndex": "0007543",
        "loopType": "rhythm|melody|harmony",
        "classification": "타악기|현악기|...",
        "InstrumentType": "Drums|Guitar|...",
        "barCount": 2,
        "isMain": "Yes|No",
        "loopInfo": {
            "InstrumentName": "Steel Drums",
            "genre": "Dance",
            "bpm": 106,
            "musicStyle": "action",
            "songForm": "Intro|Verse|Chorus|...",
            "beatCount": 4,
            "scale": "D major"
        },
        "bitrate": 48000,
        "playTime": 4485.984,
        "srcType": "Wave",
        "srcFileName": "Drums_Steel Drums_0007543_Dance_106BPM.wav"
    }
}
```

- 필드 활용 요약

  - `srcFileName` → `TS_*` 폴더 WAV 매핑
  - `loopIndex` → 중복 방지 및 데이터 분할 단위
  - `InstrumentType`, `loopInfo.InstrumentName` → 악기 조건/클래스
  - `classification`, `loopInfo.genre`, `loopInfo.musicStyle` → 장르/분위기 필터
  - `loopInfo.bpm`, `loopInfo.beatCount`, `barCount` → 템포/박자/2마디 검증
  - `loopInfo.scale` → 키 매칭·전조량 계산(허밍과의 조성 정합)
  - `loopInfo.songForm`, `isMain` → 후보 우선순위 및 A/B 패턴 구성
  - `playTime` → 예상 루프 길이 검증, 2마디 스냅 확인

- 색인 컬럼 매핑

  - `instrument_type = InstrumentType`
  - `instrument_name = loopInfo.InstrumentName`
  - `genre = loopInfo.genre`
  - `style = loopInfo.musicStyle`
  - `form = loopInfo.songForm`
  - `bpm = loopInfo.bpm`
  - `scale = loopInfo.scale`
  - `bar_count = barCount`, `beat_count = loopInfo.beatCount`
  - `play_time_ms = playTime`, `loop_index = loopIndex`
  - `is_main = isMain`, `wav_path = TS_* / srcFileName`

- 랭킹/선택 점수(개요)
  - `score = w1*|전조반음| + w2*|BPM차| + w3*태그불일치 + w4*(2마디 아님)`
  - 상위 3–4개 선별 → 전조/타임스트레치 → 프리뷰 생성

## 8) 품질 지표/평가

- Tonal match(허밍 vs 루프): 키 일치율, Tonnetz 거리, 전조량 절대값
- Rhythm match: BPM 오차, 다운비트 정합률
- Transition smoothness: 크로스페이드 경계 SNR, 사용자 MOS(1–5)
- 다양성: 루프 간 MFCC/F0 분포 거리

## 9) API 설계(초안)

- POST /analyze: 오디오 업로드 → {key, bpm, duration, segments}
- POST /generate: 옵션{instrument, mood, genre, target_key, bpm, num=4} → 루프 리스트
- POST /arrange: {loops[], plan} → 최종 오디오(및 미디)
- GET /preview/{id}: HLS/MP3 프리뷰

## 10) 기술 스택

- Python: librosa, essentia, madmom/crepe, numpy, torch(옵션), soundfile, ffmpeg
- 검색: FAISS/Annoy
- API: FastAPI + Uvicorn
- 오디오 변환: ffmpeg + pyrubberband(대안: sox/phase vocoder)

## 11) 일정/마일스톤(6주)

- W1: 인덱서/특성추출, 품질 리포트, 검색 API 스켈레톤
- W2: 허밍 분석기 베타(Key/BPM/길이), 단일 키 가정
- W3: 루프 선택/적응(전조/타임스트레치), 3–4개 후보 출력
- W4: 편곡/전환 엔진, 프리뷰 생성
- W5: API 통합, 간이 프런트, 내부 베타
- W6: 평가/튜닝, 변주 생성(옵션), 문서화

## 12) 리스크 & 대응

- 라벨 불일치/누락 → 추정값 교차검증, outlier 제외
- 저표본 클래스 → 통합/제외, 데이터 증강
- 템포·키 추정 오류 → 신뢰도 기반 보수적 매칭, 사용자 수동 선택 옵션
- 라이선스/권리 → 상업 사용 범위 확인, 출력 변환 로그 저장

## 13) 산출물

- dataset_index.(csv|parquet), feature_store/(key,bpm,emb)
- services/: analyzer, retriever, arranger, api
- scripts/: 인덱싱, 전처리, 평가, 배치
- docs/: 사용 가이드, API 스펙

## 14) 성공 기준(MVP)

- 허밍→키/BPM/길이 추정 MAE: 키 정확도 ≥ 80%, BPM 오차 ≤ 3%
- 루프 3–4개 제안 모두 재생·정합 OK, 전환 청감 품질 MOS ≥ 3.5
- 허밍 길이 동일 트랙 생성 성공률 ≥ 95%

## 15) 학습 계획(2마디 WAV 생성을 위한 MIDI 조건부 생성)

- 목표: 분석으로 얻은 key/BPM에 맞는 2마디 루프를 조건부로 생성하고, 사운드폰트 기반 렌더링을 통해 WAV로 출력한다.

### 15.1 데이터 준비

- 대상: `barCount == 2`, `loopInfo.beatCount == 4`(4/4) 루프만 사용(기타 박자는 후속 지원)
- 인덱스 생성: 7장 파이프라인의 컬럼 설계대로 `dataset_index.(csv|parquet)` 생성
  - `split`: `loopIndex` 기준으로 train/val/test 분할(중복 방지)
- MIDI 파싱 및 양자화
  - `pretty_midi`로 노트/컨트롤/프로그램 이벤트 추출
  - 박자 그리드: 16단계/박 → 1마디 64스텝, 2마디 128스텝으로 양자화
- 키 정규화(전사)
  - 모든 시퀀스를 C 메이저/마이너 기준으로 전사해 저장, 원래 키는 `transpose_semitones`로 보존
- 증강
  - 12키 전조 순환, velocity/타이밍 미세 변형, 조건 드랍아웃

### 15.2 토크나이징(권장: REMI 변형)

- 조건 토큰: `[KEY_{C|...}] [MODE_{MAJ|MIN}] [BPM_{bucket}] [INSTR_{클래스}] [GENRE_{버킷}]`
- 시퀀스 토큰: `[BAR_1] [POS_x/64] [NOTE_ON_p,v] [NOTE_OFF_p] ... [BAR_2] ... [EOS]`
- 제약: 2마디 총 길이 128스텝 초과 토큰 금지, [EOS]는 정확히 2마디 종료 시점에만 허용

### 15.3 모델(경량 Transformer Decoder)

- 파라미터 규모: 20–60M(레이어 8–12, d_model 512–768, n_head 8–12)
- 입력: 조건 프롬프트 + 2마디 이벤트(teacher forcing)
- 출력: 다음 토큰 확률, 목표는 cross-entropy 최소화(라벨 스무딩 0.1)

### 15.4 학습 하이퍼파라미터(초안)

- 시퀀스 길이: ≤1024 토큰(2마디 범위 내)
- 옵티마이저: AdamW(lr=3e-4, β=(0.9,0.98), weight_decay=0.01)
- 스케줄: warmup 2k steps → cosine decay
- 배치: 32–64, 에폭: 20–40(early stop: val loss 5에폭 정체)
- 정규화: 토큰 드롭(조건 10–20% 드랍), gradient clipping 1.0

### 15.5 길이/키 보장 규칙(미디·렌더링)

- 길이 고정
  - 생성 시 128스텝(2마디) 도달 즉시 [EOS] 강제, 초과 토큰 무시
  - 모든 Note-Off/페달 Off를 2마디 끝 tick 이하로 클램프
  - MIDI 메타: 단일 Tempo/TimeSig, `EndOfTrack`는 2마디 끝 tick에 배치
- 키 정합
  - 학습 시 C 기준, 추론 완료 후 `transpose_semitones`로 target key 전조
- WAV 길이 트림
  - 목표 샘플 수 `N = round(sr * (bars*beats*60/bpm))`로 하드 컷 + 5–10ms 페이드아웃

### 15.6 평가 지표

- Tonal match: 키 일치율, Tonnetz 거리, |전조량|
- Rhythm match: BPM 오차, 다운비트 스냅률
- 길이 정확도: MIDI 마지막 이벤트 위치, WAV 샘플 길이 오차(≤ 1프레임)
- 다양성: 루프 간 MFCC/F0 분포 거리, 토큰 n-gram 다양성
- MOS: 내부 청취 평가(1–5)

### 15.7 추론 파이프라인

- 입력: `{target_key, bpm, instrument(optional), genre(optional)}`
- 조건 프롬프트 구성 → top-p(0.9)+temperature(1.0) 샘플링, 2마디 도달 시 종료
- 전조: C 기준 생성 결과를 `target_key`로 전조
- 렌더: `pyfluidsynth` 사운드폰트 렌더 → 라우드니스 정규화 → 길이 트림

### 15.8 산출물/디렉터리

- `dataset_index.(csv|parquet)`
- `features/midi_tokens/*.jsonl`(조건+시퀀스)
- `checkpoints/midi_transformer.pt`
- `renderer/sf2/*`(사운드폰트) 및 렌더 스크립트
- (선택) `features/mel/*.npy`, `checkpoints/mel_diffusion.pt`

### 15.9 리소스/환경

- Python 의존성: `requirements.txt`(librosa, pretty_midi, torch, pyfluidsynth 등)
- 연산: 단일 GPU(12–24GB)로 수 시간~1일, mixed precision 권장

### 15.10 리스크 & 대응(학습 관점)

- 라벨-실데이터 키/BPM 불일치 → 사전 추정 교차검증, outlier 제외
- 드물거나 극단 템포 분포 → BPM 버킷화/가중 샘플링
- 과적합 → 조건 드랍, 전조 증강, early stopping

### 15.11 일정(세부)

- W1: 인덱싱/토큰화 파이프라인, train/val/test 분할, 품질 리포트
- W2: 모델 프로토타입 학습, 길이/키 보장 규칙 구현, 1차 추론 샘플
- W3: 하이퍼파라미터 튜닝, 랜더러 품질 보정(사운드폰트/페이드), 평가 자동화
- W4+: API 통합(`/generate`), 내부 베타, 문서화 및 샘플 데모 업데이트

# 가상환경 활성화

.\.venv\Scripts\Activate.ps1

# 추론 실행 (시드 고정, 출력 경로 지정)

python .\scripts\infer.py `  --checkpoint .\checkpoints\exp_20251028_ver2\midi_transformer.pt`
--vocab_path .\features\vocab.json `  --out_midi .\out\sample3.mid`
--bpm 120 --scale "D major" --instrument Drums --genre Dance `
--seed 80 --temperature 1.0 --top_p 0.9
