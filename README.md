## Mumming to Music (Handmade)

간단한 MIDI 토크나이저와 Transformer 디코더로 2마디 길이의 루프를 생성하는 프로젝트입니다. 학습은 `dataset_index.csv`를 기반으로 진행되며, 실제 대용량 데이터(`dataset/`)는 Git에서 무시됩니다.

- 원격 저장소: [hongci9999/mumming_to_music_handmade](https://github.com/hongci9999/mumming_to_music_handmade.git)

### 폴더 구조
- `scripts/`: 데이터 인덱싱, 학습, 추론 스크립트
- `services/`: 데이터셋, 모델, 토크나이저 로직
- `features/`: 토큰 사전(`vocab.json`)
- `checkpoints/`: 학습 체크포인트(`midi_transformer.pt`)
- `out/`: 생성 결과 MIDI

### 환경 준비
1) Python 3.10+ 권장, 가상환경 생성 및 활성화
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

2) 의존성 설치
```bash
pip install -r requirements.txt
```

요구 패키지(발췌): torch, pandas, pretty_midi, mido, tqdm, fastapi(옵션), uvicorn(옵션)

### 데이터 인덱싱
`dataset/label` 아래 JSON 메타 정보를 읽어 `dataset_index.csv`를 생성합니다.
```bash
python scripts/index_dataset.py --output dataset_index.csv
```

`.gitignore`에 `dataset/`이 포함되어 있어 실제 데이터 파일은 커밋되지 않습니다.

### 학습(Training)
```bash
python scripts/train.py \
  --index_csv dataset_index.csv \
  --vocab_path features/vocab.json \
  --batch_size 32 --epochs 5 --lr 3e-4 \
  --d_model 512 --n_head 8 --num_layers 8 \
  --max_seq_len 1024 \
  --save_dir checkpoints
```
최고 성능의 체크포인트는 `checkpoints/midi_transformer.pt`로 저장됩니다.

### 추론(Inference)
```bash
python scripts/infer.py \
  --checkpoint checkpoints/midi_transformer.pt \
  --vocab_path features/vocab.json \
  --out_midi out/sample.mid \
  --bpm 120 --scale "C major" --instrument Piano --genre Dance
```
생성된 MIDI는 `out/sample.mid`에 저장됩니다.

### 참고
- 윈도우에서 FFmpeg를 사용하는 패키지(ffmpeg-python 등)를 쓸 경우, 시스템에 FFmpeg 바이너리가 설치/경로 등록되어 있어야 합니다.
- GPU가 있을 경우 PyTorch CUDA 버전에 맞춰 설치하는 것을 권장합니다.


