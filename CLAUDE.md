# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GPT-SoVITS는 소량의 음성 샘플(5초~1분)로 고품질 TTS 및 음성 변환을 수행하는 시스템이다. 두 개의 핵심 모델로 구성된다:
- **GPT (T2S)**: 텍스트를 시맨틱 토큰으로 변환하는 AR(Autoregressive) 모델 (`GPT_SoVITS/AR/`)
- **SoVITS (VITS)**: 시맨틱 토큰을 오디오 파형으로 변환하는 VITS 기반 모델 (`GPT_SoVITS/module/`)

v3에서는 SoVITS 파인튜닝에 LoRA(`peft` 라이브러리)를 도입하여 `s2_train_v3_lora.py`로 학습한다.

## 환경 설정 및 실행 명령

### 설치 (macOS)
```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
brew install ffmpeg
pip install -r requirements.txt
```

### 설치 (Linux)
```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### WebUI 실행
```bash
python webui.py            # v2 모드 (기본)
python webui.py v1         # v1 모드
python webui.py en         # 영어 UI
```

### API 서버 실행
```bash
# v2 API (권장)
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

# v1 API (레거시)
python api.py -s <sovits_path> -g <gpt_path>
```

### 학습 (파인튜닝)
```bash
# Stage 1: GPT(T2S) 학습
python GPT_SoVITS/s1_train.py --config_file GPT_SoVITS/configs/s1longer.yaml

# Stage 2: SoVITS 학습 (v1/v2)
python GPT_SoVITS/s2_train.py

# Stage 2: SoVITS LoRA 학습 (v3)
python GPT_SoVITS/s2_train_v3_lora.py
```

### 데이터셋 준비 (순서대로 실행)
```bash
python GPT_SoVITS/prepare_datasets/1-get-text.py
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
python GPT_SoVITS/prepare_datasets/3-get-semantic.py
```

### Docker
```bash
docker compose -f docker-compose.yaml up -d
```

## 아키텍처 및 주요 모듈

### 추론 파이프라인
`GPT_SoVITS/TTS_infer_pack/TTS.py`가 추론의 핵심 진입점이다. `api_v2.py`는 이 모듈을 감싸는 FastAPI 서버로, `/tts` 엔드포인트에서 스트리밍 WAV를 반환한다.

```
텍스트 입력
  → TextPreprocessor (언어 감지 + G2P + BERT 임베딩)
  → T2S AR 모델 (시맨틱 토큰 생성)
  → CNHubert (레퍼런스 오디오 인코딩)
  → SynthesizerTrn/V3 (VITS 디코딩 → WAV)
```

### 모듈 구조
| 경로 | 역할 |
|------|------|
| `GPT_SoVITS/AR/` | GPT(T2S) 모델 정의, 학습 모듈 |
| `GPT_SoVITS/module/` | VITS 모델(`models.py`), 손실 함수, 데이터 유틸 |
| `GPT_SoVITS/TTS_infer_pack/` | 추론 래퍼, 텍스트 전처리, 문장 분할 |
| `GPT_SoVITS/text/` | 언어별 G2P(중국어·일본어·한국어·영어·광동어) |
| `GPT_SoVITS/feature_extractor/` | CNHubert, Whisper 인코더 |
| `GPT_SoVITS/configs/` | `tts_infer.yaml`(추론 모델 경로), `s1*.yaml`(GPT 학습) |
| `tools/` | 음성 슬라이싱, UVR5 보컬 분리, ASR, 노이즈 제거 |
| `config.py` | 전역 설정(포트, 디바이스, 모델 경로) |

### 모델 버전 구분
- **v1/v2**: `SynthesizerTrn`, `s2_train.py`
- **v3**: `SynthesizerTrnV3` + LoRA (`s2_train_v3_lora.py`), 사전학습 가중치는 `GPT_SoVITS/pretrained_models/s2Gv3.pth`
- `weight.json`에 버전별 기본 가중치 경로가 정의되어 있다.

### 다국어 지원
언어 코드: `zh`(중국어), `ja`(일본어), `en`(영어), `ko`(한국어), `yue`(광동어)  
`GPT_SoVITS/text/cleaner.py`가 언어별 G2P 모듈을 디스패치한다.

### 포트 구성 (config.py)
- `9880`: API 서버
- `9874`: WebUI 메인
- `9873`: UVR5
- `9872`: TTS 추론 WebUI
- `9871`: 자막 수정 WebUI

## 데이터셋 포맷

학습용 `.list` 파일 형식:
```
vocal_path|speaker_name|language|text
```
예: `D:\GPT-SoVITS\data\voice.wav|speaker1|zh|你好，世界。`

## 사전학습 모델 배치
- HuBERT: `GPT_SoVITS/pretrained_models/chinese-hubert-base`
- BERT: `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`
- G2PW: `GPT_SoVITS/text/G2PWModel` (중국어 TTS 필수)
- UVR5 모델: `tools/uvr5/uvr5_weights/`
- ASR 모델: `tools/asr/models/`

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `is_half` | `True` | FP16 추론 여부 |
| `is_share` | `False` | Gradio 공유 링크 |
| `language` | `Auto` | UI 언어 |
| `CUDA_VISIBLE_DEVICES` | 자동 | GPU 지정 |
