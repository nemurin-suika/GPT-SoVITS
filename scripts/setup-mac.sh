#!/bin/bash
# =============================================================================
# setup-mac.sh — macOS(Apple Silicon)에 GPT-SoVITS 추론 서버만 깔아 준다
# =============================================================================
#
#   cd <이 저장소>
#   ./scripts/setup-mac.sh
#
# 하는 일:
#   1. 환경 점검 (macOS / brew / ffmpeg / python@3.9)
#   2. .venv 생성 (python3.9 -m venv)
#   3. requirements.txt 설치
#   4. NLTK 데이터 다운로드
#   5. 사전학습 모델 다운로드 (한국어 추론에 필요한 것만, 약 1.3G)
#   6. 점검 및 남은 안내
#
# 안 하는 일 (일부러):
#   - 참조 음성 설치. 저작권 자료라 저장소에 넣지 않는다. 손으로 옮긴다.
#   - 학습용 자산(UVR5 / ASR / G2PW / BigVGAN) 설치. 추론에 필요 없고 수 기가다.
#   - 상주 등록. 기동은 ./run-mac.sh 또는 백엔드의 start-all.sh 가 맡는다.
#
# 왜 python@3.9 인가:
#   requirements.txt 가 numpy==1.23.4 / numba==0.56.4 로 고정돼 있다. 이 조합은
#   3.10 이상에서 휠이 없어 소스 빌드로 넘어가고 거기서 깨진다. 3.9 를 쓴다.
#
# 옵션:
#   --yes          되묻지 않고 전부 진행 (brew 설치 포함)
#   --skip-models  모델 다운로드를 건너뛴다 (이미 옮겨 뒀을 때)
#   --skip-deps    가상환경·pip 설치를 건너뛴다 (모델만 받고 싶을 때)
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSUME_YES=0
SKIP_MODELS=0
SKIP_DEPS=0

PY_FORMULA="python@3.9"
PY_BIN=""
VENV="$REPO_ROOT/.venv"

HF_BASE="https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"
LID_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_DIR="$REPO_ROOT/GPT_SoVITS/pretrained_models"

# 한국어 추론 경로에서 실제로 쓰이는 것만 고른 목록.
#   상대경로|원격경로|최소크기(바이트)
# 최소크기는 "받다 끊긴 파일"을 걸러내기 위한 것이다. 정확한 크기가 아니라
# 이보다 작으면 실패로 보는 하한선이다.
MODEL_FILES=(
  "chinese-hubert-base/config.json|$HF_BASE/chinese-hubert-base/config.json|1000"
  "chinese-hubert-base/preprocessor_config.json|$HF_BASE/chinese-hubert-base/preprocessor_config.json|100"
  "chinese-hubert-base/pytorch_model.bin|$HF_BASE/chinese-hubert-base/pytorch_model.bin|180000000"
  "chinese-roberta-wwm-ext-large/config.json|$HF_BASE/chinese-roberta-wwm-ext-large/config.json|500"
  "chinese-roberta-wwm-ext-large/pytorch_model.bin|$HF_BASE/chinese-roberta-wwm-ext-large/pytorch_model.bin|640000000"
  "chinese-roberta-wwm-ext-large/tokenizer.json|$HF_BASE/chinese-roberta-wwm-ext-large/tokenizer.json|100000"
  "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt|$HF_BASE/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt|150000000"
  "gsv-v2final-pretrained/s2G2333k.pth|$HF_BASE/gsv-v2final-pretrained/s2G2333k.pth|100000000"
  "fast_langdetect/lid.176.bin|$LID_URL|125000000"
)

NLTK_PACKAGES=(averaged_perceptron_tagger_eng averaged_perceptron_tagger cmudict punkt_tab)

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y)      ASSUME_YES=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --skip-deps)   SKIP_DEPS=1; shift ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "모르는 옵션: $1" >&2; exit 1 ;;
    esac
done

STEP=0
step()  { STEP=$((STEP + 1)); echo; echo "── [$STEP] $*"; }
info()  { echo "   $*"; }
warn()  { echo "   ! $*" >&2; }
die()   { echo; echo "ERROR: $*" >&2; exit 1; }

confirm() {
    [ "$ASSUME_YES" = "1" ] && return 0
    if [ ! -t 0 ]; then
        warn "비대화 실행이라 '아니오'로 처리합니다: $1"
        return 1
    fi
    printf "   %s [y/N] " "$1"
    read -r _ans
    case "$_ans" in [yY]|[yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

human() {
    # 바이트를 사람이 읽는 단위로.
    awk -v b="$1" 'BEGIN{
        split("B KB MB GB",u," "); i=1
        while (b>=1024 && i<4) { b/=1024; i++ }
        printf "%.1f%s", b, u[i]
    }'
}

echo "============================================================"
echo " GPT-SoVITS 추론 서버 설치 (macOS)"
echo "============================================================"
info "저장소: $REPO_ROOT"

# ── 1. 환경 점검 ──────────────────────────────────────────────────────────────
step "환경 점검"

[ "$(uname -s)" = "Darwin" ] || die "macOS 전용 스크립트입니다."
info "아키텍처: $(uname -m)"

command -v curl >/dev/null 2>&1 || die "curl 이 없습니다."

if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew 가 없습니다. ffmpeg 와 python@3.9 를 직접 준비해야 합니다."
else
    # ffmpeg CLI 는 tools/my_utils.py 가 서브프로세스로 호출한다. 없으면
    # 참조 음성을 읽는 단계에서 바로 실패한다. 루트의 ffmpeg.exe 는 윈도우용.
    if command -v ffmpeg >/dev/null 2>&1; then
        info "OK   ffmpeg ($(command -v ffmpeg))"
    elif confirm "ffmpeg 를 brew 로 설치할까요?"; then
        brew install ffmpeg || warn "ffmpeg 설치 실패"
    else
        warn "ffmpeg 없이는 참조 음성을 읽지 못합니다."
    fi

    if [ -x "$(brew --prefix)/opt/$PY_FORMULA/bin/python3.9" ]; then
        info "OK   $PY_FORMULA"
    elif confirm "$PY_FORMULA 를 brew 로 설치할까요?"; then
        brew install "$PY_FORMULA" || warn "$PY_FORMULA 설치 실패"
    else
        warn "$PY_FORMULA 가 없으면 의존성 설치가 깨집니다."
    fi
fi

# python3.9 실행 파일 찾기 (brew 경로 → PATH 순)
for cand in "$(brew --prefix 2>/dev/null)/opt/$PY_FORMULA/bin/python3.9" \
            "/opt/homebrew/opt/$PY_FORMULA/bin/python3.9" \
            "$(command -v python3.9 2>/dev/null)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then PY_BIN="$cand"; break; fi
done
[ -n "$PY_BIN" ] && info "python3.9: $PY_BIN" || warn "python3.9 를 찾지 못했습니다."

# ── 2. 가상환경 ───────────────────────────────────────────────────────────────
if [ "$SKIP_DEPS" = "1" ]; then
    step "가상환경 (--skip-deps 지정됨 — 건너뜀)"
else
    step "가상환경 준비"
    if [ -x "$VENV/bin/python" ]; then
        info "이미 있음: $VENV ($("$VENV/bin/python" -V 2>&1))"
    else
        [ -n "$PY_BIN" ] || die "python3.9 가 없어 가상환경을 만들 수 없습니다."
        "$PY_BIN" -m venv "$VENV" || die "가상환경 생성 실패"
        info "생성 완료: $VENV"
    fi

    # ── 3. 의존성 ─────────────────────────────────────────────────────────────
    step "의존성 설치 (requirements.txt)"
    info "처음이면 십 분 이상 걸립니다. torch 와 numba 빌드가 느립니다."
    "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null \
        || warn "pip 업그레이드 실패"
    "$VENV/bin/python" -m pip install -r "$REPO_ROOT/requirements.txt" \
        || warn "requirements 설치 실패 — 위 로그를 확인하세요."

    # requirements.txt 에 적혀 있지만 빠지기 쉬운 항목. 한국어 문장 안의 영어
    # 단어를 한글 발음으로 바꾸는 폴백에서 쓴다. 없어도 돌지만 품질이 달라진다.
    if ! "$VENV/bin/python" -c "import hangulize" >/dev/null 2>&1; then
        "$VENV/bin/python" -m pip install hangulize >/dev/null 2>&1 \
            && info "hangulize 추가 설치" \
            || warn "hangulize 설치 실패 (영어 단어 발음 폴백 품질만 영향)"
    fi

    # ── 4. NLTK 데이터 ────────────────────────────────────────────────────────
    step "NLTK 데이터"
    # 한국어 문장이라도 안에 영어 단어가 섞이면 english.py 의 품사 태거를
    # 지나간다. 데이터가 없으면 첫 추론에서 LookupError 로 멈춘다.
    for pkg in "${NLTK_PACKAGES[@]}"; do
        if "$VENV/bin/python" -c "
import nltk, sys
for p in ('taggers/$pkg', 'corpora/$pkg', 'tokenizers/$pkg'):
    try:
        nltk.data.find(p); sys.exit(0)
    except LookupError:
        pass
sys.exit(1)" >/dev/null 2>&1; then
            info "OK   $pkg"
        else
            "$VENV/bin/python" -c "import nltk; nltk.download('$pkg')" >/dev/null 2>&1 \
                && info "받음 $pkg" || warn "$pkg 다운로드 실패"
        fi
    done
fi

# ── 5. 사전학습 모델 ──────────────────────────────────────────────────────────
if [ "$SKIP_MODELS" = "1" ]; then
    step "사전학습 모델 (--skip-models 지정됨 — 건너뜀)"
else
    step "사전학습 모델 다운로드 (한국어 추론분만, 약 1.3G)"
    info "학습용 자산(UVR5 / ASR / G2PW / BigVGAN / v3)은 받지 않습니다."
    for entry in "${MODEL_FILES[@]}"; do
        rel="${entry%%|*}"
        rest="${entry#*|}"
        url="${rest%%|*}"
        minsize="${rest##*|}"
        dest="$MODEL_DIR/$rel"
        mkdir -p "$(dirname "$dest")"

        if [ -f "$dest" ]; then
            cur=$(/usr/bin/stat -f%z "$dest" 2>/dev/null || echo 0)
            if [ "$cur" -ge "$minsize" ]; then
                info "OK   $rel ($(human "$cur"))"
                continue
            fi
            warn "$rel 이 너무 작습니다($(human "$cur")) → 다시 받습니다."
            rm -f "$dest"
        fi

        info "받는 중 $rel"
        # -C - 로 중단된 다운로드를 이어받는다. 큰 파일이라 중요하다.
        if curl -fL --retry 3 --retry-delay 2 -C - -o "$dest" "$url"; then
            cur=$(/usr/bin/stat -f%z "$dest" 2>/dev/null || echo 0)
            if [ "$cur" -lt "$minsize" ]; then
                warn "$rel 크기가 모자랍니다($(human "$cur")) — 다시 실행해 주세요."
            else
                info "완료 $rel ($(human "$cur"))"
            fi
        else
            warn "$rel 다운로드 실패 — 다시 실행하면 이어받습니다."
        fi
    done
fi

# ── 6. 점검 ───────────────────────────────────────────────────────────────────
step "점검"

MAC_YAML="$REPO_ROOT/GPT_SoVITS/configs/tts_infer_mac.yaml"
if [ -f "$MAC_YAML" ]; then
    if grep -q 'device: mps' "$MAC_YAML"; then
        info "OK   tts_infer_mac.yaml (device: mps, version: v2)"
    else
        warn "tts_infer_mac.yaml 의 device 가 mps 가 아닙니다. 확인하세요."
    fi
else
    warn "$MAC_YAML 이 없습니다. 기본 tts_infer.yaml 은 device 가 cuda 라 맥에서 실패합니다."
fi

# 참조 음성. 저작권 자료라 저장소에 없다.
REF_FOUND=""
for w in "$REPO_ROOT"/*.wav; do
    [ -f "$w" ] || continue
    REF_FOUND="$REF_FOUND $(basename "$w")"
done
if [ -n "$REF_FOUND" ]; then
    info "참조 음성 후보:$REF_FOUND"
else
    warn "저장소 루트에 wav 가 없습니다. 참조 음성을 손으로 옮기세요."
fi

echo
echo "============================================================"
echo " 설치 절차 끝"
echo "============================================================"
cat <<GUIDE

손으로 할 일:
  1) 참조 음성 wav 를 이 저장소 루트에 둔다. mp3 는 안 되므로 변환한다.
       ffmpeg -i <원본>.mp3 -c:a pcm_s16le <이름>.wav
     3 초에서 10 초 사이, 잡음과 배경음이 없는 토막이 좋다.
  2) 그 음성이 실제로 말하는 내용을 그대로 적어 둔다. 기동할 때
     -pt 로 넘긴다. 내용이 틀리면 발음이 뭉개진다.

띄우기:
  ./run-mac.sh
  확인:  curl -s -o out.wav -G 'http://127.0.0.1:9880/tts' \\
           --data-urlencode 'text=안녕. 잘 들리니.' \\
           --data-urlencode 'text_lang=ko' \\
           --data-urlencode 'ref_audio_path=<참조음성>.wav' \\
           --data-urlencode 'prompt_text=<참조음성 내용>' \\
           --data-urlencode 'prompt_lang=ko'

첫 추론은 모델을 올리는 시간이 붙어 느립니다. 두 번째부터 빨라집니다.

GUIDE
