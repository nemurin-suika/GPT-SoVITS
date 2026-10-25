#!/bin/bash
# =============================================================================
# run-mac.sh — macOS 에서 GPT-SoVITS 추론 서버(api_v2)를 띄운다
# =============================================================================
#
#   ./run-mac.sh
#
# run.sh 와 다른 점:
#   - -c 로 tts_infer_mac.yaml 을 준다. 기본 tts_infer.yaml 은 device 가 cuda 라
#     맥에서는 모델을 올리는 순간 터진다.
#   - 바인딩을 127.0.0.1 로 둔다. 같은 기기의 백엔드만 쓰면 충분하고,
#     사내망에 TTS 를 열 이유가 없다.
#   - 저장소의 .venv 파이썬을 직접 쓴다. conda 활성화가 필요 없다.
#
# 환경 변수로 바꿀 수 있는 것:
#   GPTSOVITS_REF_AUDIO    참조 음성 파일 (기본 saki.wav)
#   GPTSOVITS_PROMPT_TEXT  그 음성이 말하는 내용
#   GPTSOVITS_HOST         바인딩 주소 (기본 127.0.0.1)
#   GPTSOVITS_PORT         포트 (기본 9880)
# =============================================================================

set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

REF_AUDIO="${GPTSOVITS_REF_AUDIO:-saki.wav}"
PROMPT_TEXT="${GPTSOVITS_PROMPT_TEXT:-내부가 뻑뻑해서 그런지 조금도 움직이질 않아.}"
HOST="${GPTSOVITS_HOST:-127.0.0.1}"
PORT="${GPTSOVITS_PORT:-9880}"
CONFIG="GPT_SoVITS/configs/tts_infer_mac.yaml"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3.9 || command -v python3)"
[ -n "$PY" ] || { echo "ERROR: 파이썬을 찾지 못했습니다. scripts/setup-mac.sh 를 먼저 돌리세요." >&2; exit 1; }

[ -f "$CONFIG" ] || { echo "ERROR: $CONFIG 가 없습니다." >&2; exit 1; }

if [ ! -f "$REF_AUDIO" ]; then
    echo "ERROR: 참조 음성이 없습니다 — $REF_AUDIO" >&2
    echo "       GPTSOVITS_REF_AUDIO 로 지정하거나 파일을 이 디렉토리에 두세요." >&2
    echo "       (저작권 자료라 저장소에 들어 있지 않습니다)" >&2
    exit 1
fi

echo "[GPT-SoVITS] $HOST:$PORT  참조음성=$REF_AUDIO  설정=$CONFIG"

exec "$PY" ./api_v2.py \
    -c "$CONFIG" \
    -a "$HOST" \
    -p "$PORT" \
    -ra "$REF_AUDIO" \
    -pt "$PROMPT_TEXT" \
    -pl ko
