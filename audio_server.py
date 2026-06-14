"""
?뚯꽦 泥섎━ ?쒕쾭: Demucs (?뚯썝 遺꾨━) + pyannote (?붿옄 遺꾨━ + ?꾨쿋??

GPU 癒몄떊???꾩썙?먭퀬 遊??대씪?댁뼵?멸? HTTP濡??몄텧?섎㈃ ??

?ㅼ튂 (uv 沅뚯옣):
    uv pip install fastapi 'uvicorn[standard]' python-multipart torch torchaudio demucs pyannote.audio soundfile

?먮뒗 pip:
    pip install fastapi 'uvicorn[standard]' python-multipart torch torchaudio demucs pyannote.audio soundfile

?섍꼍蹂??
    HF_TOKEN: HuggingFace ?좏겙 (?꾩닔, pyannote 紐⑤뜽??
             https://huggingface.co/pyannote/speaker-diarization-3.1 ?숈쓽 ?꾩슂
    PYANNOTE_DEVICE: cuda / cpu (湲곕낯: ?먮룞 媛먯?)

?ㅽ뻾:
    HF_TOKEN=hf_xxxxx uvicorn audio_server:app --host 0.0.0.0 --port 8765
    ?먮뒗:
    HF_TOKEN=hf_xxxxx python audio_server.py

?붾뱶?ъ씤??
    GET  /health             - ?곹깭 + device ?뺣낫
    POST /separate           - audio(multipart) ??vocals.wav 諛붿씠?덈━
    POST /diarize            - audio(multipart) ???붿옄 援ш컙 + ?꾨쿋??JSON
    POST /process            - audio(multipart) ??蹂댁뺄 異붿텧 ???붿옄 遺꾨━ (??踰??몄텧濡???
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE = os.environ.get("PYANNOTE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="ai-watching-chzzk audio server")

# ?? 紐⑤뜽 lazy load ??????????????????????????????????????????????????????????
_demucs_model = None
_diarize_pipeline = None
_embed_inference = None


def _get_demucs():
    global _demucs_model
    if _demucs_model is None:
        from demucs.pretrained import get_model
        _demucs_model = get_model("htdemucs").to(DEVICE)
        _demucs_model.eval()
        print(f"[demucs] htdemucs 濡쒕뱶 ?꾨즺 (device={DEVICE})")
    return _demucs_model


def _get_diarize():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN ?섍꼍蹂?섍? ?ㅼ젙?섏? ?딆븯?듬땲??)
        _diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN,
        )
        if DEVICE == "cuda":
            _diarize_pipeline.to(torch.device("cuda"))
        print(f"[pyannote] speaker-diarization-3.1 濡쒕뱶 ?꾨즺 (device={DEVICE})")
    return _diarize_pipeline


def _get_embed():
    global _embed_inference
    if _embed_inference is None:
        from pyannote.audio import Inference, Model
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN ?섍꼍蹂?섍? ?ㅼ젙?섏? ?딆븯?듬땲??)
        m = Model.from_pretrained("pyannote/embedding", token=HF_TOKEN)
        _embed_inference = Inference(m, window="whole")
        print("[pyannote] embedding 濡쒕뱶 ?꾨즺")
    return _embed_inference


def _load_audio(path: str):
    """torchaudio ???soundfile濡?濡쒕뱶 (torchcodec/FFmpeg DLL ?섏〈???뚰뵾)."""
    import soundfile as sf
    data, sr = sf.read(path, always_2d=True, dtype="float32")  # (frames, channels)
    waveform = torch.from_numpy(data.T)  # ??(channels, frames)
    return waveform, sr


def _save_audio_to_bytes(waveform: torch.Tensor, sr: int) -> bytes:
    import io
    import soundfile as sf
    buf = io.BytesIO()
    data = waveform.numpy().T  # ??(frames, channels)
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ?? ?ы띁 ????????????????????????????????????????????????????????????????????
def _separate_vocals_sync(in_path: str) -> bytes:
    from demucs.apply import apply_model

    waveform, sr = _load_audio(in_path)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    model = _get_demucs()
    with torch.no_grad():
        sources = apply_model(model, waveform.unsqueeze(0).to(DEVICE))[0]

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx].cpu()

    return _save_audio_to_bytes(vocals, sr)


def _diarize_sync(in_path: str) -> dict:
    from pyannote.core import Segment as PyaSegment

    diarization = _get_diarize()(in_path)

    speaker_ranges: dict[str, list[tuple[float, float]]] = {}
    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_ranges.setdefault(speaker, []).append((turn.start, turn.end))
        segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})

    embed = _get_embed()
    embeddings: dict[str, list[float]] = {}
    for speaker, ranges in speaker_ranges.items():
        longest = max(ranges, key=lambda r: r[1] - r[0])
        try:
            emb = embed.crop({"audio": in_path}, PyaSegment(longest[0], longest[1]))
            embeddings[speaker] = np.asarray(emb).flatten().astype(float).tolist()
        except Exception as e:
            print(f"[diarize] {speaker} ?꾨쿋???ㅽ뙣: {e}")

    return {"segments": segments, "embeddings": embeddings}


# ?? ?붾뱶?ъ씤????????????????????????????????????????????????????????????????
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "cuda": torch.cuda.is_available()}


@app.post("/separate")
async def separate(audio: UploadFile = File(...)):
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        vocals = _separate_vocals_sync(path)
        return Response(content=vocals, media_type="audio/wav")
    finally:
        os.unlink(path)


@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return JSONResponse(_diarize_sync(path))
    finally:
        os.unlink(path)


@app.post("/process")
async def process(audio: UploadFile = File(...)):
    """?뚯썝 遺꾨━ + ?붿옄 遺꾨━瑜???踰덉뿉. vocals???묐떟 ?ㅻ뜑 X-Vocals-Available濡쒕쭔 ?뚮━怨?
    ?붿옄 遺꾨━??遺꾨━??vocals 湲곗??쇰줈 ?섑뻾??寃곌낵瑜?JSON?쇰줈 諛섑솚."""
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(data)
        in_path = f.name

    vocals_path = None
    try:
        vocals_bytes = _separate_vocals_sync(in_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(vocals_bytes)
            vocals_path = f.name
        result = _diarize_sync(vocals_path)
        return JSONResponse(result)
    finally:
        os.unlink(in_path)
        if vocals_path:
            os.unlink(vocals_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
