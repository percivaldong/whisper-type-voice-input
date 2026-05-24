import os
import sys
import tempfile
import wave
import threading
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
os.environ["HF_HOME"] = str(PROJECT_DIR / "models")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from pynput import keyboard
from pynput.keyboard import Key, Controller as KBController
import pyperclip
import zhconv


MODEL_SIZE = "small"
MODEL_REPO = "Systran/faster-whisper-small"
MODEL_LOCAL = PROJECT_DIR / "models" / "Systran" / "faster-whisper-small"
LANGUAGE = "zh"


def _try_download_hf(endpoint=None):
    """通过 HuggingFace 下载，可选镜像端点。"""
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        safe_print(f"[whisper-type] 尝试 HuggingFace 镜像 ({endpoint}) ...")
    else:
        safe_print("[whisper-type] 尝试 HuggingFace 官方源 ...")

    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO, cache_dir=str(PROJECT_DIR / "models"))


def _try_download_modelscope():
    """通过 ModelScope 下载。"""
    safe_print("[whisper-type] 尝试 ModelScope 下载 ...")
    from modelscope import snapshot_download

    return snapshot_download(MODEL_REPO, cache_dir=str(PROJECT_DIR / "models"))


def ensure_model():
    """确保模型已下载到本地。自动检测网络并选择可用下载源。"""
    if MODEL_LOCAL.exists() and list(MODEL_LOCAL.iterdir()):
        return str(MODEL_LOCAL)

    safe_print(f"[whisper-type] 模型未缓存，开始下载 ({MODEL_SIZE}, ~460MB) ...")

    for attempt in [
        lambda: _try_download_hf(),                         # HF 官方源
        lambda: _try_download_hf("https://hf-mirror.com"),  # HF 国内镜像
        _try_download_modelscope,                            # ModelScope
    ]:
        try:
            model_dir = attempt()
            safe_print("[whisper-type] 模型下载完成")
            return model_dir
        except Exception as e:
            safe_print(f"[whisper-type] 失败 ({type(e).__name__}), 切换下一源...")

    raise RuntimeError(
        "无法下载语音识别模型，请检查网络连接。\n"
        "可手动下载模型到: " + str(MODEL_LOCAL)
    )

TRIGGER_KEY = Key.ctrl_r
SAMPLE_RATE = 16000
CHANNELS = 1

recording = False
frames = []
current_key = None
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def load_model():
    model_path = ensure_model()
    safe_print(f"[whisper-type] 加载 Whisper 模型 ({MODEL_SIZE})...")
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    safe_print("[whisper-type] 模型就绪，按住右 Ctrl 开始说话")
    return model


def transcribe(model, audio_path):
    segments, info = model.transcribe(audio_path, language=LANGUAGE, beam_size=5)
    text = " ".join(seg.text for seg in segments)
    text = zhconv.convert(text, "zh-cn")
    return text.strip()


def type_text(text):
    if not text:
        return
    original = pyperclip.paste()
    pyperclip.copy(text)
    kb = KBController()
    kb.press(Key.ctrl)
    kb.press("v")
    kb.release("v")
    kb.release(Key.ctrl)
    time.sleep(0.1)
    pyperclip.copy(original)


def save_wav(frames, path):
    audio = np.concatenate(frames, axis=0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())


def on_press(key):
    global recording, frames, current_key
    if key == TRIGGER_KEY:
        if not recording:
            recording = True
            frames = []
            current_key = key
            safe_print("[whisper-type] ● 录音中...")

def on_release(key):
    global recording, frames, current_key
    if key == TRIGGER_KEY and recording:
        recording = False
        safe_print("[whisper-type] ○ 转写中...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        save_wav(frames, tmp_path)
        text = transcribe(model, tmp_path)
        Path(tmp_path).unlink()
        if text:
            safe_print(f"[whisper-type] → {text}")
            type_text(text)
        else:
            safe_print("[whisper-type] (未识别到语音)")
    elif recording and key == current_key:
        pass


def audio_callback(indata, frame_count, time_info, status):
    if recording:
        frames.append(indata.copy())


def main():
    global model
    model = load_model()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    safe_print("[whisper-type] 退出按 Ctrl+C")
    try:
        listener.join()
    except KeyboardInterrupt:
        safe_print("\n[whisper-type] 已退出")


if __name__ == "__main__":
    main()
