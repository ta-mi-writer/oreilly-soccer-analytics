import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ==========================================
# 1. 依存ライブラリの自動インストール
# ==========================================
def install_dependencies():
    """Demucs と WhisperX を自動インストール"""
    print("[1/5] 依存ライブラリ (Demucs, WhisperX) をチェック・インストール中...")

    # Demucs
    try:
        import demucs
    except ImportError:
        print("-> Demucs をインストールしています...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "demucs"], check=True
        )

    # WhisperX
    try:
        import whisperx
    except ImportError:
        print("-> WhisperX をインストールしています...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "git+https://github.com/m-bain/whisperx.git",
            ],
            check=True,
        )

    print("[成功] 必要なライブラリの準備が完了しました。")


# ==========================================
# 2. 設定エリア
# ==========================================
# 入力動画パス
# Job 1 の出力が自動的にインポートされるため、ここから読み込めます
INPUT_VIDEO_PATH = "/kaggle/input/trimming/trimmed_videos/sfw.mp4"

# 出力フォルダ（解析レポートとクリーン音声を保存）
OUTPUT_DIR = Path("/kaggle/working/ref_audio_analysis")
TEMP_DIR = Path("/kaggle/working/temp_processing")

# WhisperX のモデルサイズ（"small", "medium", "large-v3" など。GPUなら "large-v3" や "medium" が高精度です）
WHISPER_MODEL_SIZE = "medium"

# 【任意】Hugging Face のアクセストークン（話者分離 Diarization を完全自動で行いたい場合に入力）
# 空文字 "" のままでも、高精度な「時間付き文字起こしテキスト」は出力されます！
HF_TOKEN = "Hugging Face トークン"


# ==========================================
# 3. 処理ロジック
# ==========================================
def format_timestamp(seconds: float) -> str:
    """秒数を hh:mm:ss.mmm フォーマットに変換"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def extract_and_clean_audio(video_path: str, temp_dir: Path) -> Path:
    """動画から全編音声を抽出し、Demucs で BGM・ノイズを除去"""
    print("\n[2/5] 動画から音声を抽出し、BGM・ノイズを除去中 (Demucs)...")

    raw_wav = temp_dir / "raw_full_audio.wav"

    # 音声の全編抽出
    cmd_extract = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(raw_wav),
    ]
    subprocess.run(
        cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Demucs で BGM 除去
    demucs_out = temp_dir / "demucs_out"
    cmd_demucs = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "-o",
        str(demucs_out),
        str(raw_wav),
    ]
    subprocess.run(cmd_demucs, check=True)

    vocal_wav = demucs_out / "htdemucs" / "raw_full_audio" / "vocals.wav"
    print(f"[成功] BGM除去済みのクリア音声を作成しました: {vocal_wav}")
    return vocal_wav


def run_whisperx(audio_path: Path, output_dir: Path):
    """WhisperX で高精度文字起こし ＋ タイムスタンプ ＋ (話者分離) を実行"""
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if device == "cuda" else 4
    compute_type = "float16" if device == "cuda" else "int8"

    print(
        f"\n[3/5] WhisperX による文字起こし＆タイムスタンプ解析中 (デバイス: {device})..."
    )

    # 1. Transcribe (日本語指定)
    model = whisperx.load_model(
        WHISPER_MODEL_SIZE, device, compute_type=compute_type, language="ja"
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align (タイムスタンプ精度の向上)
    print("\n[4/5] タイムスタンプのアライメント調整中...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 3. Diarize (話者分離: HF_TOKEN がある場合)
    if HF_TOKEN:
        print("\n[話者分離] HuggingFace トークンを使用して話者識別中 (PyAnnote)...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN, device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("[成功] 話者識別が完了しました！")
        except Exception as e:
            print(f"[警告] 話者識別スキップ (エラー: {e})")

    # 4. レポートテキストとメタデータの保存
    print("\n[5/5] 解析レポートを生成中...")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "transcription_report.txt"
    json_file = output_dir / "transcription_data.json"

    lines = []
    lines.append("=================================================================")
    lines.append("  Job 2: 音声解析＆タイムスタンプ・文字起こしレポート")
    lines.append("=================================================================\n")

    for seg in result["segments"]:
        start_str = format_timestamp(seg["start"])
        end_str = format_timestamp(seg["end"])
        duration = seg["end"] - seg["start"]
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        text = seg["text"].strip()

        line = f"[{start_str} -> {end_str}] ({duration:4.1f}秒) [{speaker}]: {text}"
        lines.append(line)

    report_text = "\n".join(lines)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)

    # 人間が確認できるようにコンソールにも一部出力
    print("\n--- 解析レポートの一部サンプル ---")
    for l in lines[5:20]:
        print(l)
    print("-----------------------------------")
    print(f"\n[完成] 全体レポートを保存しました: {report_file}")


def main():
    print("=== Job 2: BGM除去 ＋ WhisperX 話者分離＆タイムスタンプ解析 ===")

    install_dependencies()

    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"[エラー] 入力動画が見つかりません: {INPUT_VIDEO_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 動画からBGM除去済みクリア音声を抽出
    clean_vocal_wav = extract_and_clean_audio(INPUT_VIDEO_PATH, TEMP_DIR)

    # BGM除去済みクリア音声も成果物としてコピー保存しておく
    shutil.copy(clean_vocal_wav, OUTPUT_DIR / "clean_vocals_full.wav")

    # 2. WhisperX で文字起こし ＋ タイムスタンプ解析
    run_whisperx(clean_vocal_wav, OUTPUT_DIR)

    # クリーンアップ
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print("\n==========================================")
    print(" Job 2 の処理が正常終了しました！")
    print(f" 成果物フォルダ: {OUTPUT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()
