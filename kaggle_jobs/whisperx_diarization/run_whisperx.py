import json
import os
import subprocess
import sys
from pathlib import Path


# ==========================================
# 1. 自動依存ライブラリインストール
# ==========================================
def install_dependencies():
    print("[1/4] 依存ライブラリ (WhisperX) をチェック・インストール中...")
    try:
        import whisperx
    except ImportError:
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
    print("✅ WhisperX の準備が完了しました。")


# ==========================================
# 2. 設定エリア
# ==========================================
TARGET_VIDEO_NAME = "sfw-fin.mp4"
DEFAULT_INPUT_VIDEO_PATH = "/kaggle/input/trimming/trimmed_videos/sfw-fin.mp4"

# ★ご自身の Hugging Face トークン（hf_...）を入力してください
HF_TOKEN = "Hugging Face トークン"

WHISPER_MODEL_SIZE = "large-v3"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 2

OUTPUT_DIR = Path("/kaggle/working/whisperx_results")


# ==========================================
# 3. 処理ロジック
# ==========================================
def find_input_video(filename: str, default_path: str) -> str:
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            found_path = os.path.join(root, filename)
            print(f"🔍 [自動検出] 入力動画を発見しました: {found_path}")
            return found_path
    print(f"⚠️ デフォルトパスを使用します: {default_path}")
    return default_path


def format_timestamp(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def reconstruct_speaker_segments(result_segments):
    """単語(words)情報から話者の切り替わりを検知し、会話ごとにテキストとタイムスタンプを再分割する"""
    reconstructed = []
    current_speaker = None
    current_words = []
    current_start = None
    current_end = None

    for seg in result_segments:
        words = seg.get("words", [])
        if not words:
            # 単語データが無い場合は通常のセグメントとして処理
            reconstructed.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg.get("speaker", "SPEAKER_UNKNOWN"),
                    "text": seg["text"].strip(),
                }
            )
            continue

        for w in words:
            word_str = w.get("word", "")
            speaker = w.get("speaker", seg.get("speaker", "SPEAKER_UNKNOWN"))
            start = w.get("start", current_end if current_end else seg["start"])
            end = w.get("end", start)

            if current_speaker is None:
                current_speaker = speaker
                current_start = start

            # 話者が変わったら、前の話者のセグメントを確定して保存
            if speaker != current_speaker and current_words:
                text_str = "".join([item["word"] for item in current_words]).strip()
                if text_str:
                    reconstructed.append(
                        {
                            "start": current_start,
                            "end": current_end,
                            "speaker": current_speaker,
                            "text": text_str,
                        }
                    )
                current_speaker = speaker
                current_start = start
                current_words = []

            current_words.append(w)
            current_end = end

    # 残った最後の発話を追加
    if current_words:
        text_str = "".join([item["word"] for item in current_words]).strip()
        if text_str:
            reconstructed.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "speaker": current_speaker,
                    "text": text_str,
                }
            )

    return reconstructed


def run_whisperx_pipeline(video_path: str, output_dir: Path):
    import torch
    import whisperx
    from huggingface_hub import login
    from whisperx.diarize import DiarizationPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if device == "cuda" else 4
    compute_type = "float16" if device == "cuda" else "int8"

    print(
        f"\n[2/4] 動画ファイルから音声をロード＆高精度文字起こし中 ({WHISPER_MODEL_SIZE} / Device: {device})..."
    )

    # 1. Transcribe
    model = whisperx.load_model(
        WHISPER_MODEL_SIZE, device, compute_type=compute_type, language="ja"
    )
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align
    print("\n[3/4] タイムスタンプのアライメント調整中...")
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

    # 3. 話者分離 (Diarization)
    print(
        f"\n[4/4] 話者分離を実行中 (話者数: {MIN_SPEAKERS}〜{MAX_SPEAKERS} 人固定)..."
    )
    if HF_TOKEN and not HF_TOKEN.startswith("YOUR_"):
        try:
            # Hugging Face Hub に事前ログイン認証
            login(token=HF_TOKEN)

            # 新旧バージョンの引数名の差分に対応（フォールバック処理）
            try:
                diarize_model = DiarizationPipeline(
                    use_auth_token=HF_TOKEN, device=device
                )
            except TypeError:
                try:
                    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
                except TypeError:
                    diarize_model = DiarizationPipeline(device=device)

            diarize_segments = diarize_model(
                audio, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("✅ 話者識別・単語への割当が完了しました！")
        except Exception as e:
            print(f"\n❌ 話者識別エラーが発生しました:\n{e}", file=sys.stderr)
            print(
                "⚠️ Hugging Faceのモデル規約への同意状況（3箇所）やトークンの有効性をご確認ください。",
                file=sys.stderr,
            )
            raise e
    else:
        print(
            "⚠️ HF_TOKEN が設定されていないか初期値のままのため、話者分離をスキップします。"
        )

    # 4. 話者の交代ポイントでテキストとタイムスタンプを再分割・再構成
    fine_segments = reconstruct_speaker_segments(result["segments"])

    # 5. レポート出力と 5〜10秒 発話の自動WAV切り出し保存
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "transcription_report.txt"
    json_file = output_dir / "transcription_data.json"
    clips_dir = output_dir / "ref_audio_candidates"
    clips_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=================================================================")
    lines.append(
        f"  WhisperX ({WHISPER_MODEL_SIZE}) 話者分離会話レポート (話者交代ごとに再分割済)"
    )
    lines.append("=================================================================\n")

    saved_clip_count = 0
    for i, seg in enumerate(fine_segments):
        start_sec = seg["start"]
        end_sec = seg["end"]
        duration = end_sec - start_sec
        speaker = seg["speaker"]
        text = seg["text"]

        start_str = format_timestamp(start_sec)
        end_str = format_timestamp(end_sec)

        line = f"[{start_str} -> {end_str}] ({duration:4.1f}秒) [{speaker}]: {text}"
        lines.append(line)

        # 5秒〜10秒の発話を自動切り出し保存
        if 5.0 <= duration <= 10.0:
            out_clip_wav = clips_dir / f"{speaker}_clip_{i}_{start_sec:.1f}s.wav"
            cmd_slice = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_sec),
                "-to",
                str(end_sec),
                "-i",
                video_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(out_clip_wav),
            ]
            subprocess.run(
                cmd_slice, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            saved_clip_count += 1

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)

    print("\n--- 再分割後の会話レポート（一部サンプル） ---")
    for l in lines[5:25]:
        print(l)
    print("---------------------------------------------")
    print(f"\n✅ 会話レポート保存: {report_file}")
    print(f"✅ 5〜10秒のWAV候補ファイル ({saved_clip_count}件): {clips_dir}")


def main():
    print("=== WhisperX Large 話者分離 ＆ 会話再構成処理 ===")
    install_dependencies()

    input_video_path = find_input_video(TARGET_VIDEO_NAME, DEFAULT_INPUT_VIDEO_PATH)
    if not os.path.exists(input_video_path):
        print(f"❌ エラー: 入力動画が見つかりません: {input_video_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_whisperx_pipeline(input_video_path, OUTPUT_DIR)

    print("\n==========================================")
    print(" 全処理が正常に完了しました！")
    print("==========================================")


if __name__ == "__main__":
    main()
