# Kaggle Jobs: WhisperX Diarization - Complete File Contents

このドキュメントは、`kaggle_jobs/whisperx_diarization` フォルダに含まれるすべてのファイルの内容を1つのマークダウンファイルにまとめたものです。このファイルをプロンプトとして使用することを目的とします。

## フォルダ構成

```
kaggle_jobs/whisperx_diarization/
├── kernel-metadata.json
├── run_whisperx.py
└── output/
    ├── whisperx-large-diarization.log
    └── whisperx_results/
        ├── transcription_data.json
        └── transcription_report.txt
```

---
## kernel-metadata.json

```json
{
  "id": "masashiki/whisperx-large-diarization",
  "title": "WhisperX Large Diarization",
  "code_file": "run_whisperx.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "kernel_sources": ["masashiki/trimming"],
  "competition_sources": [],
  "model_sources": []
}
```

---

## run_whisperx.py
```python
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
TARGET_VIDEO_NAME = "sfw.mp4"
DEFAULT_INPUT_VIDEO_PATH = "/kaggle/input/trimming/trimmed_videos/sfw.mp4"

# ハードコーディングされたアクセストークン
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # ★ご自身のトークン（hf_...）が入っている部分

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
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN, device=device
            )
            diarize_segments = diarize_model(
                audio, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("✅ 話者識別・単語への割当が完了しました！")
        except Exception as e:
            print(f"⚠️ 話者識別エラー: {e}", file=sys.stderr)

    # ★ 4. 話者の交代ポイントでテキストとタイムスタンプを再分割・再構成
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
```

---

## output/whisperx-large-diarization.log
```log
[{"stream_name":"stdout","time":1.536565534,"data":"=== WhisperX Large 話者分離 ＆ 会話再構成処理 ===\n"}
,{"stream_name":"stdout","time":1.5366373800000002,"data":"[1/4] 依存ライブラリ (WhisperX) をチェック・インストール中...\n"}
,{"stream_name":"stdout","time":16.97126433,"data":"     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.8/44.8 kB 1.9 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":20.111261778,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.5/39.5 MB 49.6 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":20.152936615,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 47.9 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":20.34271206,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.7/16.7 MB 79.3 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":20.392692589,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 894.6/894.6 kB 28.2 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":31.444120194,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 887.9/887.9 MB 1.9 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":33.826865877,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 155.6/155.6 MB 7.8 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":40.029596164,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 322.4/322.4 MB 4.8 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":40.173543961,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.0/4.0 MB 31.6 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":40.251766016,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 23.4 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":40.524975751,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.6/8.6 MB 33.7 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":41.542108795,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.5/35.5 MB 27.4 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":41.595130651,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 848.6/848.6 kB 23.2 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.113261541,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.2/19.2 MB 33.4 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.264465707,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 72.5/72.5 kB 4.0 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.298591196,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 137.2/137.2 kB 7.0 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.331175443,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.0/60.0 kB 3.1 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.362938824,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 204.6/204.6 kB 13.0 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.393170827,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.5/57.5 kB 3.6 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.429183896,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 53.7/53.7 kB 2.3 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.462663341,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.1/52.1 kB 2.5 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.554058163,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127.8/127.8 kB 8.3 MB/s eta 0:00:00\n"}
,{"stream_name":"stdout","time":42.58347831,"data":"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 48.5/48.5 kB 3.0 MB/s eta 0:00:00\n"}
,{"stream_name":"stderr","time":155.089318405,"data":"ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n"}
,{"stream_name":"stderr","time":155.089373323,"data":"bigframes 2.39.0 requires google-cloud-bigquery-storage\u003c3.0.0,\u003e=2.30.0, which is not installed.\n"}
,{"stream_name":"stderr","time":155.08937875,"data":"google-adk 1.29.0 requires google-cloud-bigquery-storage\u003e=2.0.0, which is not installed.\n"}
,{"stream_name":"stderr","time":155.089382189,"data":"ydata-profiling 4.18.4 requires numpy\u003c2.4,\u003e=1.22, but you have numpy 2.5.1 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089385531,"data":"google-colab 1.0.0 requires jupyter-server==2.14.0, but you have jupyter-server 2.12.5 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089388741,"data":"google-colab 1.0.0 requires pandas==2.2.2, but you have pandas 2.3.3 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089391829,"data":"dopamine-rl 4.1.2 requires gym\u003c=0.25.2, but you have gym 0.26.2 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089409778,"data":"moviepy 1.0.3 requires decorator\u003c5.0,\u003e=4.0.2, but you have decorator 5.3.1 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089415379,"data":"numba 0.60.0 requires numpy\u003c2.1,\u003e=1.22, but you have numpy 2.5.1 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089420083,"data":"google-adk 1.29.0 requires opentelemetry-api\u003c1.39.0,\u003e=1.36.0, but you have opentelemetry-api 1.44.0 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089425924,"data":"google-adk 1.29.0 requires opentelemetry-sdk\u003c1.39.0,\u003e=1.36.0, but you have opentelemetry-sdk 1.44.0 which is incompatible.\n"}
,{"stream_name":"stderr","time":155.089431581,"data":"opentelemetry-exporter-gcp-logging 1.11.0a0 requires opentelemetry-sdk\u003c1.39.0,\u003e=1.35.0, but you have opentelemetry-sdk 1.44.0 which is incompatible.\n"}
,{"stream_name":"stdout","time":155.663262396,"data":"✅ WhisperX の準備が完了しました。\n"}
,{"stream_name":"stdout","time":155.676691188,"data":"🔍 [自動検出] 入力動画を発見しました: /kaggle/input/notebooks/masashiki/trimming/trimmed_videos/sfw.mp4\n"}
,{"stream_name":"stdout","time":161.277843736,"data":"\n"}
,{"stream_name":"stdout","time":161.277921622,"data":"[2/4] 動画ファイルから音声をロード＆高精度文字起こし中 (large-v3 / Device: cuda)...\n"}
,{"stream_name":"stderr","time":205.465083903,"data":"Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n"}
,{"stream_name":"stdout","time":220.039262581,"data":"2026-07-31 07:40:04 - whisperx.vads.pyannote - INFO - Performing voice activity detection using Pyannote...\n"}
,{"stream_name":"stderr","time":220.512647205,"data":"Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.6.5. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint ../../usr/local/lib/python3.12/dist-packages/whisperx/assets/pytorch_model.bin`\n"}
,{"stream_name":"stderr","time":221.819896837,"data":"/usr/local/lib/python3.12/dist-packages/pyannote/audio/utils/reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.\n"}
,{"stream_name":"stderr","time":221.81997033,"data":"It can be re-enabled by calling\n"}
,{"stream_name":"stderr","time":221.819978191,"data":"   \u003e\u003e\u003e import torch\n"}
,{"stream_name":"stderr","time":221.819984921,"data":"   \u003e\u003e\u003e torch.backends.cuda.matmul.allow_tf32 = True\n"}
,{"stream_name":"stderr","time":221.819991161,"data":"   \u003e\u003e\u003e torch.backends.cudnn.allow_tf32 = True\n"}
,{"stream_name":"stderr","time":221.819996478,"data":"See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.\n"}
,{"stream_name":"stderr","time":221.820001625,"data":"\n"}
,{"stream_name":"stderr","time":221.820006388,"data":"  warnings.warn(\n"}
,{"stream_name":"stdout","time":229.433165841,"data":"\n"}
,{"stream_name":"stdout","time":229.433220261,"data":"[3/4] タイムスタンプのアライメント調整中...\n"}
,{"stream_name":"stderr","time":239.319005219,"data":"\rLoading weights:   0%|          | 0/424 [00:00\u003c?, ?it/s]\rLoading weights:   0%|          | 1/424 [00:00\u003c00:00, 7157.52it/s, Materializing param=lm_head.bias]\rLoading weights:   0%|          | 1/424 [00:00\u003c00:00, 3682.44it/s, Materializing param=lm_head.bias]\rLoading weights:   0%|          | 2/424 [00:00\u003c00:00, 3586.41it/s, Materializing param=lm_head.weight]\rLoading weights:   0%|          | 2/424 [00:00\u003c00:00, 3087.45it/s, Materializing param=lm_head.weight]\rLoading weights:   1%|          | 3/424 [00:00\u003c00:00, 3427.65it/s, Materializing param=wav2vec2.encoder.layer_norm.bias]\rLoading weights:   1%|          | 3/424 [00:00\u003c00:00, 3096.95it/s, Materializing param=wav2vec2.encoder.layer_norm.bias]\rLoading weights:   1%|          | 4/424 [00:00\u003c00:00, 3248.88it/s, Materializing param=wav2vec2.encoder.layer_norm.weight]\rLoading weights:   1%|          | 4/424 [00:00\u003c00:00, 3011.53it/s, Materializing param=wav2vec2.encoder.layer_norm.weight]\rLoading weights:   1%|          | 5/424 [00:00\u003c00:00, 3292.75it/s, Materializing param=wav2vec2.encoder.layers.0.attention.k_proj.bias]\rLoading weights:   1%|          | 5/424 [00:00\u003c00:00, 3145.57it/s, Materializing param=wav2vec2.encoder.layers.0.attention.k_proj.bias]\rLoading weights:   1%|▏         | 6/424 [00:00\u003c00:00, 3334.55it/s, Materializing param=wav2vec2.encoder.layers.0.attention.k_proj.weight]\rLoading weights:   1%|▏         | 6/424 [00:00\u003c00:00, 3195.66it/s, Materializing param=wav2vec2.encoder.layers.0.attention.k_proj.weight]\rLoading weights:   2%|▏         | 7/424 [00:00\u003c00:00, 3296.30it/s, Materializing param=wav2vec2.encoder.layers.0.attention.out_proj.bias]\rLoading weights:   2%|▏         | 7/424 [00:00\u003c00:00, 3179.91it/s, Materializing param=wav2vec2.encoder.layers.0.attention.out_proj.bias]\rLoading weights:   2%|▏         | 8/424 [00:00\u003c00:00, 3324.85it/s, Materializing param=wav2vec2.encoder.layers.0.attention.out_proj.weight]\rLoading weights:   2%|▏         | 8/424 [00:00\u003c00:00, 3211.26it/s, Materializing param=wav2vec2.encoder.layers.0.attention.out_proj.weight]\rLoading weights:   2%|▏         | 9/424 [00:00\u003c00:00, 3325.29it/s, Materializing param=wav2vec2.encoder.layers.0.attention.q_proj.bias]    \rLoading weights:   2%|▏         | 9/424 [00:00\u003c00:00, 3238.57it/s, Materializing param=wav2vec2.encoder.layers.0.attention.q_proj.bias]\rLoading weights:   2%|▏         | 10/424 [00:00\u003c00:00, 3360.01it/s, Materializing param=wav2vec2.encoder.layers.0.attention.q_proj.weight]\rLoading weights:   2%|▏         | 10/424 [00:00\u003c00:00, 3239.09it/s, Materializing param=wav2vec2.encoder.layers.0.attention.q_proj.weight]\rLoading weights:   3%|▎         | 11/424 [00:00\u003c00:00, 3300.00it/s, Materializing param=wav2vec2.encoder.layers.0.attention.v_proj.bias]  \rLoading weights:   3%|▎         | 11/424 [00:00\u003c00:00, 3196.21it/s, Materializing param=wav2vec2.encoder.layers.0.attention.v_proj.bias]\rLoading weights:   3%|▎         | 12/424 [00:00\u003c00:00, 3277.87it/s, Materializing param=wav2vec2.encoder.layers.0.attention.v_proj.weight]\rLoading weights:   3%|▎         | 12/424 [00:00\u003c00:00, 3208.90it/s, Materializing param=wav2vec2.encoder.layers.0.attention.v_proj.weight]\rLoading weights:   3%|▎         | 13/424 [00:00\u003c00:00, 3258.00it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.bias]\rLoading weights:   3%|▎         | 13/424 [00:00\u003c00:00, 3153.43it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.bias]\rLoading weights:   3%|▎         | 14/424 [00:00\u003c00:00, 3190.28it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.weight]\rLoading weights:   3%|▎         | 14/424 [00:00\u003c00:00, 3106.56it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.weight]\rLoading weights:   4%|▎         | 15/424 [00:00\u003c00:00, 3173.82it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.output_dense.bias]        \rLoading weights:   4%|▎         | 15/424 [00:00\u003c00:00, 3113.50it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.output_dense.bias]\rLoading weights:   4%|▍         | 16/424 [00:00\u003c00:00, 3124.83it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.output_dense.weight]\rLoading weights:   4%|▍         | 16/424 [00:00\u003c00:00, 3078.81it/s, Materializing param=wav2vec2.encoder.layers.0.feed_forward.output_dense.weight]\rLoading weights:   4%|▍         | 17/424 [00:00\u003c00:00, 3125.96it/s, Materializing param=wav2vec2.encoder.layers.0.final_layer_norm.bias]           \rLoading weights:   4%|▍         | 17/424 [00:00\u003c00:00, 3077.26it/s, Materializing param=wav2vec2.encoder.layers.0.final_layer_norm.bias]\rLoading weights:   4%|▍         | 18/424 [00:00\u003c00:00, 3126.71it/s, Materializing param=wav2vec2.encoder.layers.0.final_layer_norm.weight]\rLoading weights:   4%|▍         | 18/424 [00:00\u003c00:00, 3076.88it/s, Materializing param=wav2vec2.encoder.layers.0.final_layer_norm.weight]\rLoading weights:   4%|▍         | 19/424 [00:00\u003c00:00, 3095.91it/s, Materializing param=wav2vec2.encoder.layers.0.layer_norm.bias]        \rLoading weights:   4%|▍         | 19/424 [00:00\u003c00:00, 3048.19it/s, Materializing param=wav2vec2.encoder.layers.0.layer_norm.bias]\rLoading weights:   5%|▍         | 20/424 [00:00\u003c00:00, 3085.64it/s, Materializing param=wav2vec2.encoder.layers.0.layer_norm.weight]\rLoading weights:   5%|▍         | 20/424 [00:00\u003c00:00, 3048.41it/s, Materializing param=wav2vec2.encoder.layers.0.layer_norm.weight]\rLoading weights:   5%|▍         | 21/424 [00:00\u003c00:00, 3073.50it/s, Materializing param=wav2vec2.encoder.layers.1.attention.k_proj.bias]\rLoading weights:   5%|▍         | 21/424 [00:00\u003c00:00, 3027.55it/s, Materializing param=wav2vec2.encoder.layers.1.attention.k_proj.bias]\rLoading weights:   5%|▌         | 22/424 [00:00\u003c00:00, 3038.65it/s, Materializing param=wav2vec2.encoder.layers.1.attention.k_proj.weight]\rLoading weights:   5%|▌         | 22/424 [00:00\u003c00:00, 2993.31it/s, Materializing param=wav2vec2.encoder.layers.1.attention.k_proj.weight]\rLoading weights:   5%|▌         | 23/424 [00:00\u003c00:00, 3034.38it/s, Materializing param=wav2vec2.encoder.layers.1.attention.out_proj.bias]\rLoading weights:   5%|▌         | 23/424 [00:00\u003c00:00, 2990.92it/s, Materializing param=wav2vec2.encoder.layers.1.attention.out_proj.bias]\rLoading weights:   6%|▌         | 24/424 [00:00\u003c00:00, 3018.84it/s, Materializing param=wav2vec2.encoder.layers.1.attention.out_proj.weight]\rLoading weights:   6%|▌         | 24/424 [00:00\u003c00:00, 2965.13it/s, Materializing param=wav2vec2.encoder.layers.1.attention.out_proj.weight]\rLoading weights:   6%|▌         | 25/424 [00:00\u003c00:00, 2996.10it/s, Materializing param=wav2vec2.encoder.layers.1.attention.q_proj.bias]    \rLoading weights:   6%|▌         | 25/424 [00:00\u003c00:00, 2954.82it/s, Materializing param=wav2vec2.encoder.layers.1.attention.q_proj.bias]\rLoading weights:   6%|▌         | 26/424 [00:00\u003c00:00, 2981.60it/s, Materializing param=wav2vec2.encoder.layers.1.attention.q_proj.weight]\rLoading weights:   6%|▌         | 26/424 [00:00\u003c00:00, 2939.09it/s, Materializing param=wav2vec2.encoder.layers.1.attention.q_proj.weight]\rLoading weights:   6%|▋         | 27/424 [00:00\u003c00:00, 2961.46it/s, Materializing param=wav2vec2.encoder.layers.1.attention.v_proj.bias]  \rLoading weights:   6%|▋         | 27/424 [00:00\u003c00:00, 2923.99it/s, Materializing param=wav2vec2.encoder.layers.1.attention.v_proj.bias]\rLoading weights:   7%|▋         | 28/424 [00:00\u003c00:00, 2952.77it/s, Materializing param=wav2vec2.encoder.layers.1.attention.v_proj.weight]\rLoading weights:   7%|▋         | 28/424 [00:00\u003c00:00, 2908.74it/s, Materializing param=wav2vec2.encoder.layers.1.attention.v_proj.weight]\rLoading weights:   7%|▋         | 29/424 [00:00\u003c00:00, 2927.86it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.intermediate_dense.bias]\rLoading weights:   7%|▋         | 29/424 [00:00\u003c00:00, 2884.32it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.intermediate_dense.bias]\rLoading weights:   7%|▋         | 30/424 [00:00\u003c00:00, 2899.36it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.intermediate_dense.weight]\rLoading weights:   7%|▋         | 30/424 [00:00\u003c00:00, 2867.12it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.intermediate_dense.weight]\rLoading weights:   7%|▋         | 31/424 [00:00\u003c00:00, 2896.75it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.output_dense.bias]        \rLoading weights:   7%|▋         | 31/424 [00:00\u003c00:00, 2866.60it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.output_dense.bias]\rLoading weights:   8%|▊         | 32/424 [00:00\u003c00:00, 2887.27it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.output_dense.weight]\rLoading weights:   8%|▊         | 32/424 [00:00\u003c00:00, 2851.27it/s, Materializing param=wav2vec2.encoder.layers.1.feed_forward.output_dense.weight]\rLoading weights:   8%|▊         | 33/424 [00:00\u003c00:00, 2875.38it/s, Materializing param=wav2vec2.encoder.layers.1.final_layer_norm.bias]           \rLoading weights:   8%|▊         | 33/424 [00:00\u003c00:00, 2846.81it/s, Materializing param=wav2vec2.encoder.layers.1.final_layer_norm.bias]\rLoading weights:   8%|▊         | 34/424 [00:00\u003c00:00, 2867.73it/s, Materializing param=wav2vec2.encoder.layers.1.final_layer_norm.weight]\rLoading weights:   8%|▊         | 34/424 [00:00\u003c00:00, 2839.58it/s, Materializing param=wav2vec2.encoder.layers.1.final_layer_norm.weight]\rLoading weights:   8%|▊         | 35/424 [00:00\u003c00:00, 2860.61it/s, Materializing param=wav2vec2.encoder.layers.1.layer_norm.bias]        \rLoading weights:   8%|▊         | 35/424 [00:00\u003c00:00, 2833.55it/s, Materializing param=wav2vec2.encoder.layers.1.layer_norm.bias]\rLoading weights:   8%|▊         | 36/424 [00:00\u003c00:00, 2857.37it/s, Materializing param=wav2vec2.encoder.layers.1.layer_norm.weight]\rLoading weights:   8%|▊         | 36/424 [00:00\u003c00:00, 2827.09it/s, Materializing param=wav2vec2.encoder.layers.1.layer_norm.weight]\rLoading weights:   9%|▊         | 37/424 [00:00\u003c00:00, 2848.82it/s, Materializing param=wav2vec2.encoder.layers.2.attention.k_proj.bias]\rLoading weights:   9%|▊         | 37/424 [00:00\u003c00:00, 2811.09it/s, Materializing param=wav2vec2.encoder.layers.2.attention.k_proj.bias]\rLoading weights:   9%|▉         | 38/424 [00:00\u003c00:00, 2834.04it/s, Materializing param=wav2vec2.encoder.layers.2.attention.k_proj.weight]\rLoading weights:   9%|▉         | 38/424 [00:00\u003c00:00, 2812.93it/s, Materializing param=wav2vec2.encoder.layers.2.attention.k_proj.weight]\rLoading weights:   9%|▉         | 39/424 [00:00\u003c00:00, 2837.77it/s, Materializing param=wav2vec2.encoder.layers.2.attention.out_proj.bias]\rLoading weights:   9%|▉         | 39/424 [00:00\u003c00:00, 2816.04it/s, Materializing param=wav2vec2.encoder.layers.2.attention.out_proj.bias]\rLoading weights:   9%|▉         | 40/424 [00:00\u003c00:00, 2831.55it/s, Materializing param=wav2vec2.encoder.layers.2.attention.out_proj.weight]\rLoading weights:   9%|▉         | 40/424 [00:00\u003c00:00, 2813.36it/s, Materializing param=wav2vec2.encoder.layers.2.attention.out_proj.weight]\rLoading weights:  10%|▉         | 41/424 [00:00\u003c00:00, 2824.91it/s, Materializing param=wav2vec2.encoder.layers.2.attention.q_proj.bias]    \rLoading weights:  10%|▉         | 41/424 [00:00\u003c00:00, 2810.46it/s, Materializing param=wav2vec2.encoder.layers.2.attention.q_proj.bias]\rLoading weights:  10%|▉         | 42/424 [00:00\u003c00:00, 2769.13it/s, Materializing param=wav2vec2.encoder.layers.2.attention.q_proj.weight]\rLoading weights:  10%|▉         | 42/424 [00:00\u003c00:00, 2745.05it/s, Materializing param=wav2vec2.encoder.layers.2.attention.q_proj.weight]\rLoading weights:  10%|█         | 43/424 [00:00\u003c00:00, 2770.47it/s, Materializing param=wav2vec2.encoder.layers.2.attention.v_proj.bias]  \rLoading weights:  10%|█         | 43/424 [00:00\u003c00:00, 2756.12it/s, Materializing param=wav2vec2.encoder.layers.2.attention.v_proj.bias]\rLoading weights:  10%|█         | 44/424 [00:00\u003c00:00, 2772.22it/s, Materializing param=wav2vec2.encoder.layers.2.attention.v_proj.weight]\rLoading weights:  10%|█         | 44/424 [00:00\u003c00:00, 2757.88it/s, Materializing param=wav2vec2.encoder.layers.2.attention.v_proj.weight]\rLoading weights:  11%|█         | 45/424 [00:00\u003c00:00, 2780.43it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.intermediate_dense.bias]\rLoading weights:  11%|█         | 45/424 [00:00\u003c00:00, 2765.84it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.intermediate_dense.bias]\rLoading weights:  11%|█         | 46/424 [00:00\u003c00:00, 2785.34it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.intermediate_dense.weight]\rLoading weights:  11%|█         | 46/424 [00:00\u003c00:00, 2772.14it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.intermediate_dense.weight]\rLoading weights:  11%|█         | 47/424 [00:00\u003c00:00, 2793.03it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.output_dense.bias]        \rLoading weights:  11%|█         | 47/424 [00:00\u003c00:00, 2779.88it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.output_dense.bias]\rLoading weights:  11%|█▏        | 48/424 [00:00\u003c00:00, 2796.47it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.output_dense.weight]\rLoading weights:  11%|█▏        | 48/424 [00:00\u003c00:00, 2782.14it/s, Materializing param=wav2vec2.encoder.layers.2.feed_forward.output_dense.weight]\rLoading weights:  12%|█▏        | 49/424 [00:00\u003c00:00, 2801.81it/s, Materializing param=wav2vec2.encoder.layers.2.final_layer_norm.bias]           \rLoading weights:  12%|█▏        | 49/424 [00:00\u003c00:00, 2786.27it/s, Materializing param=wav2vec2.encoder.layers.2.final_layer_norm.bias]\rLoading weights:  12%|█▏        | 50/424 [00:00\u003c00:00, 2806.49it/s, Materializing param=wav2vec2.encoder.layers.2.final_layer_norm.weight]\rLoading weights:  12%|█▏        | 50/424 [00:00\u003c00:00, 2794.08it/s, Materializing param=wav2vec2.encoder.layers.2.final_layer_norm.weight]\rLoading weights:  12%|█▏        | 51/424 [00:00\u003c00:00, 2806.92it/s, Materializing param=wav2vec2.encoder.layers.2.layer_norm.bias]        \rLoading weights:  12%|█▏        | 51/424 [00:00\u003c00:00, 2794.23it/s, Materializing param=wav2vec2.encoder.layers.2.layer_norm.bias]\rLoading weights:  12%|█▏        | 52/424 [00:00\u003c00:00, 2814.53it/s, Materializing param=wav2vec2.encoder.layers.2.layer_norm.weight]\rLoading weights:  12%|█▏        | 52/424 [00:00\u003c00:00, 2801.09it/s, Materializing param=wav2vec2.encoder.layers.2.layer_norm.weight]\rLoading weights:  12%|█▎        | 53/424 [00:00\u003c00:00, 2822.12it/s, Materializing param=wav2vec2.encoder.layers.3.attention.k_proj.bias]\rLoading weights:  12%|█▎        | 53/424 [00:00\u003c00:00, 2809.63it/s, Materializing param=wav2vec2.encoder.layers.3.attention.k_proj.bias]\rLoading weights:  13%|█▎        | 54/424 [00:00\u003c00:00, 2822.93it/s, Materializing param=wav2vec2.encoder.layers.3.attention.k_proj.weight]\rLoading weights:  13%|█▎        | 54/424 [00:00\u003c00:00, 2810.92it/s, Materializing param=wav2vec2.encoder.layers.3.attention.k_proj.weight]\rLoading weights:  13%|█▎        | 55/424 [00:00\u003c00:00, 2824.69it/s, Materializing param=wav2vec2.encoder.layers.3.attention.out_proj.bias]\rLoading weights:  13%|█▎        | 55/424 [00:00\u003c00:00, 2813.22it/s, Materializing param=wav2vec2.encoder.layers.3.attention.out_proj.bias]\rLoading weights:  13%|█▎        | 56/424 [00:00\u003c00:00, 2826.45it/s, Materializing param=wav2vec2.encoder.layers.3.attention.out_proj.weight]\rLoading weights:  13%|█▎        | 56/424 [00:00\u003c00:00, 2804.92it/s, Materializing param=wav2vec2.encoder.layers.3.attention.out_proj.weight]\rLoading weights:  13%|█▎        | 57/424 [00:00\u003c00:00, 2818.89it/s, Materializing param=wav2vec2.encoder.layers.3.attention.q_proj.bias]    \rLoading weights:  13%|█▎        | 57/424 [00:00\u003c00:00, 2805.72it/s, Materializing param=wav2vec2.encoder.layers.3.attention.q_proj.bias]\rLoading weights:  14%|█▎        | 58/424 [00:00\u003c00:00, 2814.38it/s, Materializing param=wav2vec2.encoder.layers.3.attention.q_proj.weight]\rLoading weights:  14%|█▎        | 58/424 [00:00\u003c00:00, 2801.45it/s, Materializing param=wav2vec2.encoder.layers.3.attention.q_proj.weight]\rLoading weights:  14%|█▍        | 59/424 [00:00\u003c00:00, 2815.87it/s, Materializing param=wav2vec2.encoder.layers.3.attention.v_proj.bias]  \rLoading weights:  14%|█▍        | 59/424 [00:00\u003c00:00, 2805.05it/s, Materializing param=wav2vec2.encoder.layers.3.attention.v_proj.bias]\rLoading weights:  14%|█▍        | 60/424 [00:00\u003c00:00, 2813.46it/s, Materializing param=wav2vec2.encoder.layers.3.attention.v_proj.weight]\rLoading weights:  14%|█▍        | 60/424 [00:00\u003c00:00, 2797.63it/s, Materializing param=wav2vec2.encoder.layers.3.attention.v_proj.weight]\rLoading weights:  14%|█▍        | 61/424 [00:00\u003c00:00, 2811.94it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.bias]\rLoading weights:  14%|█▍        | 61/424 [00:00\u003c00:00, 2775.12it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.bias]\rLoading weights:  15%|█▍        | 62/424 [00:00\u003c00:00, 2784.64it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.weight]\rLoading weights:  15%|█▍        | 62/424 [00:00\u003c00:00, 2774.10it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.weight]\rLoading weights:  15%|█▍        | 63/424 [00:00\u003c00:00, 2791.06it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.output_dense.bias]        \rLoading weights:  15%|█▍        | 63/424 [00:00\u003c00:00, 2781.95it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.output_dense.bias]\rLoading weights:  15%|█▌        | 64/424 [00:00\u003c00:00, 2796.61it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.output_dense.weight]\rLoading weights:  15%|█▌        | 64/424 [00:00\u003c00:00, 2785.96it/s, Materializing param=wav2vec2.encoder.layers.3.feed_forward.output_dense.weight]\rLoading weights:  15%|█▌        | 65/424 [00:00\u003c00:00, 2797.38it/s, Materializing param=wav2vec2.encoder.layers.3.final_layer_norm.bias]           \rLoading weights:  15%|█▌        | 65/424 [00:00\u003c00:00, 2786.54it/s, Materializing param=wav2vec2.encoder.layers.3.final_layer_norm.bias]\rLoading weights:  16%|█▌        | 66/424 [00:00\u003c00:00, 2802.91it/s, Materializing param=wav2vec2.encoder.layers.3.final_layer_norm.weight]\rLoading weights:  16%|█▌        | 66/424 [00:00\u003c00:00, 2794.06it/s, Materializing param=wav2vec2.encoder.layers.3.final_layer_norm.weight]\rLoading weights:  16%|█▌        | 67/424 [00:00\u003c00:00, 2810.13it/s, Materializing param=wav2vec2.encoder.layers.3.layer_norm.bias]        \rLoading weights:  16%|█▌        | 67/424 [00:00\u003c00:00, 2801.11it/s, Materializing param=wav2vec2.encoder.layers.3.layer_norm.bias]\rLoading weights:  16%|█▌        | 68/424 [00:00\u003c00:00, 2813.30it/s, Materializing param=wav2vec2.encoder.layers.3.layer_norm.weight]\rLoading weights:  16%|█▌        | 68/424 [00:00\u003c00:00, 2804.37it/s, Materializing param=wav2vec2.encoder.layers.3.layer_norm.weight]\rLoading weights:  16%|█▋        | 69/424 [00:00\u003c00:00, 2819.49it/s, Materializing param=wav2vec2.encoder.layers.4.attention.k_proj.bias]\rLoading weights:  16%|█▋        | 69/424 [00:00\u003c00:00, 2809.37it/s, Materializing param=wav2vec2.encoder.layers.4.attention.k_proj.bias]\rLoading weights:  17%|█▋        | 70/424 [00:00\u003c00:00, 2826.68it/s, Materializing param=wav2vec2.encoder.layers.4.attention.k_proj.weight]\rLoading weights:  17%|█▋        | 70/424 [00:00\u003c00:00, 2815.91it/s, Materializing param=wav2vec2.encoder.layers.4.attention.k_proj.weight]\rLoading weights:  17%|█▋        | 71/424 [00:00\u003c00:00, 2823.59it/s, Materializing param=wav2vec2.encoder.layers.4.attention.out_proj.bias]\rLoading weights:  17%|█▋        | 71/424 [00:00\u003c00:00, 2812.68it/s, Materializing param=wav2vec2.encoder.layers.4.attention.out_proj.bias]\rLoading weights:  17%|█▋        | 72/424 [00:00\u003c00:00, 2823.84it/s, Materializing param=wav2vec2.encoder.layers.4.attention.out_proj.weight]\rLoading weights:  17%|█▋        | 72/424 [00:00\u003c00:00, 2814.86it/s, Materializing param=wav2vec2.encoder.layers.4.attention.out_proj.weight]\rLoading weights:  17%|█▋        | 73/424 [00:00\u003c00:00, 2826.06it/s, Materializing param=wav2vec2.encoder.layers.4.attention.q_proj.bias]    \rLoading weights:  17%|█▋        | 73/424 [00:00\u003c00:00, 2814.58it/s, Materializing param=wav2vec2.encoder.layers.4.attention.q_proj.bias]\rLoading weights:  17%|█▋        | 74/424 [00:00\u003c00:00, 2816.71it/s, Materializing param=wav2vec2.encoder.layers.4.attention.q_proj.weight]\rLoading weights:  17%|█▋        | 74/424 [00:00\u003c00:00, 2807.79it/s, Materializing param=wav2vec2.encoder.layers.4.attention.q_proj.weight]\rLoading weights:  18%|█▊        | 75/424 [00:00\u003c00:00, 2821.10it/s, Materializing param=wav2vec2.encoder.layers.4.attention.v_proj.bias]  \rLoading weights:  18%|█▊        | 75/424 [00:00\u003c00:00, 2813.56it/s, Materializing param=wav2vec2.encoder.layers.4.attention.v_proj.bias]\rLoading weights:  18%|█▊        | 76/424 [00:00\u003c00:00, 2826.15it/s, Materializing param=wav2vec2.encoder.layers.4.attention.v_proj.weight]\rLoading weights:  18%|█▊        | 76/424 [00:00\u003c00:00, 2816.49it/s, Materializing param=wav2vec2.encoder.layers.4.attention.v_proj.weight]\rLoading weights:  18%|█▊        | 77/424 [00:00\u003c00:00, 2828.18it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.intermediate_dense.bias]\rLoading weights:  18%|█▊        | 77/424 [00:00\u003c00:00, 2820.38it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.intermediate_dense.bias]\rLoading weights:  18%|█▊        | 78/424 [00:00\u003c00:00, 2835.09it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.intermediate_dense.weight]\rLoading weights:  18%|█▊        | 78/424 [00:00\u003c00:00, 2827.62it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.intermediate_dense.weight]\rLoading weights:  19%|█▊        | 79/424 [00:00\u003c00:00, 2839.14it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.output_dense.bias]        \rLoading weights:  19%|█▊        | 79/424 [00:00\u003c00:00, 2831.06it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.output_dense.bias]\rLoading weights:  19%|█▉        | 80/424 [00:00\u003c00:00, 2840.47it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.output_dense.weight]\rLoading weights:  19%|█▉        | 80/424 [00:00\u003c00:00, 2831.91it/s, Materializing param=wav2vec2.encoder.layers.4.feed_forward.output_dense.weight]\rLoading weights:  19%|█▉        | 81/424 [00:00\u003c00:00, 2842.72it/s, Materializing param=wav2vec2.encoder.layers.4.final_layer_norm.bias]           \rLoading weights:  19%|█▉        | 81/424 [00:00\u003c00:00, 2833.92it/s, Materializing param=wav2vec2.encoder.layers.4.final_layer_norm.bias]\rLoading weights:  19%|█▉        | 82/424 [00:00\u003c00:00, 2844.25it/s, Materializing param=wav2vec2.encoder.layers.4.final_layer_norm.weight]\rLoading weights:  19%|█▉        | 82/424 [00:00\u003c00:00, 2834.27it/s, Materializing param=wav2vec2.encoder.layers.4.final_layer_norm.weight]\rLoading weights:  20%|█▉        | 83/424 [00:00\u003c00:00, 2839.75it/s, Materializing param=wav2vec2.encoder.layers.4.layer_norm.bias]        \rLoading weights:  20%|█▉        | 83/424 [00:00\u003c00:00, 2830.30it/s, Materializing param=wav2vec2.encoder.layers.4.layer_norm.bias]\rLoading weights:  20%|█▉        | 84/424 [00:00\u003c00:00, 2837.41it/s, Materializing param=wav2vec2.encoder.layers.4.layer_norm.weight]\rLoading weights:  20%|█▉        | 84/424 [00:00\u003c00:00, 2827.73it/s, Materializing param=wav2vec2.encoder.layers.4.layer_norm.weight]\rLoading weights:  20%|██        | 85/424 [00:00\u003c00:00, 2837.53it/s, Materializing param=wav2vec2.encoder.layers.5.attention.k_proj.bias]\rLoading weights:  20%|██        | 85/424 [00:00\u003c00:00, 2824.83it/s, Materializing param=wav2vec2.encoder.layers.5.attention.k_proj.bias]\rLoading weights:  20%|██        | 86/424 [00:00\u003c00:00, 2833.66it/s, Materializing param=wav2vec2.encoder.layers.5.attention.k_proj.weight]\rLoading weights:  20%|██        | 86/424 [00:00\u003c00:00, 2824.00it/s, Materializing param=wav2vec2.encoder.layers.5.attention.k_proj.weight]\rLoading weights:  21%|██        | 87/424 [00:00\u003c00:00, 2832.49it/s, Materializing param=wav2vec2.encoder.layers.5.attention.out_proj.bias]\rLoading weights:  21%|██        | 87/424 [00:00\u003c00:00, 2821.26it/s, Materializing param=wav2vec2.encoder.layers.5.attention.out_proj.bias]\rLoading weights:  21%|██        | 88/424 [00:00\u003c00:00, 2830.08it/s, Materializing param=wav2vec2.encoder.layers.5.attention.out_proj.weight]\rLoading weights:  21%|██        | 88/424 [00:00\u003c00:00, 2821.86it/s, Materializing param=wav2vec2.encoder.layers.5.attention.out_proj.weight]\rLoading weights:  21%|██        | 89/424 [00:00\u003c00:00, 2833.49it/s, Materializing param=wav2vec2.encoder.layers.5.attention.q_proj.bias]    \rLoading weights:  21%|██        | 89/424 [00:00\u003c00:00, 2827.06it/s, Materializing param=wav2vec2.encoder.layers.5.attention.q_proj.bias]\rLoading weights:  21%|██        | 90/424 [00:00\u003c00:00, 2835.18it/s, Materializing param=wav2vec2.encoder.layers.5.attention.q_proj.weight]\rLoading weights:  21%|██        | 90/424 [00:00\u003c00:00, 2826.37it/s, Materializing param=wav2vec2.encoder.layers.5.attention.q_proj.weight]\rLoading weights:  21%|██▏       | 91/424 [00:00\u003c00:00, 2832.90it/s, Materializing param=wav2vec2.encoder.layers.5.attention.v_proj.bias]  \rLoading weights:  21%|██▏       | 91/424 [00:00\u003c00:00, 2825.03it/s, Materializing param=wav2vec2.encoder.layers.5.attention.v_proj.bias]\rLoading weights:  22%|██▏       | 92/424 [00:00\u003c00:00, 2831.54it/s, Materializing param=wav2vec2.encoder.layers.5.attention.v_proj.weight]\rLoading weights:  22%|██▏       | 92/424 [00:00\u003c00:00, 2823.25it/s, Materializing param=wav2vec2.encoder.layers.5.attention.v_proj.weight]\rLoading weights:  22%|██▏       | 93/424 [00:00\u003c00:00, 2830.97it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.intermediate_dense.bias]\rLoading weights:  22%|██▏       | 93/424 [00:00\u003c00:00, 2824.39it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.intermediate_dense.bias]\rLoading weights:  22%|██▏       | 94/424 [00:00\u003c00:00, 2835.60it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.intermediate_dense.weight]\rLoading weights:  22%|██▏       | 94/424 [00:00\u003c00:00, 2829.09it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.intermediate_dense.weight]\rLoading weights:  22%|██▏       | 95/424 [00:00\u003c00:00, 2834.94it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.output_dense.bias]        \rLoading weights:  22%|██▏       | 95/424 [00:00\u003c00:00, 2826.31it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.output_dense.bias]\rLoading weights:  23%|██▎       | 96/424 [00:00\u003c00:00, 2832.73it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.output_dense.weight]\rLoading weights:  23%|██▎       | 96/424 [00:00\u003c00:00, 2824.13it/s, Materializing param=wav2vec2.encoder.layers.5.feed_forward.output_dense.weight]\rLoading weights:  23%|██▎       | 97/424 [00:00\u003c00:00, 2832.19it/s, Materializing param=wav2vec2.encoder.layers.5.final_layer_norm.bias]           \rLoading weights:  23%|██▎       | 97/424 [00:00\u003c00:00, 2824.92it/s, Materializing param=wav2vec2.encoder.layers.5.final_layer_norm.bias]\rLoading weights:  23%|██▎       | 98/424 [00:00\u003c00:00, 2833.38it/s, Materializing param=wav2vec2.encoder.layers.5.final_layer_norm.weight]\rLoading weights:  23%|██▎       | 98/424 [00:00\u003c00:00, 2825.40it/s, Materializing param=wav2vec2.encoder.layers.5.final_layer_norm.weight]\rLoading weights:  23%|██▎       | 99/424 [00:00\u003c00:00, 2832.40it/s, Materializing param=wav2vec2.encoder.layers.5.layer_norm.bias]        \rLoading weights:  23%|██▎       | 99/424 [00:00\u003c00:00, 2821.93it/s, Materializing param=wav2vec2.encoder.layers.5.layer_norm.bias]\rLoading weights:  24%|██▎       | 100/424 [00:00\u003c00:00, 2832.21it/s, Materializing param=wav2vec2.encoder.layers.5.layer_norm.weight]\rLoading weights:  24%|██▎       | 100/424 [00:00\u003c00:00, 2825.53it/s, Materializing param=wav2vec2.encoder.layers.5.layer_norm.weight]\rLoading weights:  24%|██▍       | 101/424 [00:00\u003c00:00, 2835.28it/s, Materializing param=wav2vec2.encoder.layers.6.attention.k_proj.bias]\rLoading weights:  24%|██▍       | 101/424 [00:00\u003c00:00, 2828.75it/s, Materializing param=wav2vec2.encoder.layers.6.attention.k_proj.bias]\rLoading weights:  24%|██▍       | 102/424 [00:00\u003c00:00, 2837.33it/s, Materializing param=wav2vec2.encoder.layers.6.attention.k_proj.weight]\rLoading weights:  24%|██▍       | 102/424 [00:00\u003c00:00, 2830.48it/s, Materializing param=wav2vec2.encoder.layers.6.attention.k_proj.weight]\rLoading weights:  24%|██▍       | 103/424 [00:00\u003c00:00, 2837.28it/s, Materializing param=wav2vec2.encoder.layers.6.attention.out_proj.bias]\rLoading weights:  24%|██▍       | 103/424 [00:00\u003c00:00, 2830.46it/s, Materializing param=wav2vec2.encoder.layers.6.attention.out_proj.bias]\rLoading weights:  25%|██▍       | 104/424 [00:00\u003c00:00, 2840.12it/s, Materializing param=wav2vec2.encoder.layers.6.attention.out_proj.weight]\rLoading weights:  25%|██▍       | 104/424 [00:00\u003c00:00, 2833.47it/s, Materializing param=wav2vec2.encoder.layers.6.attention.out_proj.weight]\rLoading weights:  25%|██▍       | 105/424 [00:00\u003c00:00, 2842.29it/s, Materializing param=wav2vec2.encoder.layers.6.attention.q_proj.bias]    \rLoading weights:  25%|██▍       | 105/424 [00:00\u003c00:00, 2832.04it/s, Materializing param=wav2vec2.encoder.layers.6.attention.q_proj.bias]\rLoading weights:  25%|██▌       | 106/424 [00:00\u003c00:00, 2841.22it/s, Materializing param=wav2vec2.encoder.layers.6.attention.q_proj.weight]\rLoading weights:  25%|██▌       | 106/424 [00:00\u003c00:00, 2834.04it/s, Materializing param=wav2vec2.encoder.layers.6.attention.q_proj.weight]\rLoading weights:  25%|██▌       | 107/424 [00:00\u003c00:00, 2845.22it/s, Materializing param=wav2vec2.encoder.layers.6.attention.v_proj.bias]  \rLoading weights:  25%|██▌       | 107/424 [00:00\u003c00:00, 2839.13it/s, Materializing param=wav2vec2.encoder.layers.6.attention.v_proj.bias]\rLoading weights:  25%|██▌       | 108/424 [00:00\u003c00:00, 2850.93it/s, Materializing param=wav2vec2.encoder.layers.6.attention.v_proj.weight]\rLoading weights:  25%|██▌       | 108/424 [00:00\u003c00:00, 2844.06it/s, Materializing param=wav2vec2.encoder.layers.6.attention.v_proj.weight]\rLoading weights:  26%|██▌       | 109/424 [00:00\u003c00:00, 2851.35it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.intermediate_dense.bias]\rLoading weights:  26%|██▌       | 109/424 [00:00\u003c00:00, 2844.69it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.intermediate_dense.bias]\rLoading weights:  26%|██▌       | 110/424 [00:00\u003c00:00, 2853.60it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.intermediate_dense.weight]\rLoading weights:  26%|██▌       | 110/424 [00:00\u003c00:00, 2847.00it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.intermediate_dense.weight]\rLoading weights:  26%|██▌       | 111/424 [00:00\u003c00:00, 2855.77it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.output_dense.bias]        \rLoading weights:  26%|██▌       | 111/424 [00:00\u003c00:00, 2846.50it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.output_dense.bias]\rLoading weights:  26%|██▋       | 112/424 [00:00\u003c00:00, 2852.06it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.output_dense.weight]\rLoading weights:  26%|██▋       | 112/424 [00:00\u003c00:00, 2845.32it/s, Materializing param=wav2vec2.encoder.layers.6.feed_forward.output_dense.weight]\rLoading weights:  27%|██▋       | 113/424 [00:00\u003c00:00, 2853.85it/s, Materializing param=wav2vec2.encoder.layers.6.final_layer_norm.bias]           \rLoading weights:  27%|██▋       | 113/424 [00:00\u003c00:00, 2848.60it/s, Materializing param=wav2vec2.encoder.layers.6.final_layer_norm.bias]\rLoading weights:  27%|██▋       | 114/424 [00:00\u003c00:00, 2856.88it/s, Materializing param=wav2vec2.encoder.layers.6.final_layer_norm.weight]\rLoading weights:  27%|██▋       | 114/424 [00:00\u003c00:00, 2847.64it/s, Materializing param=wav2vec2.encoder.layers.6.final_layer_norm.weight]\rLoading weights:  27%|██▋       | 115/424 [00:00\u003c00:00, 2854.20it/s, Materializing param=wav2vec2.encoder.layers.6.layer_norm.bias]        \rLoading weights:  27%|██▋       | 115/424 [00:00\u003c00:00, 2846.73it/s, Materializing param=wav2vec2.encoder.layers.6.layer_norm.bias]\rLoading weights:  27%|██▋       | 116/424 [00:00\u003c00:00, 2853.52it/s, Materializing param=wav2vec2.encoder.layers.6.layer_norm.weight]\rLoading weights:  27%|██▋       | 116/424 [00:00\u003c00:00, 2845.06it/s, Materializing param=wav2vec2.encoder.layers.6.layer_norm.weight]\rLoading weights:  28%|██▊       | 117/424 [00:00\u003c00:00, 2853.00it/s, Materializing param=wav2vec2.encoder.layers.7.attention.k_proj.bias]\rLoading weights:  28%|██▊       | 117/424 [00:00\u003c00:00, 2843.99it/s, Materializing param=wav2vec2.encoder.layers.7.attention.k_proj.bias]\rLoading weights:  28%|██▊       | 118/424 [00:00\u003c00:00, 2849.67it/s, Materializing param=wav2vec2.encoder.layers.7.attention.k_proj.weight]\rLoading weights:  28%|██▊       | 118/424 [00:00\u003c00:00, 2840.10it/s, Materializing param=wav2vec2.encoder.layers.7.attention.k_proj.weight]\rLoading weights:  28%|██▊       | 119/424 [00:00\u003c00:00, 2846.87it/s, Materializing param=wav2vec2.encoder.layers.7.attention.out_proj.bias]\rLoading weights:  28%|██▊       | 119/424 [00:00\u003c00:00, 2839.31it/s, Materializing param=wav2vec2.encoder.layers.7.attention.out_proj.bias]\rLoading weights:  28%|██▊       | 120/424 [00:00\u003c00:00, 2844.19it/s, Materializing param=wav2vec2.encoder.layers.7.attention.out_proj.weight]\rLoading weights:  28%|██▊       | 120/424 [00:00\u003c00:00, 2835.12it/s, Materializing param=wav2vec2.encoder.layers.7.attention.out_proj.weight]\rLoading weights:  29%|██▊       | 121/424 [00:00\u003c00:00, 2840.10it/s, Materializing param=wav2vec2.encoder.layers.7.attention.q_proj.bias]    \rLoading weights:  29%|██▊       | 121/424 [00:00\u003c00:00, 2830.54it/s, Materializing param=wav2vec2.encoder.layers.7.attention.q_proj.bias]\rLoading weights:  29%|██▉       | 122/424 [00:00\u003c00:00, 2837.45it/s, Materializing param=wav2vec2.encoder.layers.7.attention.q_proj.weight]\rLoading weights:  29%|██▉       | 122/424 [00:00\u003c00:00, 2825.88it/s, Materializing param=wav2vec2.encoder.layers.7.attention.q_proj.weight]\rLoading weights:  29%|██▉       | 123/424 [00:00\u003c00:00, 2833.32it/s, Materializing param=wav2vec2.encoder.layers.7.attention.v_proj.bias]  \rLoading weights:  29%|██▉       | 123/424 [00:00\u003c00:00, 2826.10it/s, Materializing param=wav2vec2.encoder.layers.7.attention.v_proj.bias]\rLoading weights:  29%|██▉       | 124/424 [00:00\u003c00:00, 2833.60it/s, Materializing param=wav2vec2.encoder.layers.7.attention.v_proj.weight]\rLoading weights:  29%|██▉       | 124/424 [00:00\u003c00:00, 2825.21it/s, Materializing param=wav2vec2.encoder.layers.7.attention.v_proj.weight]\rLoading weights:  29%|██▉       | 125/424 [00:00\u003c00:00, 2831.02it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.intermediate_dense.bias]\rLoading weights:  29%|██▉       | 125/424 [00:00\u003c00:00, 2824.16it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.intermediate_dense.bias]\rLoading weights:  30%|██▉       | 126/424 [00:00\u003c00:00, 2831.86it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.intermediate_dense.weight]\rLoading weights:  30%|██▉       | 126/424 [00:00\u003c00:00, 2824.05it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.intermediate_dense.weight]\rLoading weights:  30%|██▉       | 127/424 [00:00\u003c00:00, 2828.90it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.output_dense.bias]        \rLoading weights:  30%|██▉       | 127/424 [00:00\u003c00:00, 2815.15it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.output_dense.bias]\rLoading weights:  30%|███       | 128/424 [00:00\u003c00:00, 2817.20it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.output_dense.weight]\rLoading weights:  30%|███       | 128/424 [00:00\u003c00:00, 2807.42it/s, Materializing param=wav2vec2.encoder.layers.7.feed_forward.output_dense.weight]\rLoading weights:  30%|███       | 129/424 [00:00\u003c00:00, 2811.96it/s, Materializing param=wav2vec2.encoder.layers.7.final_layer_norm.bias]           \rLoading weights:  30%|███       | 129/424 [00:00\u003c00:00, 2804.68it/s, Materializing param=wav2vec2.encoder.layers.7.final_layer_norm.bias]\rLoading weights:  31%|███       | 130/424 [00:00\u003c00:00, 2811.49it/s, Materializing param=wav2vec2.encoder.layers.7.final_layer_norm.weight]\rLoading weights:  31%|███       | 130/424 [00:00\u003c00:00, 2803.39it/s, Materializing param=wav2vec2.encoder.layers.7.final_layer_norm.weight]\rLoading weights:  31%|███       | 131/424 [00:00\u003c00:00, 2812.10it/s, Materializing param=wav2vec2.encoder.layers.7.layer_norm.bias]        \rLoading weights:  31%|███       | 131/424 [00:00\u003c00:00, 2804.72it/s, Materializing param=wav2vec2.encoder.layers.7.layer_norm.bias]\rLoading weights:  31%|███       | 132/424 [00:00\u003c00:00, 2811.20it/s, Materializing param=wav2vec2.encoder.layers.7.layer_norm.weight]\rLoading weights:  31%|███       | 132/424 [00:00\u003c00:00, 2803.37it/s, Materializing param=wav2vec2.encoder.layers.7.layer_norm.weight]\rLoading weights:  31%|███▏      | 133/424 [00:00\u003c00:00, 2810.60it/s, Materializing param=wav2vec2.encoder.layers.8.attention.k_proj.bias]\rLoading weights:  31%|███▏      | 133/424 [00:00\u003c00:00, 2803.41it/s, Materializing param=wav2vec2.encoder.layers.8.attention.k_proj.bias]\rLoading weights:  32%|███▏      | 134/424 [00:00\u003c00:00, 2809.03it/s, Materializing param=wav2vec2.encoder.layers.8.attention.k_proj.weight]\rLoading weights:  32%|███▏      | 134/424 [00:00\u003c00:00, 2801.15it/s, Materializing param=wav2vec2.encoder.layers.8.attention.k_proj.weight]\rLoading weights:  32%|███▏      | 135/424 [00:00\u003c00:00, 2772.60it/s, Materializing param=wav2vec2.encoder.layers.8.attention.out_proj.bias]\rLoading weights:  32%|███▏      | 135/424 [00:00\u003c00:00, 2764.37it/s, Materializing param=wav2vec2.encoder.layers.8.attention.out_proj.bias]\rLoading weights:  32%|███▏      | 136/424 [00:00\u003c00:00, 2770.27it/s, Materializing param=wav2vec2.encoder.layers.8.attention.out_proj.weight]\rLoading weights:  32%|███▏      | 136/424 [00:00\u003c00:00, 2762.42it/s, Materializing param=wav2vec2.encoder.layers.8.attention.out_proj.weight]\rLoading weights:  32%|███▏      | 137/424 [00:00\u003c00:00, 2769.37it/s, Materializing param=wav2vec2.encoder.layers.8.attention.q_proj.bias]    \rLoading weights:  32%|███▏      | 137/424 [00:00\u003c00:00, 2762.97it/s, Materializing param=wav2vec2.encoder.layers.8.attention.q_proj.bias]\rLoading weights:  33%|███▎      | 138/424 [00:00\u003c00:00, 2769.13it/s, Materializing param=wav2vec2.encoder.layers.8.attention.q_proj.weight]\rLoading weights:  33%|███▎      | 138/424 [00:00\u003c00:00, 2762.20it/s, Materializing param=wav2vec2.encoder.layers.8.attention.q_proj.weight]\rLoading weights:  33%|███▎      | 139/424 [00:00\u003c00:00, 2766.24it/s, Materializing param=wav2vec2.encoder.layers.8.attention.v_proj.bias]  \rLoading weights:  33%|███▎      | 139/424 [00:00\u003c00:00, 2758.57it/s, Materializing param=wav2vec2.encoder.layers.8.attention.v_proj.bias]\rLoading weights:  33%|███▎      | 140/424 [00:00\u003c00:00, 2763.84it/s, Materializing param=wav2vec2.encoder.layers.8.attention.v_proj.weight]\rLoading weights:  33%|███▎      | 140/424 [00:00\u003c00:00, 2757.05it/s, Materializing param=wav2vec2.encoder.layers.8.attention.v_proj.weight]\rLoading weights:  33%|███▎      | 141/424 [00:00\u003c00:00, 2761.06it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.intermediate_dense.bias]\rLoading weights:  33%|███▎      | 141/424 [00:00\u003c00:00, 2753.86it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.intermediate_dense.bias]\rLoading weights:  33%|███▎      | 142/424 [00:00\u003c00:00, 2754.83it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.intermediate_dense.weight]\rLoading weights:  33%|███▎      | 142/424 [00:00\u003c00:00, 2748.17it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.intermediate_dense.weight]\rLoading weights:  34%|███▎      | 143/424 [00:00\u003c00:00, 2753.08it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.output_dense.bias]        \rLoading weights:  34%|███▎      | 143/424 [00:00\u003c00:00, 2744.71it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.output_dense.bias]\rLoading weights:  34%|███▍      | 144/424 [00:00\u003c00:00, 2749.01it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.output_dense.weight]\rLoading weights:  34%|███▍      | 144/424 [00:00\u003c00:00, 2742.36it/s, Materializing param=wav2vec2.encoder.layers.8.feed_forward.output_dense.weight]\rLoading weights:  34%|███▍      | 145/424 [00:00\u003c00:00, 2749.38it/s, Materializing param=wav2vec2.encoder.layers.8.final_layer_norm.bias]           \rLoading weights:  34%|███▍      | 145/424 [00:00\u003c00:00, 2743.44it/s, Materializing param=wav2vec2.encoder.layers.8.final_layer_norm.bias]\rLoading weights:  34%|███▍      | 146/424 [00:00\u003c00:00, 2747.62it/s, Materializing param=wav2vec2.encoder.layers.8.final_layer_norm.weight]\rLoading weights:  34%|███▍      | 146/424 [00:00\u003c00:00, 2740.14it/s, Materializing param=wav2vec2.encoder.layers.8.final_layer_norm.weight]\rLoading weights:  35%|███▍      | 147/424 [00:00\u003c00:00, 2745.32it/s, Materializing param=wav2vec2.encoder.layers.8.layer_norm.bias]        \rLoading weights:  35%|███▍      | 147/424 [00:00\u003c00:00, 2739.30it/s, Materializing param=wav2vec2.encoder.layers.8.layer_norm.bias]\rLoading weights:  35%|███▍      | 148/424 [00:00\u003c00:00, 2745.19it/s, Materializing param=wav2vec2.encoder.layers.8.layer_norm.weight]\rLoading weights:  35%|███▍      | 148/424 [00:00\u003c00:00, 2737.07it/s, Materializing param=wav2vec2.encoder.layers.8.layer_norm.weight]\rLoading weights:  35%|███▌      | 149/424 [00:00\u003c00:00, 2743.79it/s, Materializing param=wav2vec2.encoder.layers.9.attention.k_proj.bias]\rLoading weights:  35%|███▌      | 149/424 [00:00\u003c00:00, 2737.90it/s, Materializing param=wav2vec2.encoder.layers.9.attention.k_proj.bias]\rLoading weights:  35%|███▌      | 150/424 [00:00\u003c00:00, 2746.16it/s, Materializing param=wav2vec2.encoder.layers.9.attention.k_proj.weight]\rLoading weights:  35%|███▌      | 150/424 [00:00\u003c00:00, 2730.77it/s, Materializing param=wav2vec2.encoder.layers.9.attention.k_proj.weight]\rLoading weights:  36%|███▌      | 151/424 [00:00\u003c00:00, 2735.75it/s, Materializing param=wav2vec2.encoder.layers.9.attention.out_proj.bias]\rLoading weights:  36%|███▌      | 151/424 [00:00\u003c00:00, 2728.97it/s, Materializing param=wav2vec2.encoder.layers.9.attention.out_proj.bias]\rLoading weights:  36%|███▌      | 152/424 [00:00\u003c00:00, 2735.11it/s, Materializing param=wav2vec2.encoder.layers.9.attention.out_proj.weight]\rLoading weights:  36%|███▌      | 152/424 [00:00\u003c00:00, 2729.26it/s, Materializing param=wav2vec2.encoder.layers.9.attention.out_proj.weight]\rLoading weights:  36%|███▌      | 153/424 [00:00\u003c00:00, 2735.42it/s, Materializing param=wav2vec2.encoder.layers.9.attention.q_proj.bias]    \rLoading weights:  36%|███▌      | 153/424 [00:00\u003c00:00, 2729.42it/s, Materializing param=wav2vec2.encoder.layers.9.attention.q_proj.bias]\rLoading weights:  36%|███▋      | 154/424 [00:00\u003c00:00, 2735.22it/s, Materializing param=wav2vec2.encoder.layers.9.attention.q_proj.weight]\rLoading weights:  36%|███▋      | 154/424 [00:00\u003c00:00, 2729.63it/s, Materializing param=wav2vec2.encoder.layers.9.attention.q_proj.weight]\rLoading weights:  37%|███▋      | 155/424 [00:00\u003c00:00, 2735.61it/s, Materializing param=wav2vec2.encoder.layers.9.attention.v_proj.bias]  \rLoading weights:  37%|███▋      | 155/424 [00:00\u003c00:00, 2729.18it/s, Materializing param=wav2vec2.encoder.layers.9.attention.v_proj.bias]\rLoading weights:  37%|███▋      | 156/424 [00:00\u003c00:00, 2733.55it/s, Materializing param=wav2vec2.encoder.layers.9.attention.v_proj.weight]\rLoading weights:  37%|███▋      | 156/424 [00:00\u003c00:00, 2726.81it/s, Materializing param=wav2vec2.encoder.layers.9.attention.v_proj.weight]\rLoading weights:  37%|███▋      | 157/424 [00:00\u003c00:00, 2731.84it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.intermediate_dense.bias]\rLoading weights:  37%|███▋      | 157/424 [00:00\u003c00:00, 2726.35it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.intermediate_dense.bias]\rLoading weights:  37%|███▋      | 158/424 [00:00\u003c00:00, 2731.52it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.intermediate_dense.weight]\rLoading weights:  37%|███▋      | 158/424 [00:00\u003c00:00, 2726.02it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.intermediate_dense.weight]\rLoading weights:  38%|███▊      | 159/424 [00:00\u003c00:00, 2729.94it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.output_dense.bias]        \rLoading weights:  38%|███▊      | 159/424 [00:00\u003c00:00, 2723.70it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.output_dense.bias]\rLoading weights:  38%|███▊      | 160/424 [00:00\u003c00:00, 2727.73it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.output_dense.weight]\rLoading weights:  38%|███▊      | 160/424 [00:00\u003c00:00, 2721.25it/s, Materializing param=wav2vec2.encoder.layers.9.feed_forward.output_dense.weight]\rLoading weights:  38%|███▊      | 161/424 [00:00\u003c00:00, 2725.37it/s, Materializing param=wav2vec2.encoder.layers.9.final_layer_norm.bias]           \rLoading weights:  38%|███▊      | 161/424 [00:00\u003c00:00, 2720.05it/s, Materializing param=wav2vec2.encoder.layers.9.final_layer_norm.bias]\rLoading weights:  38%|███▊      | 162/424 [00:00\u003c00:00, 2725.91it/s, Materializing param=wav2vec2.encoder.layers.9.final_layer_norm.weight]\rLoading weights:  38%|███▊      | 162/424 [00:00\u003c00:00, 2720.44it/s, Materializing param=wav2vec2.encoder.layers.9.final_layer_norm.weight]\rLoading weights:  38%|███▊      | 163/424 [00:00\u003c00:00, 2726.71it/s, Materializing param=wav2vec2.encoder.layers.9.layer_norm.bias]        \rLoading weights:  38%|███▊      | 163/424 [00:00\u003c00:00, 2719.55it/s, Materializing param=wav2vec2.encoder.layers.9.layer_norm.bias]\rLoading weights:  39%|███▊      | 164/424 [00:00\u003c00:00, 2725.20it/s, Materializing param=wav2vec2.encoder.layers.9.layer_norm.weight]\rLoading weights:  39%|███▊      | 164/424 [00:00\u003c00:00, 2719.87it/s, Materializing param=wav2vec2.encoder.layers.9.layer_norm.weight]\rLoading weights:  39%|███▉      | 165/424 [00:00\u003c00:00, 2724.80it/s, Materializing param=wav2vec2.encoder.layers.10.attention.k_proj.bias]\rLoading weights:  39%|███▉      | 165/424 [00:00\u003c00:00, 2719.38it/s, Materializing param=wav2vec2.encoder.layers.10.attention.k_proj.bias]\rLoading weights:  39%|███▉      | 166/424 [00:00\u003c00:00, 2723.33it/s, Materializing param=wav2vec2.encoder.layers.10.attention.k_proj.weight]\rLoading weights:  39%|███▉      | 166/424 [00:00\u003c00:00, 2717.16it/s, Materializing param=wav2vec2.encoder.layers.10.attention.k_proj.weight]\rLoading weights:  39%|███▉      | 167/424 [00:00\u003c00:00, 2723.01it/s, Materializing param=wav2vec2.encoder.layers.10.attention.out_proj.bias]\rLoading weights:  39%|███▉      | 167/424 [00:00\u003c00:00, 2717.78it/s, Materializing param=wav2vec2.encoder.layers.10.attention.out_proj.bias]\rLoading weights:  40%|███▉      | 168/424 [00:00\u003c00:00, 2723.65it/s, Materializing param=wav2vec2.encoder.layers.10.attention.out_proj.weight]\rLoading weights:  40%|███▉      | 168/424 [00:00\u003c00:00, 2717.83it/s, Materializing param=wav2vec2.encoder.layers.10.attention.out_proj.weight]\rLoading weights:  40%|███▉      | 169/424 [00:00\u003c00:00, 2724.25it/s, Materializing param=wav2vec2.encoder.layers.10.attention.q_proj.bias]    \rLoading weights:  40%|███▉      | 169/424 [00:00\u003c00:00, 2716.03it/s, Materializing param=wav2vec2.encoder.layers.10.attention.q_proj.bias]\rLoading weights:  40%|████      | 170/424 [00:00\u003c00:00, 2721.36it/s, Materializing param=wav2vec2.encoder.layers.10.attention.q_proj.weight]\rLoading weights:  40%|████      | 170/424 [00:00\u003c00:00, 2715.86it/s, Materializing param=wav2vec2.encoder.layers.10.attention.q_proj.weight]\rLoading weights:  40%|████      | 171/424 [00:00\u003c00:00, 2720.84it/s, Materializing param=wav2vec2.encoder.layers.10.attention.v_proj.bias]  \rLoading weights:  40%|████      | 171/424 [00:00\u003c00:00, 2715.99it/s, Materializing param=wav2vec2.encoder.layers.10.attention.v_proj.bias]\rLoading weights:  41%|████      | 172/424 [00:00\u003c00:00, 2720.13it/s, Materializing param=wav2vec2.encoder.layers.10.attention.v_proj.weight]\rLoading weights:  41%|████      | 172/424 [00:00\u003c00:00, 2714.73it/s, Materializing param=wav2vec2.encoder.layers.10.attention.v_proj.weight]\rLoading weights:  41%|████      | 173/424 [00:00\u003c00:00, 2719.93it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.intermediate_dense.bias]\rLoading weights:  41%|████      | 173/424 [00:00\u003c00:00, 2715.06it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.intermediate_dense.bias]\rLoading weights:  41%|████      | 174/424 [00:00\u003c00:00, 2720.37it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.intermediate_dense.weight]\rLoading weights:  41%|████      | 174/424 [00:00\u003c00:00, 2713.44it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.intermediate_dense.weight]\rLoading weights:  41%|████▏     | 175/424 [00:00\u003c00:00, 2718.44it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.output_dense.bias]        \rLoading weights:  41%|████▏     | 175/424 [00:00\u003c00:00, 2713.67it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.output_dense.bias]\rLoading weights:  42%|████▏     | 176/424 [00:00\u003c00:00, 2720.13it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.output_dense.weight]\rLoading weights:  42%|████▏     | 176/424 [00:00\u003c00:00, 2715.16it/s, Materializing param=wav2vec2.encoder.layers.10.feed_forward.output_dense.weight]\rLoading weights:  42%|████▏     | 177/424 [00:00\u003c00:00, 2721.55it/s, Materializing param=wav2vec2.encoder.layers.10.final_layer_norm.bias]           \rLoading weights:  42%|████▏     | 177/424 [00:00\u003c00:00, 2715.29it/s, Materializing param=wav2vec2.encoder.layers.10.final_layer_norm.bias]\rLoading weights:  42%|████▏     | 178/424 [00:00\u003c00:00, 2720.24it/s, Materializing param=wav2vec2.encoder.layers.10.final_layer_norm.weight]\rLoading weights:  42%|████▏     | 178/424 [00:00\u003c00:00, 2714.43it/s, Materializing param=wav2vec2.encoder.layers.10.final_layer_norm.weight]\rLoading weights:  42%|████▏     | 179/424 [00:00\u003c00:00, 2719.92it/s, Materializing param=wav2vec2.encoder.layers.10.layer_norm.bias]        \rLoading weights:  42%|████▏     | 179/424 [00:00\u003c00:00, 2715.16it/s, Materializing param=wav2vec2.encoder.layers.10.layer_norm.bias]\rLoading weights:  42%|████▏     | 180/424 [00:00\u003c00:00, 2719.53it/s, Materializing param=wav2vec2.encoder.layers.10.layer_norm.weight]\rLoading weights:  42%|████▏     | 180/424 [00:00\u003c00:00, 2714.24it/s, Materializing param=wav2vec2.encoder.layers.10.layer_norm.weight]\rLoading weights:  43%|████▎     | 181/424 [00:00\u003c00:00, 2719.86it/s, Materializing param=wav2vec2.encoder.layers.11.attention.k_proj.bias]\rLoading weights:  43%|████▎     | 181/424 [00:00\u003c00:00, 2714.23it/s, Materializing param=wav2vec2.encoder.layers.11.attention.k_proj.bias]\rLoading weights:  43%|████▎     | 182/424 [00:00\u003c00:00, 2718.54it/s, Materializing param=wav2vec2.encoder.layers.11.attention.k_proj.weight]\rLoading weights:  43%|████▎     | 182/424 [00:00\u003c00:00, 2712.79it/s, Materializing param=wav2vec2.encoder.layers.11.attention.k_proj.weight]\rLoading weights:  43%|████▎     | 183/424 [00:00\u003c00:00, 2715.87it/s, Materializing param=wav2vec2.encoder.layers.11.attention.out_proj.bias]\rLoading weights:  43%|████▎     | 183/424 [00:00\u003c00:00, 2711.27it/s, Materializing param=wav2vec2.encoder.layers.11.attention.out_proj.bias]\rLoading weights:  43%|████▎     | 184/424 [00:00\u003c00:00, 2715.18it/s, Materializing param=wav2vec2.encoder.layers.11.attention.out_proj.weight]\rLoading weights:  43%|████▎     | 184/424 [00:00\u003c00:00, 2710.09it/s, Materializing param=wav2vec2.encoder.layers.11.attention.out_proj.weight]\rLoading weights:  44%|████▎     | 185/424 [00:00\u003c00:00, 2713.63it/s, Materializing param=wav2vec2.encoder.layers.11.attention.q_proj.bias]    \rLoading weights:  44%|████▎     | 185/424 [00:00\u003c00:00, 2708.98it/s, Materializing param=wav2vec2.encoder.layers.11.attention.q_proj.bias]\rLoading weights:  44%|████▍     | 186/424 [00:00\u003c00:00, 2714.61it/s, Materializing param=wav2vec2.encoder.layers.11.attention.q_proj.weight]\rLoading weights:  44%|████▍     | 186/424 [00:00\u003c00:00, 2711.18it/s, Materializing param=wav2vec2.encoder.layers.11.attention.q_proj.weight]\rLoading weights:  44%|████▍     | 187/424 [00:00\u003c00:00, 2718.83it/s, Materializing param=wav2vec2.encoder.layers.11.attention.v_proj.bias]  \rLoading weights:  44%|████▍     | 187/424 [00:00\u003c00:00, 2714.89it/s, Materializing param=wav2vec2.encoder.layers.11.attention.v_proj.bias]\rLoading weights:  44%|████▍     | 188/424 [00:00\u003c00:00, 2717.26it/s, Materializing param=wav2vec2.encoder.layers.11.attention.v_proj.weight]\rLoading weights:  44%|████▍     | 188/424 [00:00\u003c00:00, 2711.74it/s, Materializing param=wav2vec2.encoder.layers.11.attention.v_proj.weight]\rLoading weights:  45%|████▍     | 189/424 [00:00\u003c00:00, 2716.00it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.intermediate_dense.bias]\rLoading weights:  45%|████▍     | 189/424 [00:00\u003c00:00, 2709.96it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.intermediate_dense.bias]\rLoading weights:  45%|████▍     | 190/424 [00:00\u003c00:00, 2713.81it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.intermediate_dense.weight]\rLoading weights:  45%|████▍     | 190/424 [00:00\u003c00:00, 2708.13it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.intermediate_dense.weight]\rLoading weights:  45%|████▌     | 191/424 [00:00\u003c00:00, 2711.95it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.output_dense.bias]        \rLoading weights:  45%|████▌     | 191/424 [00:00\u003c00:00, 2708.26it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.output_dense.bias]\rLoading weights:  45%|████▌     | 192/424 [00:00\u003c00:00, 2713.06it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.output_dense.weight]\rLoading weights:  45%|████▌     | 192/424 [00:00\u003c00:00, 2709.00it/s, Materializing param=wav2vec2.encoder.layers.11.feed_forward.output_dense.weight]\rLoading weights:  46%|████▌     | 193/424 [00:00\u003c00:00, 2712.05it/s, Materializing param=wav2vec2.encoder.layers.11.final_layer_norm.bias]           \rLoading weights:  46%|████▌     | 193/424 [00:00\u003c00:00, 2708.46it/s, Materializing param=wav2vec2.encoder.layers.11.final_layer_norm.bias]\rLoading weights:  46%|████▌     | 194/424 [00:00\u003c00:00, 2713.28it/s, Materializing param=wav2vec2.encoder.layers.11.final_layer_norm.weight]\rLoading weights:  46%|████▌     | 194/424 [00:00\u003c00:00, 2710.49it/s, Materializing param=wav2vec2.encoder.layers.11.final_layer_norm.weight]\rLoading weights:  46%|████▌     | 195/424 [00:00\u003c00:00, 2715.45it/s, Materializing param=wav2vec2.encoder.layers.11.layer_norm.bias]        \rLoading weights:  46%|████▌     | 195/424 [00:00\u003c00:00, 2712.82it/s, Materializing param=wav2vec2.encoder.layers.11.layer_norm.bias]\rLoading weights:  46%|████▌     | 196/424 [00:00\u003c00:00, 2716.98it/s, Materializing param=wav2vec2.encoder.layers.11.layer_norm.weight]\rLoading weights:  46%|████▌     | 196/424 [00:00\u003c00:00, 2712.59it/s, Materializing param=wav2vec2.encoder.layers.11.layer_norm.weight]\rLoading weights:  46%|████▋     | 197/424 [00:00\u003c00:00, 2717.55it/s, Materializing param=wav2vec2.encoder.layers.12.attention.k_proj.bias]\rLoading weights:  46%|████▋     | 197/424 [00:00\u003c00:00, 2712.71it/s, Materializing param=wav2vec2.encoder.layers.12.attention.k_proj.bias]\rLoading weights:  47%|████▋     | 198/424 [00:00\u003c00:00, 2717.49it/s, Materializing param=wav2vec2.encoder.layers.12.attention.k_proj.weight]\rLoading weights:  47%|████▋     | 198/424 [00:00\u003c00:00, 2713.09it/s, Materializing param=wav2vec2.encoder.layers.12.attention.k_proj.weight]\rLoading weights:  47%|████▋     | 199/424 [00:00\u003c00:00, 2716.74it/s, Materializing param=wav2vec2.encoder.layers.12.attention.out_proj.bias]\rLoading weights:  47%|████▋     | 199/424 [00:00\u003c00:00, 2712.17it/s, Materializing param=wav2vec2.encoder.layers.12.attention.out_proj.bias]\rLoading weights:  47%|████▋     | 200/424 [00:00\u003c00:00, 2716.73it/s, Materializing param=wav2vec2.encoder.layers.12.attention.out_proj.weight]\rLoading weights:  47%|████▋     | 200/424 [00:00\u003c00:00, 2710.16it/s, Materializing param=wav2vec2.encoder.layers.12.attention.out_proj.weight]\rLoading weights:  47%|████▋     | 201/424 [00:00\u003c00:00, 2711.69it/s, Materializing param=wav2vec2.encoder.layers.12.attention.q_proj.bias]    \rLoading weights:  47%|████▋     | 201/424 [00:00\u003c00:00, 2703.75it/s, Materializing param=wav2vec2.encoder.layers.12.attention.q_proj.bias]\rLoading weights:  48%|████▊     | 202/424 [00:00\u003c00:00, 2707.86it/s, Materializing param=wav2vec2.encoder.layers.12.attention.q_proj.weight]\rLoading weights:  48%|████▊     | 202/424 [00:00\u003c00:00, 2702.73it/s, Materializing param=wav2vec2.encoder.layers.12.attention.q_proj.weight]\rLoading weights:  48%|████▊     | 203/424 [00:00\u003c00:00, 2705.76it/s, Materializing param=wav2vec2.encoder.layers.12.attention.v_proj.bias]  \rLoading weights:  48%|████▊     | 203/424 [00:00\u003c00:00, 2699.80it/s, Materializing param=wav2vec2.encoder.layers.12.attention.v_proj.bias]\rLoading weights:  48%|████▊     | 204/424 [00:00\u003c00:00, 2703.39it/s, Materializing param=wav2vec2.encoder.layers.12.attention.v_proj.weight]\rLoading weights:  48%|████▊     | 204/424 [00:00\u003c00:00, 2698.02it/s, Materializing param=wav2vec2.encoder.layers.12.attention.v_proj.weight]\rLoading weights:  48%|████▊     | 205/424 [00:00\u003c00:00, 2702.10it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.intermediate_dense.bias]\rLoading weights:  48%|████▊     | 205/424 [00:00\u003c00:00, 2697.43it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.intermediate_dense.bias]\rLoading weights:  49%|████▊     | 206/424 [00:00\u003c00:00, 2701.16it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.intermediate_dense.weight]\rLoading weights:  49%|████▊     | 206/424 [00:00\u003c00:00, 2696.70it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.intermediate_dense.weight]\rLoading weights:  49%|████▉     | 207/424 [00:00\u003c00:00, 2700.01it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.output_dense.bias]        \rLoading weights:  49%|████▉     | 207/424 [00:00\u003c00:00, 2694.48it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.output_dense.bias]\rLoading weights:  49%|████▉     | 208/424 [00:00\u003c00:00, 2697.44it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.output_dense.weight]\rLoading weights:  49%|████▉     | 208/424 [00:00\u003c00:00, 2693.85it/s, Materializing param=wav2vec2.encoder.layers.12.feed_forward.output_dense.weight]\rLoading weights:  49%|████▉     | 209/424 [00:00\u003c00:00, 2696.70it/s, Materializing param=wav2vec2.encoder.layers.12.final_layer_norm.bias]           \rLoading weights:  49%|████▉     | 209/424 [00:00\u003c00:00, 2692.03it/s, Materializing param=wav2vec2.encoder.layers.12.final_layer_norm.bias]\rLoading weights:  50%|████▉     | 210/424 [00:00\u003c00:00, 2696.33it/s, Materializing param=wav2vec2.encoder.layers.12.final_layer_norm.weight]\rLoading weights:  50%|████▉     | 210/424 [00:00\u003c00:00, 2691.33it/s, Materializing param=wav2vec2.encoder.layers.12.final_layer_norm.weight]\rLoading weights:  50%|████▉     | 211/424 [00:00\u003c00:00, 2694.69it/s, Materializing param=wav2vec2.encoder.layers.12.layer_norm.bias]        \rLoading weights:  50%|████▉     | 211/424 [00:00\u003c00:00, 2689.33it/s, Materializing param=wav2vec2.encoder.layers.12.layer_norm.bias]\rLoading weights:  50%|█████     | 212/424 [00:00\u003c00:00, 2692.91it/s, Materializing param=wav2vec2.encoder.layers.12.layer_norm.weight]\rLoading weights:  50%|█████     | 212/424 [00:00\u003c00:00, 2688.40it/s, Materializing param=wav2vec2.encoder.layers.12.layer_norm.weight]\rLoading weights:  50%|█████     | 213/424 [00:00\u003c00:00, 2692.49it/s, Materializing param=wav2vec2.encoder.layers.13.attention.k_proj.bias]\rLoading weights:  50%|█████     | 213/424 [00:00\u003c00:00, 2687.54it/s, Materializing param=wav2vec2.encoder.layers.13.attention.k_proj.bias]\rLoading weights:  50%|█████     | 214/424 [00:00\u003c00:00, 2691.14it/s, Materializing param=wav2vec2.encoder.layers.13.attention.k_proj.weight]\rLoading weights:  50%|█████     | 214/424 [00:00\u003c00:00, 2686.62it/s, Materializing param=wav2vec2.encoder.layers.13.attention.k_proj.weight]\rLoading weights:  51%|█████     | 215/424 [00:00\u003c00:00, 2690.53it/s, Materializing param=wav2vec2.encoder.layers.13.attention.out_proj.bias]\rLoading weights:  51%|█████     | 215/424 [00:00\u003c00:00, 2685.09it/s, Materializing param=wav2vec2.encoder.layers.13.attention.out_proj.bias]\rLoading weights:  51%|█████     | 216/424 [00:00\u003c00:00, 2688.76it/s, Materializing param=wav2vec2.encoder.layers.13.attention.out_proj.weight]\rLoading weights:  51%|█████     | 216/424 [00:00\u003c00:00, 2684.43it/s, Materializing param=wav2vec2.encoder.layers.13.attention.out_proj.weight]\rLoading weights:  51%|█████     | 217/424 [00:00\u003c00:00, 2687.90it/s, Materializing param=wav2vec2.encoder.layers.13.attention.q_proj.bias]    \rLoading weights:  51%|█████     | 217/424 [00:00\u003c00:00, 2684.42it/s, Materializing param=wav2vec2.encoder.layers.13.attention.q_proj.bias]\rLoading weights:  51%|█████▏    | 218/424 [00:00\u003c00:00, 2687.83it/s, Materializing param=wav2vec2.encoder.layers.13.attention.q_proj.weight]\rLoading weights:  51%|█████▏    | 218/424 [00:00\u003c00:00, 2683.92it/s, Materializing param=wav2vec2.encoder.layers.13.attention.q_proj.weight]\rLoading weights:  52%|█████▏    | 219/424 [00:00\u003c00:00, 2687.40it/s, Materializing param=wav2vec2.encoder.layers.13.attention.v_proj.bias]  \rLoading weights:  52%|█████▏    | 219/424 [00:00\u003c00:00, 2683.15it/s, Materializing param=wav2vec2.encoder.layers.13.attention.v_proj.bias]\rLoading weights:  52%|█████▏    | 220/424 [00:00\u003c00:00, 2687.00it/s, Materializing param=wav2vec2.encoder.layers.13.attention.v_proj.weight]\rLoading weights:  52%|█████▏    | 220/424 [00:00\u003c00:00, 2683.54it/s, Materializing param=wav2vec2.encoder.layers.13.attention.v_proj.weight]\rLoading weights:  52%|█████▏    | 221/424 [00:00\u003c00:00, 2687.33it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.intermediate_dense.bias]\rLoading weights:  52%|█████▏    | 221/424 [00:00\u003c00:00, 2682.19it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.intermediate_dense.bias]\rLoading weights:  52%|█████▏    | 222/424 [00:00\u003c00:00, 2686.37it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.intermediate_dense.weight]\rLoading weights:  52%|█████▏    | 222/424 [00:00\u003c00:00, 2683.60it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.intermediate_dense.weight]\rLoading weights:  53%|█████▎    | 223/424 [00:00\u003c00:00, 2687.44it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.output_dense.bias]        \rLoading weights:  53%|█████▎    | 223/424 [00:00\u003c00:00, 2683.29it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.output_dense.bias]\rLoading weights:  53%|█████▎    | 224/424 [00:00\u003c00:00, 2686.50it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.output_dense.weight]\rLoading weights:  53%|█████▎    | 224/424 [00:00\u003c00:00, 2682.27it/s, Materializing param=wav2vec2.encoder.layers.13.feed_forward.output_dense.weight]\rLoading weights:  53%|█████▎    | 225/424 [00:00\u003c00:00, 2687.68it/s, Materializing param=wav2vec2.encoder.layers.13.final_layer_norm.bias]           \rLoading weights:  53%|█████▎    | 225/424 [00:00\u003c00:00, 2685.01it/s, Materializing param=wav2vec2.encoder.layers.13.final_layer_norm.bias]\rLoading weights:  53%|█████▎    | 226/424 [00:00\u003c00:00, 2689.99it/s, Materializing param=wav2vec2.encoder.layers.13.final_layer_norm.weight]\rLoading weights:  53%|█████▎    | 226/424 [00:00\u003c00:00, 2685.33it/s, Materializing param=wav2vec2.encoder.layers.13.final_layer_norm.weight]\rLoading weights:  54%|█████▎    | 227/424 [00:00\u003c00:00, 2690.02it/s, Materializing param=wav2vec2.encoder.layers.13.layer_norm.bias]        \rLoading weights:  54%|█████▎    | 227/424 [00:00\u003c00:00, 2687.03it/s, Materializing param=wav2vec2.encoder.layers.13.layer_norm.bias]\rLoading weights:  54%|█████▍    | 228/424 [00:00\u003c00:00, 2691.12it/s, Materializing param=wav2vec2.encoder.layers.13.layer_norm.weight]\rLoading weights:  54%|█████▍    | 228/424 [00:00\u003c00:00, 2687.95it/s, Materializing param=wav2vec2.encoder.layers.13.layer_norm.weight]\rLoading weights:  54%|█████▍    | 229/424 [00:00\u003c00:00, 2690.79it/s, Materializing param=wav2vec2.encoder.layers.14.attention.k_proj.bias]\rLoading weights:  54%|█████▍    | 229/424 [00:00\u003c00:00, 2687.90it/s, Materializing param=wav2vec2.encoder.layers.14.attention.k_proj.bias]\rLoading weights:  54%|█████▍    | 230/424 [00:00\u003c00:00, 2691.59it/s, Materializing param=wav2vec2.encoder.layers.14.attention.k_proj.weight]\rLoading weights:  54%|█████▍    | 230/424 [00:00\u003c00:00, 2688.69it/s, Materializing param=wav2vec2.encoder.layers.14.attention.k_proj.weight]\rLoading weights:  54%|█████▍    | 231/424 [00:00\u003c00:00, 2692.53it/s, Materializing param=wav2vec2.encoder.layers.14.attention.out_proj.bias]\rLoading weights:  54%|█████▍    | 231/424 [00:00\u003c00:00, 2688.87it/s, Materializing param=wav2vec2.encoder.layers.14.attention.out_proj.bias]\rLoading weights:  55%|█████▍    | 232/424 [00:00\u003c00:00, 2690.20it/s, Materializing param=wav2vec2.encoder.layers.14.attention.out_proj.weight]\rLoading weights:  55%|█████▍    | 232/424 [00:00\u003c00:00, 2686.77it/s, Materializing param=wav2vec2.encoder.layers.14.attention.out_proj.weight]\rLoading weights:  55%|█████▍    | 233/424 [00:00\u003c00:00, 2691.14it/s, Materializing param=wav2vec2.encoder.layers.14.attention.q_proj.bias]    \rLoading weights:  55%|█████▍    | 233/424 [00:00\u003c00:00, 2688.68it/s, Materializing param=wav2vec2.encoder.layers.14.attention.q_proj.bias]\rLoading weights:  55%|█████▌    | 234/424 [00:00\u003c00:00, 2692.51it/s, Materializing param=wav2vec2.encoder.layers.14.attention.q_proj.weight]\rLoading weights:  55%|█████▌    | 234/424 [00:00\u003c00:00, 2689.86it/s, Materializing param=wav2vec2.encoder.layers.14.attention.q_proj.weight]\rLoading weights:  55%|█████▌    | 235/424 [00:00\u003c00:00, 2693.00it/s, Materializing param=wav2vec2.encoder.layers.14.attention.v_proj.bias]  \rLoading weights:  55%|█████▌    | 235/424 [00:00\u003c00:00, 2690.74it/s, Materializing param=wav2vec2.encoder.layers.14.attention.v_proj.bias]\rLoading weights:  56%|█████▌    | 236/424 [00:00\u003c00:00, 2695.43it/s, Materializing param=wav2vec2.encoder.layers.14.attention.v_proj.weight]\rLoading weights:  56%|█████▌    | 236/424 [00:00\u003c00:00, 2693.09it/s, Materializing param=wav2vec2.encoder.layers.14.attention.v_proj.weight]\rLoading weights:  56%|█████▌    | 237/424 [00:00\u003c00:00, 2696.07it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.intermediate_dense.bias]\rLoading weights:  56%|█████▌    | 237/424 [00:00\u003c00:00, 2693.66it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.intermediate_dense.bias]\rLoading weights:  56%|█████▌    | 238/424 [00:00\u003c00:00, 2676.63it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.intermediate_dense.weight]\rLoading weights:  56%|█████▌    | 238/424 [00:00\u003c00:00, 2673.55it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.intermediate_dense.weight]\rLoading weights:  56%|█████▋    | 239/424 [00:00\u003c00:00, 2676.37it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.output_dense.bias]        \rLoading weights:  56%|█████▋    | 239/424 [00:00\u003c00:00, 2673.15it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.output_dense.bias]\rLoading weights:  57%|█████▋    | 240/424 [00:00\u003c00:00, 2676.38it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.output_dense.weight]\rLoading weights:  57%|█████▋    | 240/424 [00:00\u003c00:00, 2673.57it/s, Materializing param=wav2vec2.encoder.layers.14.feed_forward.output_dense.weight]\rLoading weights:  57%|█████▋    | 241/424 [00:00\u003c00:00, 2674.97it/s, Materializing param=wav2vec2.encoder.layers.14.final_layer_norm.bias]           \rLoading weights:  57%|█████▋    | 241/424 [00:00\u003c00:00, 2671.04it/s, Materializing param=wav2vec2.encoder.layers.14.final_layer_norm.bias]\rLoading weights:  57%|█████▋    | 242/424 [00:00\u003c00:00, 2675.00it/s, Materializing param=wav2vec2.encoder.layers.14.final_layer_norm.weight]\rLoading weights:  57%|█████▋    | 242/424 [00:00\u003c00:00, 2671.70it/s, Materializing param=wav2vec2.encoder.layers.14.final_layer_norm.weight]\rLoading weights:  57%|█████▋    | 243/424 [00:00\u003c00:00, 2675.80it/s, Materializing param=wav2vec2.encoder.layers.14.layer_norm.bias]        \rLoading weights:  57%|█████▋    | 243/424 [00:00\u003c00:00, 2672.90it/s, Materializing param=wav2vec2.encoder.layers.14.layer_norm.bias]\rLoading weights:  58%|█████▊    | 244/424 [00:00\u003c00:00, 2676.48it/s, Materializing param=wav2vec2.encoder.layers.14.layer_norm.weight]\rLoading weights:  58%|█████▊    | 244/424 [00:00\u003c00:00, 2673.50it/s, Materializing param=wav2vec2.encoder.layers.14.layer_norm.weight]\rLoading weights:  58%|█████▊    | 245/424 [00:00\u003c00:00, 2677.15it/s, Materializing param=wav2vec2.encoder.layers.15.attention.k_proj.bias]\rLoading weights:  58%|█████▊    | 245/424 [00:00\u003c00:00, 2673.21it/s, Materializing param=wav2vec2.encoder.layers.15.attention.k_proj.bias]\rLoading weights:  58%|█████▊    | 246/424 [00:00\u003c00:00, 2676.55it/s, Materializing param=wav2vec2.encoder.layers.15.attention.k_proj.weight]\rLoading weights:  58%|█████▊    | 246/424 [00:00\u003c00:00, 2673.13it/s, Materializing param=wav2vec2.encoder.layers.15.attention.k_proj.weight]\rLoading weights:  58%|█████▊    | 247/424 [00:00\u003c00:00, 2676.47it/s, Materializing param=wav2vec2.encoder.layers.15.attention.out_proj.bias]\rLoading weights:  58%|█████▊    | 247/424 [00:00\u003c00:00, 2673.16it/s, Materializing param=wav2vec2.encoder.layers.15.attention.out_proj.bias]\rLoading weights:  58%|█████▊    | 248/424 [00:00\u003c00:00, 2676.21it/s, Materializing param=wav2vec2.encoder.layers.15.attention.out_proj.weight]\rLoading weights:  58%|█████▊    | 248/424 [00:00\u003c00:00, 2672.84it/s, Materializing param=wav2vec2.encoder.layers.15.attention.out_proj.weight]\rLoading weights:  59%|█████▊    | 249/424 [00:00\u003c00:00, 2676.65it/s, Materializing param=wav2vec2.encoder.layers.15.attention.q_proj.bias]    \rLoading weights:  59%|█████▊    | 249/424 [00:00\u003c00:00, 2671.40it/s, Materializing param=wav2vec2.encoder.layers.15.attention.q_proj.bias]\rLoading weights:  59%|█████▉    | 250/424 [00:00\u003c00:00, 2675.61it/s, Materializing param=wav2vec2.encoder.layers.15.attention.q_proj.weight]\rLoading weights:  59%|█████▉    | 250/424 [00:00\u003c00:00, 2672.91it/s, Materializing param=wav2vec2.encoder.layers.15.attention.q_proj.weight]\rLoading weights:  59%|█████▉    | 251/424 [00:00\u003c00:00, 2676.97it/s, Materializing param=wav2vec2.encoder.layers.15.attention.v_proj.bias]  \rLoading weights:  59%|█████▉    | 251/424 [00:00\u003c00:00, 2673.64it/s, Materializing param=wav2vec2.encoder.layers.15.attention.v_proj.bias]\rLoading weights:  59%|█████▉    | 252/424 [00:00\u003c00:00, 2676.48it/s, Materializing param=wav2vec2.encoder.layers.15.attention.v_proj.weight]\rLoading weights:  59%|█████▉    | 252/424 [00:00\u003c00:00, 2673.00it/s, Materializing param=wav2vec2.encoder.layers.15.attention.v_proj.weight]\rLoading weights:  60%|█████▉    | 253/424 [00:00\u003c00:00, 2676.62it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.intermediate_dense.bias]\rLoading weights:  60%|█████▉    | 253/424 [00:00\u003c00:00, 2672.94it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.intermediate_dense.bias]\rLoading weights:  60%|█████▉    | 254/424 [00:00\u003c00:00, 2676.75it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.intermediate_dense.weight]\rLoading weights:  60%|█████▉    | 254/424 [00:00\u003c00:00, 2673.29it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.intermediate_dense.weight]\rLoading weights:  60%|██████    | 255/424 [00:00\u003c00:00, 2675.27it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.output_dense.bias]        \rLoading weights:  60%|██████    | 255/424 [00:00\u003c00:00, 2671.75it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.output_dense.bias]\rLoading weights:  60%|██████    | 256/424 [00:00\u003c00:00, 2674.93it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.output_dense.weight]\rLoading weights:  60%|██████    | 256/424 [00:00\u003c00:00, 2671.22it/s, Materializing param=wav2vec2.encoder.layers.15.feed_forward.output_dense.weight]\rLoading weights:  61%|██████    | 257/424 [00:00\u003c00:00, 2674.40it/s, Materializing param=wav2vec2.encoder.layers.15.final_layer_norm.bias]           \rLoading weights:  61%|██████    | 257/424 [00:00\u003c00:00, 2670.97it/s, Materializing param=wav2vec2.encoder.layers.15.final_layer_norm.bias]\rLoading weights:  61%|██████    | 258/424 [00:00\u003c00:00, 2674.15it/s, Materializing param=wav2vec2.encoder.layers.15.final_layer_norm.weight]\rLoading weights:  61%|██████    | 258/424 [00:00\u003c00:00, 2670.36it/s, Materializing param=wav2vec2.encoder.layers.15.final_layer_norm.weight]\rLoading weights:  61%|██████    | 259/424 [00:00\u003c00:00, 2673.37it/s, Materializing param=wav2vec2.encoder.layers.15.layer_norm.bias]        \rLoading weights:  61%|██████    | 259/424 [00:00\u003c00:00, 2669.85it/s, Materializing param=wav2vec2.encoder.layers.15.layer_norm.bias]\rLoading weights:  61%|██████▏   | 260/424 [00:00\u003c00:00, 2673.80it/s, Materializing param=wav2vec2.encoder.layers.15.layer_norm.weight]\rLoading weights:  61%|██████▏   | 260/424 [00:00\u003c00:00, 2669.80it/s, Materializing param=wav2vec2.encoder.layers.15.layer_norm.weight]\rLoading weights:  62%|██████▏   | 261/424 [00:00\u003c00:00, 2674.21it/s, Materializing param=wav2vec2.encoder.layers.16.attention.k_proj.bias]\rLoading weights:  62%|██████▏   | 261/424 [00:00\u003c00:00, 2671.16it/s, Materializing param=wav2vec2.encoder.layers.16.attention.k_proj.bias]\rLoading weights:  62%|██████▏   | 262/424 [00:00\u003c00:00, 2674.71it/s, Materializing param=wav2vec2.encoder.layers.16.attention.k_proj.weight]\rLoading weights:  62%|██████▏   | 262/424 [00:00\u003c00:00, 2670.97it/s, Materializing param=wav2vec2.encoder.layers.16.attention.k_proj.weight]\rLoading weights:  62%|██████▏   | 263/424 [00:00\u003c00:00, 2674.68it/s, Materializing param=wav2vec2.encoder.layers.16.attention.out_proj.bias]\rLoading weights:  62%|██████▏   | 263/424 [00:00\u003c00:00, 2672.22it/s, Materializing param=wav2vec2.encoder.layers.16.attention.out_proj.bias]\rLoading weights:  62%|██████▏   | 264/424 [00:00\u003c00:00, 2669.42it/s, Materializing param=wav2vec2.encoder.layers.16.attention.out_proj.weight]\rLoading weights:  62%|██████▏   | 264/424 [00:00\u003c00:00, 2666.86it/s, Materializing param=wav2vec2.encoder.layers.16.attention.out_proj.weight]\rLoading weights:  62%|██████▎   | 265/424 [00:00\u003c00:00, 2669.87it/s, Materializing param=wav2vec2.encoder.layers.16.attention.q_proj.bias]    \rLoading weights:  62%|██████▎   | 265/424 [00:00\u003c00:00, 2665.93it/s, Materializing param=wav2vec2.encoder.layers.16.attention.q_proj.bias]\rLoading weights:  63%|██████▎   | 266/424 [00:00\u003c00:00, 2669.51it/s, Materializing param=wav2vec2.encoder.layers.16.attention.q_proj.weight]\rLoading weights:  63%|██████▎   | 266/424 [00:00\u003c00:00, 2667.37it/s, Materializing param=wav2vec2.encoder.layers.16.attention.q_proj.weight]\rLoading weights:  63%|██████▎   | 267/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.attention.v_proj.bias]  \rLoading weights:  63%|██████▎   | 267/424 [00:00\u003c00:00, 2666.00it/s, Materializing param=wav2vec2.encoder.layers.16.attention.v_proj.bias]\rLoading weights:  63%|██████▎   | 268/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.attention.v_proj.bias]\rLoading weights:  63%|██████▎   | 268/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.attention.v_proj.weight]\rLoading weights:  63%|██████▎   | 268/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.attention.v_proj.weight]\rLoading weights:  63%|██████▎   | 269/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.intermediate_dense.bias]\rLoading weights:  63%|██████▎   | 269/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.intermediate_dense.bias]\rLoading weights:  64%|██████▎   | 270/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.intermediate_dense.weight]\rLoading weights:  64%|██████▎   | 270/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.intermediate_dense.weight]\rLoading weights:  64%|██████▍   | 271/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.output_dense.bias]        \rLoading weights:  64%|██████▍   | 271/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.output_dense.bias]\rLoading weights:  64%|██████▍   | 272/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.output_dense.weight]\rLoading weights:  64%|██████▍   | 272/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.feed_forward.output_dense.weight]\rLoading weights:  64%|██████▍   | 273/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.final_layer_norm.bias]           \rLoading weights:  64%|██████▍   | 273/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.final_layer_norm.bias]\rLoading weights:  65%|██████▍   | 274/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.final_layer_norm.weight]\rLoading weights:  65%|██████▍   | 274/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.final_layer_norm.weight]\rLoading weights:  65%|██████▍   | 275/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.layer_norm.bias]        \rLoading weights:  65%|██████▍   | 275/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.layer_norm.bias]\rLoading weights:  65%|██████▌   | 276/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.layer_norm.weight]\rLoading weights:  65%|██████▌   | 276/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.16.layer_norm.weight]\rLoading weights:  65%|██████▌   | 277/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.k_proj.bias]\rLoading weights:  65%|██████▌   | 277/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.k_proj.bias]\rLoading weights:  66%|██████▌   | 278/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.k_proj.weight]\rLoading weights:  66%|██████▌   | 278/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.k_proj.weight]\rLoading weights:  66%|██████▌   | 279/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.out_proj.bias]\rLoading weights:  66%|██████▌   | 279/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.out_proj.bias]\rLoading weights:  66%|██████▌   | 280/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.out_proj.weight]\rLoading weights:  66%|██████▌   | 280/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.out_proj.weight]\rLoading weights:  66%|██████▋   | 281/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.q_proj.bias]    \rLoading weights:  66%|██████▋   | 281/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.q_proj.bias]\rLoading weights:  67%|██████▋   | 282/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.q_proj.weight]\rLoading weights:  67%|██████▋   | 282/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.q_proj.weight]\rLoading weights:  67%|██████▋   | 283/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.v_proj.bias]  \rLoading weights:  67%|██████▋   | 283/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.v_proj.bias]\rLoading weights:  67%|██████▋   | 284/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.v_proj.weight]\rLoading weights:  67%|██████▋   | 284/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.attention.v_proj.weight]\rLoading weights:  67%|██████▋   | 285/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.intermediate_dense.bias]\rLoading weights:  67%|██████▋   | 285/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.intermediate_dense.bias]\rLoading weights:  67%|██████▋   | 286/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.intermediate_dense.weight]\rLoading weights:  67%|██████▋   | 286/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.intermediate_dense.weight]\rLoading weights:  68%|██████▊   | 287/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.output_dense.bias]        \rLoading weights:  68%|██████▊   | 287/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.output_dense.bias]\rLoading weights:  68%|██████▊   | 288/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.output_dense.weight]\rLoading weights:  68%|██████▊   | 288/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.feed_forward.output_dense.weight]\rLoading weights:  68%|██████▊   | 289/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.final_layer_norm.bias]           \rLoading weights:  68%|██████▊   | 289/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.final_layer_norm.bias]\rLoading weights:  68%|██████▊   | 290/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.final_layer_norm.weight]\rLoading weights:  68%|██████▊   | 290/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.final_layer_norm.weight]\rLoading weights:  69%|██████▊   | 291/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.layer_norm.bias]        \rLoading weights:  69%|██████▊   | 291/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.layer_norm.bias]\rLoading weights:  69%|██████▉   | 292/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.layer_norm.weight]\rLoading weights:  69%|██████▉   | 292/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.17.layer_norm.weight]\rLoading weights:  69%|██████▉   | 293/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.k_proj.bias]\rLoading weights:  69%|██████▉   | 293/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.k_proj.bias]\rLoading weights:  69%|██████▉   | 294/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.k_proj.weight]\rLoading weights:  69%|██████▉   | 294/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.k_proj.weight]\rLoading weights:  70%|██████▉   | 295/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.out_proj.bias]\rLoading weights:  70%|██████▉   | 295/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.out_proj.bias]\rLoading weights:  70%|██████▉   | 296/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.out_proj.weight]\rLoading weights:  70%|██████▉   | 296/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.out_proj.weight]\rLoading weights:  70%|███████   | 297/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.q_proj.bias]    \rLoading weights:  70%|███████   | 297/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.q_proj.bias]\rLoading weights:  70%|███████   | 298/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.q_proj.weight]\rLoading weights:  70%|███████   | 298/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.q_proj.weight]\rLoading weights:  71%|███████   | 299/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.v_proj.bias]  \rLoading weights:  71%|███████   | 299/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.v_proj.bias]\rLoading weights:  71%|███████   | 300/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.v_proj.weight]\rLoading weights:  71%|███████   | 300/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.attention.v_proj.weight]\rLoading weights:  71%|███████   | 301/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.intermediate_dense.bias]\rLoading weights:  71%|███████   | 301/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.intermediate_dense.bias]\rLoading weights:  71%|███████   | 302/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.intermediate_dense.weight]\rLoading weights:  71%|███████   | 302/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.intermediate_dense.weight]\rLoading weights:  71%|███████▏  | 303/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.output_dense.bias]        \rLoading weights:  71%|███████▏  | 303/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.output_dense.bias]\rLoading weights:  72%|███████▏  | 304/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.output_dense.weight]\rLoading weights:  72%|███████▏  | 304/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.feed_forward.output_dense.weight]\rLoading weights:  72%|███████▏  | 305/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.final_layer_norm.bias]           \rLoading weights:  72%|███████▏  | 305/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.final_layer_norm.bias]\rLoading weights:  72%|███████▏  | 306/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.final_layer_norm.weight]\rLoading weights:  72%|███████▏  | 306/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.final_layer_norm.weight]\rLoading weights:  72%|███████▏  | 307/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.layer_norm.bias]        \rLoading weights:  72%|███████▏  | 307/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.layer_norm.bias]\rLoading weights:  73%|███████▎  | 308/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.layer_norm.weight]\rLoading weights:  73%|███████▎  | 308/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.18.layer_norm.weight]\rLoading weights:  73%|███████▎  | 309/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.k_proj.bias]\rLoading weights:  73%|███████▎  | 309/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.k_proj.bias]\rLoading weights:  73%|███████▎  | 310/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.k_proj.weight]\rLoading weights:  73%|███████▎  | 310/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.k_proj.weight]\rLoading weights:  73%|███████▎  | 311/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.out_proj.bias]\rLoading weights:  73%|███████▎  | 311/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.out_proj.bias]\rLoading weights:  74%|███████▎  | 312/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.out_proj.weight]\rLoading weights:  74%|███████▎  | 312/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.out_proj.weight]\rLoading weights:  74%|███████▍  | 313/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.q_proj.bias]    \rLoading weights:  74%|███████▍  | 313/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.q_proj.bias]\rLoading weights:  74%|███████▍  | 314/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.q_proj.weight]\rLoading weights:  74%|███████▍  | 314/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.q_proj.weight]\rLoading weights:  74%|███████▍  | 315/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.v_proj.bias]  \rLoading weights:  74%|███████▍  | 315/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.v_proj.bias]\rLoading weights:  75%|███████▍  | 316/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.v_proj.weight]\rLoading weights:  75%|███████▍  | 316/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.attention.v_proj.weight]\rLoading weights:  75%|███████▍  | 317/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.intermediate_dense.bias]\rLoading weights:  75%|███████▍  | 317/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.intermediate_dense.bias]\rLoading weights:  75%|███████▌  | 318/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.intermediate_dense.weight]\rLoading weights:  75%|███████▌  | 318/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.intermediate_dense.weight]\rLoading weights:  75%|███████▌  | 319/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.output_dense.bias]        \rLoading weights:  75%|███████▌  | 319/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.output_dense.bias]\rLoading weights:  75%|███████▌  | 320/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.output_dense.weight]\rLoading weights:  75%|███████▌  | 320/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.feed_forward.output_dense.weight]\rLoading weights:  76%|███████▌  | 321/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.final_layer_norm.bias]           \rLoading weights:  76%|███████▌  | 321/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.final_layer_norm.bias]\rLoading weights:  76%|███████▌  | 322/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.final_layer_norm.weight]\rLoading weights:  76%|███████▌  | 322/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.final_layer_norm.weight]\rLoading weights:  76%|███████▌  | 323/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.layer_norm.bias]        \rLoading weights:  76%|███████▌  | 323/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.layer_norm.bias]\rLoading weights:  76%|███████▋  | 324/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.layer_norm.weight]\rLoading weights:  76%|███████▋  | 324/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.19.layer_norm.weight]\rLoading weights:  77%|███████▋  | 325/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.k_proj.bias]\rLoading weights:  77%|███████▋  | 325/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.k_proj.bias]\rLoading weights:  77%|███████▋  | 326/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.k_proj.weight]\rLoading weights:  77%|███████▋  | 326/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.k_proj.weight]\rLoading weights:  77%|███████▋  | 327/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.out_proj.bias]\rLoading weights:  77%|███████▋  | 327/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.out_proj.bias]\rLoading weights:  77%|███████▋  | 328/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.out_proj.weight]\rLoading weights:  77%|███████▋  | 328/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.out_proj.weight]\rLoading weights:  78%|███████▊  | 329/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.q_proj.bias]    \rLoading weights:  78%|███████▊  | 329/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.q_proj.bias]\rLoading weights:  78%|███████▊  | 330/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.q_proj.weight]\rLoading weights:  78%|███████▊  | 330/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.q_proj.weight]\rLoading weights:  78%|███████▊  | 331/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.v_proj.bias]  \rLoading weights:  78%|███████▊  | 331/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.v_proj.bias]\rLoading weights:  78%|███████▊  | 332/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.v_proj.weight]\rLoading weights:  78%|███████▊  | 332/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.attention.v_proj.weight]\rLoading weights:  79%|███████▊  | 333/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.intermediate_dense.bias]\rLoading weights:  79%|███████▊  | 333/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.intermediate_dense.bias]\rLoading weights:  79%|███████▉  | 334/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.intermediate_dense.weight]\rLoading weights:  79%|███████▉  | 334/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.intermediate_dense.weight]\rLoading weights:  79%|███████▉  | 335/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.output_dense.bias]        \rLoading weights:  79%|███████▉  | 335/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.output_dense.bias]\rLoading weights:  79%|███████▉  | 336/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.output_dense.weight]\rLoading weights:  79%|███████▉  | 336/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.feed_forward.output_dense.weight]\rLoading weights:  79%|███████▉  | 337/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.final_layer_norm.bias]           \rLoading weights:  79%|███████▉  | 337/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.final_layer_norm.bias]\rLoading weights:  80%|███████▉  | 338/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.final_layer_norm.weight]\rLoading weights:  80%|███████▉  | 338/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.final_layer_norm."}
,{"stream_name":"stderr","time":239.319005219,"data":"weight]\rLoading weights:  80%|███████▉  | 339/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.layer_norm.bias]        \rLoading weights:  80%|███████▉  | 339/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.layer_norm.bias]\rLoading weights:  80%|████████  | 340/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.layer_norm.weight]\rLoading weights:  80%|████████  | 340/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.20.layer_norm.weight]\rLoading weights:  80%|████████  | 341/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.k_proj.bias]\rLoading weights:  80%|████████  | 341/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.k_proj.bias]\rLoading weights:  81%|████████  | 342/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.k_proj.weight]\rLoading weights:  81%|████████  | 342/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.k_proj.weight]\rLoading weights:  81%|████████  | 343/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.out_proj.bias]\rLoading weights:  81%|████████  | 343/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.out_proj.bias]\rLoading weights:  81%|████████  | 344/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.out_proj.weight]\rLoading weights:  81%|████████  | 344/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.out_proj.weight]\rLoading weights:  81%|████████▏ | 345/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.q_proj.bias]    \rLoading weights:  81%|████████▏ | 345/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.q_proj.bias]\rLoading weights:  82%|████████▏ | 346/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.q_proj.weight]\rLoading weights:  82%|████████▏ | 346/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.q_proj.weight]\rLoading weights:  82%|████████▏ | 347/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.v_proj.bias]  \rLoading weights:  82%|████████▏ | 347/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.v_proj.bias]\rLoading weights:  82%|████████▏ | 348/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.v_proj.weight]\rLoading weights:  82%|████████▏ | 348/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.attention.v_proj.weight]\rLoading weights:  82%|████████▏ | 349/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.intermediate_dense.bias]\rLoading weights:  82%|████████▏ | 349/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.intermediate_dense.bias]\rLoading weights:  83%|████████▎ | 350/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.intermediate_dense.weight]\rLoading weights:  83%|████████▎ | 350/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.intermediate_dense.weight]\rLoading weights:  83%|████████▎ | 351/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.output_dense.bias]        \rLoading weights:  83%|████████▎ | 351/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.output_dense.bias]\rLoading weights:  83%|████████▎ | 352/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.output_dense.weight]\rLoading weights:  83%|████████▎ | 352/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.feed_forward.output_dense.weight]\rLoading weights:  83%|████████▎ | 353/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.final_layer_norm.bias]           \rLoading weights:  83%|████████▎ | 353/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.final_layer_norm.bias]\rLoading weights:  83%|████████▎ | 354/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.final_layer_norm.weight]\rLoading weights:  83%|████████▎ | 354/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.final_layer_norm.weight]\rLoading weights:  84%|████████▎ | 355/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.layer_norm.bias]        \rLoading weights:  84%|████████▎ | 355/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.layer_norm.bias]\rLoading weights:  84%|████████▍ | 356/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.layer_norm.weight]\rLoading weights:  84%|████████▍ | 356/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.21.layer_norm.weight]\rLoading weights:  84%|████████▍ | 357/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.k_proj.bias]\rLoading weights:  84%|████████▍ | 357/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.k_proj.bias]\rLoading weights:  84%|████████▍ | 358/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.k_proj.weight]\rLoading weights:  84%|████████▍ | 358/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.k_proj.weight]\rLoading weights:  85%|████████▍ | 359/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.out_proj.bias]\rLoading weights:  85%|████████▍ | 359/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.out_proj.bias]\rLoading weights:  85%|████████▍ | 360/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.out_proj.weight]\rLoading weights:  85%|████████▍ | 360/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.out_proj.weight]\rLoading weights:  85%|████████▌ | 361/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.q_proj.bias]    \rLoading weights:  85%|████████▌ | 361/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.q_proj.bias]\rLoading weights:  85%|████████▌ | 362/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.q_proj.weight]\rLoading weights:  85%|████████▌ | 362/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.q_proj.weight]\rLoading weights:  86%|████████▌ | 363/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.v_proj.bias]  \rLoading weights:  86%|████████▌ | 363/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.v_proj.bias]\rLoading weights:  86%|████████▌ | 364/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.v_proj.weight]\rLoading weights:  86%|████████▌ | 364/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.attention.v_proj.weight]\rLoading weights:  86%|████████▌ | 365/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.intermediate_dense.bias]\rLoading weights:  86%|████████▌ | 365/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.intermediate_dense.bias]\rLoading weights:  86%|████████▋ | 366/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.intermediate_dense.weight]\rLoading weights:  86%|████████▋ | 366/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.intermediate_dense.weight]\rLoading weights:  87%|████████▋ | 367/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.output_dense.bias]        \rLoading weights:  87%|████████▋ | 367/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.output_dense.bias]\rLoading weights:  87%|████████▋ | 368/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.output_dense.weight]\rLoading weights:  87%|████████▋ | 368/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.feed_forward.output_dense.weight]\rLoading weights:  87%|████████▋ | 369/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.final_layer_norm.bias]           \rLoading weights:  87%|████████▋ | 369/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.final_layer_norm.bias]\rLoading weights:  87%|████████▋ | 370/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.final_layer_norm.weight]\rLoading weights:  87%|████████▋ | 370/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.final_layer_norm.weight]\rLoading weights:  88%|████████▊ | 371/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.layer_norm.bias]        \rLoading weights:  88%|████████▊ | 371/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.layer_norm.bias]\rLoading weights:  88%|████████▊ | 372/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.layer_norm.weight]\rLoading weights:  88%|████████▊ | 372/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.22.layer_norm.weight]\rLoading weights:  88%|████████▊ | 373/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.k_proj.bias]\rLoading weights:  88%|████████▊ | 373/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.k_proj.bias]\rLoading weights:  88%|████████▊ | 374/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.k_proj.weight]\rLoading weights:  88%|████████▊ | 374/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.k_proj.weight]\rLoading weights:  88%|████████▊ | 375/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.out_proj.bias]\rLoading weights:  88%|████████▊ | 375/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.out_proj.bias]\rLoading weights:  89%|████████▊ | 376/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.out_proj.weight]\rLoading weights:  89%|████████▊ | 376/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.out_proj.weight]\rLoading weights:  89%|████████▉ | 377/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.q_proj.bias]    \rLoading weights:  89%|████████▉ | 377/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.q_proj.bias]\rLoading weights:  89%|████████▉ | 378/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.q_proj.weight]\rLoading weights:  89%|████████▉ | 378/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.q_proj.weight]\rLoading weights:  89%|████████▉ | 379/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.v_proj.bias]  \rLoading weights:  89%|████████▉ | 379/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.v_proj.bias]\rLoading weights:  90%|████████▉ | 380/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.v_proj.weight]\rLoading weights:  90%|████████▉ | 380/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.attention.v_proj.weight]\rLoading weights:  90%|████████▉ | 381/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.intermediate_dense.bias]\rLoading weights:  90%|████████▉ | 381/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.intermediate_dense.bias]\rLoading weights:  90%|█████████ | 382/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.intermediate_dense.weight]\rLoading weights:  90%|█████████ | 382/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.intermediate_dense.weight]\rLoading weights:  90%|█████████ | 383/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.output_dense.bias]        \rLoading weights:  90%|█████████ | 383/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.output_dense.bias]\rLoading weights:  91%|█████████ | 384/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.output_dense.weight]\rLoading weights:  91%|█████████ | 384/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.feed_forward.output_dense.weight]\rLoading weights:  91%|█████████ | 385/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.final_layer_norm.bias]           \rLoading weights:  91%|█████████ | 385/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.final_layer_norm.bias]\rLoading weights:  91%|█████████ | 386/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.final_layer_norm.weight]\rLoading weights:  91%|█████████ | 386/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.final_layer_norm.weight]\rLoading weights:  91%|█████████▏| 387/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.layer_norm.bias]        \rLoading weights:  91%|█████████▏| 387/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.layer_norm.bias]\rLoading weights:  92%|█████████▏| 388/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.layer_norm.weight]\rLoading weights:  92%|█████████▏| 388/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.layers.23.layer_norm.weight]\rLoading weights:  92%|█████████▏| 389/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.pos_conv_embed.conv.bias]   \rLoading weights:  92%|█████████▏| 389/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.pos_conv_embed.conv.bias]\rLoading weights:  92%|█████████▏| 390/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0]\rLoading weights:  92%|█████████▏| 390/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0]\rLoading weights:  92%|█████████▏| 391/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1]\rLoading weights:  92%|█████████▏| 391/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1]\rLoading weights:  92%|█████████▏| 392/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.conv.bias]                    \rLoading weights:  92%|█████████▏| 392/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.conv.bias]\rLoading weights:  93%|█████████▎| 393/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.conv.weight]\rLoading weights:  93%|█████████▎| 393/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.conv.weight]\rLoading weights:  93%|█████████▎| 394/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias]\rLoading weights:  93%|█████████▎| 394/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias]\rLoading weights:  93%|█████████▎| 395/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight]\rLoading weights:  93%|█████████▎| 395/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight]\rLoading weights:  93%|█████████▎| 396/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.conv.bias]        \rLoading weights:  93%|█████████▎| 396/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.conv.bias]\rLoading weights:  94%|█████████▎| 397/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.conv.weight]\rLoading weights:  94%|█████████▎| 397/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.conv.weight]\rLoading weights:  94%|█████████▍| 398/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.layer_norm.bias]\rLoading weights:  94%|█████████▍| 398/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.layer_norm.bias]\rLoading weights:  94%|█████████▍| 399/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.layer_norm.weight]\rLoading weights:  94%|█████████▍| 399/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.1.layer_norm.weight]\rLoading weights:  94%|█████████▍| 400/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.conv.bias]        \rLoading weights:  94%|█████████▍| 400/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.conv.bias]\rLoading weights:  95%|█████████▍| 401/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.conv.weight]\rLoading weights:  95%|█████████▍| 401/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.conv.weight]\rLoading weights:  95%|█████████▍| 402/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.layer_norm.bias]\rLoading weights:  95%|█████████▍| 402/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.layer_norm.bias]\rLoading weights:  95%|█████████▌| 403/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.layer_norm.weight]\rLoading weights:  95%|█████████▌| 403/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.2.layer_norm.weight]\rLoading weights:  95%|█████████▌| 404/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.conv.bias]        \rLoading weights:  95%|█████████▌| 404/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.conv.bias]\rLoading weights:  96%|█████████▌| 405/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.conv.weight]\rLoading weights:  96%|█████████▌| 405/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.conv.weight]\rLoading weights:  96%|█████████▌| 406/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.layer_norm.bias]\rLoading weights:  96%|█████████▌| 406/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.layer_norm.bias]\rLoading weights:  96%|█████████▌| 407/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.layer_norm.weight]\rLoading weights:  96%|█████████▌| 407/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.3.layer_norm.weight]\rLoading weights:  96%|█████████▌| 408/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.conv.bias]        \rLoading weights:  96%|█████████▌| 408/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.conv.bias]\rLoading weights:  96%|█████████▋| 409/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.conv.weight]\rLoading weights:  96%|█████████▋| 409/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.conv.weight]\rLoading weights:  97%|█████████▋| 410/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.layer_norm.bias]\rLoading weights:  97%|█████████▋| 410/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.layer_norm.bias]\rLoading weights:  97%|█████████▋| 411/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.layer_norm.weight]\rLoading weights:  97%|█████████▋| 411/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.4.layer_norm.weight]\rLoading weights:  97%|█████████▋| 412/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.conv.bias]        \rLoading weights:  97%|█████████▋| 412/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.conv.bias]\rLoading weights:  97%|█████████▋| 413/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.conv.weight]\rLoading weights:  97%|█████████▋| 413/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.conv.weight]\rLoading weights:  98%|█████████▊| 414/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.layer_norm.bias]\rLoading weights:  98%|█████████▊| 414/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.layer_norm.bias]\rLoading weights:  98%|█████████▊| 415/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.layer_norm.weight]\rLoading weights:  98%|█████████▊| 415/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.5.layer_norm.weight]\rLoading weights:  98%|█████████▊| 416/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.conv.bias]        \rLoading weights:  98%|█████████▊| 416/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.conv.bias]\rLoading weights:  98%|█████████▊| 417/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.conv.weight]\rLoading weights:  98%|█████████▊| 417/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.conv.weight]\rLoading weights:  99%|█████████▊| 418/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.layer_norm.bias]\rLoading weights:  99%|█████████▊| 418/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.layer_norm.bias]\rLoading weights:  99%|█████████▉| 419/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.layer_norm.weight]\rLoading weights:  99%|█████████▉| 419/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_extractor.conv_layers.6.layer_norm.weight]\rLoading weights:  99%|█████████▉| 420/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.layer_norm.bias]               \rLoading weights:  99%|█████████▉| 420/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.layer_norm.bias]\rLoading weights:  99%|█████████▉| 421/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.layer_norm.weight]\rLoading weights:  99%|█████████▉| 421/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.layer_norm.weight]\rLoading weights: 100%|█████████▉| 422/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.projection.bias]  \rLoading weights: 100%|█████████▉| 422/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.projection.bias]\rLoading weights: 100%|█████████▉| 423/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.projection.weight]\rLoading weights: 100%|█████████▉| 423/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.feature_projection.projection.weight]\rLoading weights: 100%|██████████| 424/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.masked_spec_embed]                   \rLoading weights: 100%|██████████| 424/424 [00:00\u003c00:00, 2669.86it/s, Materializing param=wav2vec2.masked_spec_embed]\rLoading weights: 100%|██████████| 424/424 [00:00\u003c00:00, 2462.65it/s, Materializing param=wav2vec2.masked_spec_embed]\n"}
,{"stream_name":"stdout","time":245.44225772,"data":"\n"}
,{"stream_name":"stdout","time":245.442334926,"data":"[4/4] 話者分離を実行中 (話者数: 2〜2 人固定)...\n"}
,{"stream_name":"stdout","time":245.448401098,"data":"\n"}
,{"stream_name":"stdout","time":245.44842609,"data":"--- 再分割後の会話レポート（一部サンプル） ---\n"}
,{"stream_name":"stdout","time":245.448433795,"data":"---------------------------------------------\n"}
,{"stream_name":"stdout","time":245.44843977,"data":"\n"}
,{"stream_name":"stdout","time":245.44844518,"data":"✅ 会話レポート保存: /kaggle/working/whisperx_results/transcription_report.txt\n"}
,{"stream_name":"stdout","time":245.448450266,"data":"✅ 5〜10秒のWAV候補ファイル (0件): /kaggle/working/whisperx_results/ref_audio_candidates\n"}
,{"stream_name":"stdout","time":245.448455152,"data":"\n"}
,{"stream_name":"stdout","time":245.448459613,"data":"==========================================\n"}
,{"stream_name":"stdout","time":245.448464204,"data":" 全処理が正常に完了しました！\n"}
,{"stream_name":"stdout","time":245.448470756,"data":"==========================================\n"}
,{"stream_name":"stderr","time":254.210262606,"data":"/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\\|'\n"}
,{"stream_name":"stderr","time":254.210307741,"data":"  cells[i][c] = re.sub('\\\\\\\\\\|', '|', cell)\n"}
,{"stream_name":"stderr","time":254.569862916,"data":"/usr/local/lib/python3.12/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\\_'\n"}
,{"stream_name":"stderr","time":254.569907971,"data":"  text = re.sub(r'_', '\\_', text) # Escape underscores in display text\n"}
,{"stream_name":"stderr","time":255.371714342,"data":"[NbConvertApp] Converting notebook __script__.ipynb to html\n"}
,{"stream_name":"stderr","time":256.277745696,"data":"[NbConvertApp] Writing 309174 bytes to __results__.html\n"}
]```

---

## output/whisperx_results/transcription_report.txt
```text
=================================================================
  WhisperX (large-v3) 話者分離会話レポート (話者交代ごとに再分割済)
=================================================================

[00:00:04.063 -> 00:04:10.380] (246.3秒) [SPEAKER_UNKNOWN]: おはようございますおはようございますそしたらまずは自己紹介をお願いいたします青井壮です青井壮ちゃん今おいくつですか18歳です18歳すごい若いですね先日デビューの撮影を行ったと思うんですけど緊張しました。どこら辺が緊張しましたか?初めてだったんで、全部が緊張です。ダニエルさんとのケッチはどうでしたか?気持ち良かったです。そうですね。今日は2本目の撮影です。緊張してますか?この前よりはしてないんですけど、少ししてます。そんなにないです。そんなにない。なんかね、見てるとそんなに緊張してない感じで。今日はリラックスして撮影できそうですか?はい。でね、2本目の今日は初めてのことをやりたいと思います。はい。それは、わかんないです今日は人生初の仲出しをしてみたいと思います仲出しって何かイメージあります?あんまりしちゃいけないそうそうしないじゃないですかはいただ今日はね初中出しコンドームをねしないおちんちんが入ってきてそして最後はね中に出したいと思いますはい大丈夫ですかなんか怖がってきてませんかうーんちょっと緊張してきました大丈夫です今日はいろいろとね初めてのことをたくさんしていきたいと思いますのでよろしくお願いいたしますお願いします最後2本目の撮影の意気込みを撮影スタートしていきましょうはい楽しめたらいいなって思いますよろしくお願いしますお願いします```

---

## output/whisperx_results/transcription_data.json
```json
[
  {
    "start": 4.063,
    "end": 27.936,
    "text": "おはようございますおはようございますそしたらまずは自己紹介をお願いいたします青井壮です青井壮ちゃん今おいくつですか18歳です18歳すごい若いですね先日デビューの撮影を行ったと思うんですけど",
    "words": [
      {
        "word": "お",
        "start": 4.063,
        "end": 4.083,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 4.083,
        "end": 4.103,
        "score": 0.0
      },
      {
        "word": "よ",
        "start": 4.103,
        "end": 4.123,
        "score": 0.0
      },
      {
        "word": "う",
        "start": 4.123,
        "end": 4.143,
        "score": 0.0
      },
      {
        "word": "ご",
        "start": 4.143,
        "end": 4.163,
        "score": 0.0
      },
      {
        "word": "ざ",
        "start": 4.163,
        "end": 4.183,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 4.183,
        "end": 4.203,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 4.203,
        "end": 4.223,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 4.223,
        "end": 4.263,
        "score": 0.5
      },
      {
        "word": "お",
        "start": 4.263,
        "end": 4.283,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 4.283,
        "end": 4.303,
        "score": 0.0
      },
      {
        "word": "よ",
        "start": 4.303,
        "end": 4.323,
        "score": 0.0
      },
      {
        "word": "う",
        "start": 4.323,
        "end": 4.343,
        "score": 0.0
      },
      {
        "word": "ご",
        "start": 4.343,
        "end": 4.363,
        "score": 0.0
      },
      {
        "word": "ざ",
        "start": 4.363,
        "end": 4.383,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 4.383,
        "end": 4.403,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 4.403,
        "end": 4.423,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 4.423,
        "end": 4.443,
        "score": 0.0
      },
      {
        "word": "そ",
        "start": 4.443,
        "end": 8.346,
        "score": 0.995
      },
      {
        "word": "し",
        "start": 8.346,
        "end": 8.426,
        "score": 0.75
      },
      {
        "word": "た",
        "start": 8.426,
        "end": 9.246,
        "score": 0.976
      },
      {
        "word": "ら",
        "start": 9.246,
        "end": 9.266,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 9.266,
        "end": 9.306,
        "score": 0.5
      },
      {
        "word": "ず",
        "start": 9.306,
        "end": 9.326,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 9.326,
        "end": 9.346,
        "score": 0.0
      },
      {
        "word": "自",
        "start": 9.346,
        "end": 9.366,
        "score": 0.0
      },
      {
        "word": "己",
        "start": 9.366,
        "end": 9.386,
        "score": 0.0
      },
      {
        "word": "紹",
        "start": 9.386,
        "end": 9.466,
        "score": 0.75
      },
      {
        "word": "介",
        "start": 9.466,
        "end": 9.486,
        "score": 0.0
      },
      {
        "word": "を",
        "start": 9.486,
        "end": 9.506,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 9.506,
        "end": 9.526,
        "score": 0.041
      },
      {
        "word": "願",
        "start": 9.526,
        "end": 9.606,
        "score": 0.75
      },
      {
        "word": "い",
        "start": 9.606,
        "end": 9.626,
        "score": 0.992
      },
      {
        "word": "い",
        "start": 9.626,
        "end": 9.726,
        "score": 0.838
      },
      {
        "word": "た",
        "start": 9.726,
        "end": 9.746,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 9.746,
        "end": 9.766,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 9.766,
        "end": 9.786,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 9.786,
        "end": 9.806,
        "score": 0.0
      },
      {
        "word": "青",
        "start": 9.806,
        "end": 9.826,
        "score": 0.0
      },
      {
        "word": "井",
        "start": 9.826,
        "end": 9.846,
        "score": 0.0
      },
      {
        "word": "壮",
        "start": 9.846,
        "end": 10.026,
        "score": 0.875
      },
      {
        "word": "で",
        "start": 10.026,
        "end": 10.207,
        "score": 0.909
      },
      {
        "word": "す",
        "start": 10.207,
        "end": 15.169,
        "score": 0.996
      },
      {
        "word": "青",
        "start": 15.169,
        "end": 15.189,
        "score": 0.0
      },
      {
        "word": "井",
        "start": 15.189,
        "end": 15.209,
        "score": 0.0
      },
      {
        "word": "壮",
        "start": 15.209,
        "end": 15.229,
        "score": 0.0
      },
      {
        "word": "ち",
        "start": 15.229,
        "end": 15.249,
        "score": 0.0
      },
      {
        "word": "ゃ",
        "start": 15.249,
        "end": 15.269,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 15.269,
        "end": 15.289,
        "score": 0.0
      },
      {
        "word": "今",
        "start": 15.289,
        "end": 15.309,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 15.309,
        "end": 15.329,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 15.329,
        "end": 15.349,
        "score": 0.0
      },
      {
        "word": "く",
        "start": 15.349,
        "end": 15.369,
        "score": 0.0
      },
      {
        "word": "つ",
        "start": 15.369,
        "end": 15.389,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 15.389,
        "end": 15.409,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 15.409,
        "end": 15.429,
        "score": 0.0
      },
      {
        "word": "か",
        "start": 15.429,
        "end": 15.529,
        "score": 0.8
      },
      {
        "word": "1",
        "start": 15.529,
        "end": 15.549,
        "score": 0.0
      },
      {
        "word": "8",
        "start": 15.549,
        "end": 15.569,
        "score": 0.0
      },
      {
        "word": "歳",
        "start": 15.569,
        "end": 15.589,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 15.589,
        "end": 15.609,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 15.609,
        "end": 15.629,
        "score": 0.0
      },
      {
        "word": "1",
        "start": 15.629,
        "end": 15.649,
        "score": 0.0
      },
      {
        "word": "8",
        "start": 15.649,
        "end": 15.669,
        "score": 0.0
      },
      {
        "word": "歳",
        "start": 15.669,
        "end": 15.689,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 15.689,
        "end": 15.709,
        "score": 0.0
      },
      {
        "word": "ご",
        "start": 15.709,
        "end": 15.729,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 15.729,
        "end": 15.749,
        "score": 0.0
      },
      {
        "word": "若",
        "start": 15.749,
        "end": 15.769,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 15.769,
        "end": 15.79,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 15.79,
        "end": 15.81,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 15.81,
        "end": 15.83,
        "score": 0.0
      },
      {
        "word": "ね",
        "start": 15.83,
        "end": 15.85,
        "score": 0.0
      },
      {
        "word": "先",
        "start": 15.85,
        "end": 15.87,
        "score": 0.0
      },
      {
        "word": "日",
        "start": 15.87,
        "end": 15.89,
        "score": 0.0
      },
      {
        "word": "デ",
        "start": 15.89,
        "end": 15.91,
        "score": 0.0
      },
      {
        "word": "ビ",
        "start": 15.91,
        "end": 15.93,
        "score": 0.0
      },
      {
        "word": "ュ",
        "start": 15.93,
        "end": 15.99,
        "score": 0.667
      },
      {
        "word": "ー",
        "start": 15.99,
        "end": 19.752,
        "score": 0.995
      },
      {
        "word": "の",
        "start": 19.752,
        "end": 19.812,
        "score": 0.667
      },
      {
        "word": "撮",
        "start": 19.812,
        "end": 19.912,
        "score": 0.8
      },
      {
        "word": "影",
        "start": 19.912,
        "end": 25.455,
        "score": 0.996
      },
      {
        "word": "を",
        "start": 25.455,
        "end": 25.475,
        "score": 0.0
      },
      {
        "word": "行",
        "start": 25.475,
        "end": 26.315,
        "score": 0.976
      },
      {
        "word": "っ",
        "start": 26.315,
        "end": 27.015,
        "score": 0.971
      },
      {
        "word": "た",
        "start": 27.015,
        "end": 27.076,
        "score": 0.667
      },
      {
        "word": "と",
        "start": 27.076,
        "end": 27.176,
        "score": 0.8
      },
      {
        "word": "思",
        "start": 27.176,
        "end": 27.196,
        "score": 0.0
      },
      {
        "word": "う",
        "start": 27.196,
        "end": 27.356,
        "score": 0.875
      },
      {
        "word": "ん",
        "start": 27.356,
        "end": 27.396,
        "score": 0.5
      },
      {
        "word": "で",
        "start": 27.396,
        "end": 27.876,
        "score": 0.958
      },
      {
        "word": "す",
        "start": 27.876,
        "end": 27.896,
        "score": 0.0
      },
      {
        "word": "け",
        "start": 27.896,
        "end": 27.916,
        "score": 0.0
      },
      {
        "word": "ど",
        "start": 27.916,
        "end": 27.936,
        "score": 0.0
      }
    ],
    "avg_logprob": -0.0906599811208782
  },
  {
    "start": 39.242,
    "end": 61.169,
    "text": "緊張しました。どこら辺が緊張しましたか?初めてだったんで、全部が緊張です。ダニエルさんとのケッチはどうでしたか?気持ち良かったです。",
    "words": [
      {
        "word": "緊",
        "start": 39.242,
        "end": 39.262,
        "score": 0.0
      },
      {
        "word": "張",
        "start": 39.262,
        "end": 39.282,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 39.282,
        "end": 39.302,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 39.302,
        "end": 39.342,
        "score": 0.5
      },
      {
        "word": "し",
        "start": 39.342,
        "end": 39.502,
        "score": 0.875
      },
      {
        "word": "た",
        "start": 39.502,
        "end": 39.522,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 39.522,
        "end": 39.542,
        "score": 0.812
      },
      {
        "word": "ど",
        "start": 39.542,
        "end": 39.683,
        "score": 0.857
      },
      {
        "word": "こ",
        "start": 39.683,
        "end": 39.703,
        "score": 0.0
      },
      {
        "word": "ら",
        "start": 39.703,
        "end": 39.723,
        "score": 0.0
      },
      {
        "word": "辺",
        "start": 39.723,
        "end": 39.823,
        "score": 0.8
      },
      {
        "word": "が",
        "start": 39.823,
        "end": 39.863,
        "score": 0.5
      },
      {
        "word": "緊",
        "start": 39.863,
        "end": 39.883,
        "score": 0.0
      },
      {
        "word": "張",
        "start": 39.883,
        "end": 40.003,
        "score": 0.833
      },
      {
        "word": "し",
        "start": 40.003,
        "end": 40.023,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 40.023,
        "end": 40.063,
        "score": 0.5
      },
      {
        "word": "し",
        "start": 40.063,
        "end": 40.083,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 40.083,
        "end": 45.164,
        "score": 0.996
      },
      {
        "word": "か",
        "start": 45.164,
        "end": 45.184,
        "score": 0.0
      },
      {
        "word": "?",
        "start": 45.184,
        "end": 45.364,
        "score": 0.953
      },
      {
        "word": "初",
        "start": 45.364,
        "end": 45.524,
        "score": 0.875
      },
      {
        "word": "め",
        "start": 45.524,
        "end": 45.784,
        "score": 0.923
      },
      {
        "word": "て",
        "start": 45.784,
        "end": 45.924,
        "score": 0.857
      },
      {
        "word": "だ",
        "start": 45.924,
        "end": 46.064,
        "score": 0.857
      },
      {
        "word": "っ",
        "start": 46.064,
        "end": 46.164,
        "score": 0.8
      },
      {
        "word": "た",
        "start": 46.164,
        "end": 46.184,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 46.184,
        "end": 46.485,
        "score": 0.933
      },
      {
        "word": "で",
        "start": 46.485,
        "end": 49.045,
        "score": 0.992
      },
      {
        "word": "、",
        "start": 49.045,
        "end": 49.065,
        "score": 0.413
      },
      {
        "word": "全",
        "start": 49.065,
        "end": 49.325,
        "score": 0.923
      },
      {
        "word": "部",
        "start": 49.325,
        "end": 49.445,
        "score": 0.833
      },
      {
        "word": "が",
        "start": 49.445,
        "end": 49.465,
        "score": 0.0
      },
      {
        "word": "緊",
        "start": 49.465,
        "end": 49.485,
        "score": 0.0
      },
      {
        "word": "張",
        "start": 49.485,
        "end": 49.626,
        "score": 0.857
      },
      {
        "word": "で",
        "start": 49.626,
        "end": 49.646,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 49.646,
        "end": 49.886,
        "score": 0.917
      },
      {
        "word": "。",
        "start": 49.886,
        "end": 49.906,
        "score": 0.006
      },
      {
        "word": "ダ",
        "start": 49.906,
        "end": 50.126,
        "score": 0.908
      },
      {
        "word": "ニ",
        "start": 50.126,
        "end": 50.146,
        "score": 0.0
      },
      {
        "word": "エ",
        "start": 50.146,
        "end": 50.246,
        "score": 0.8
      },
      {
        "word": "ル",
        "start": 50.246,
        "end": 50.266,
        "score": 0.0
      },
      {
        "word": "さ",
        "start": 50.266,
        "end": 59.208,
        "score": 0.998
      },
      {
        "word": "ん",
        "start": 59.208,
        "end": 59.368,
        "score": 0.875
      },
      {
        "word": "と",
        "start": 59.368,
        "end": 59.388,
        "score": 0.0
      },
      {
        "word": "の",
        "start": 59.388,
        "end": 59.408,
        "score": 0.0
      },
      {
        "word": "ケ",
        "start": 59.408,
        "end": 59.428,
        "score": 0.0
      },
      {
        "word": "ッ",
        "start": 59.428,
        "end": 59.468,
        "score": 0.5
      },
      {
        "word": "チ",
        "start": 59.468,
        "end": 59.488,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 59.488,
        "end": 59.509,
        "score": 0.0
      },
      {
        "word": "ど",
        "start": 59.509,
        "end": 59.529,
        "score": 0.0
      },
      {
        "word": "う",
        "start": 59.529,
        "end": 59.549,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 59.549,
        "end": 59.569,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 59.569,
        "end": 59.589,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 59.589,
        "end": 59.609,
        "score": 0.0
      },
      {
        "word": "か",
        "start": 59.609,
        "end": 59.629,
        "score": 0.0
      },
      {
        "word": "?",
        "start": 59.629,
        "end": 59.649,
        "score": 0.0
      },
      {
        "word": "気",
        "start": 59.649,
        "end": 59.669,
        "score": 0.0
      },
      {
        "word": "持",
        "start": 59.669,
        "end": 59.849,
        "score": 0.889
      },
      {
        "word": "ち",
        "start": 59.849,
        "end": 59.869,
        "score": 0.0
      },
      {
        "word": "良",
        "start": 59.869,
        "end": 60.089,
        "score": 0.908
      },
      {
        "word": "か",
        "start": 60.089,
        "end": 60.289,
        "score": 0.9
      },
      {
        "word": "っ",
        "start": 60.289,
        "end": 60.309,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 60.309,
        "end": 60.489,
        "score": 0.889
      },
      {
        "word": "で",
        "start": 60.489,
        "end": 60.509,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 60.509,
        "end": 61.149,
        "score": 0.966
      },
      {
        "word": "。",
        "start": 61.149,
        "end": 61.169,
        "score": 1.0
      }
    ],
    "avg_logprob": -0.1810937476158142
  },
  {
    "start": 81.648,
    "end": 96.412,
    "text": "そうですね。今日は2本目の撮影です。緊張してますか?この前よりはしてないんですけど、少ししてます。",
    "words": [
      {
        "word": "そ",
        "start": 81.648,
        "end": 81.808,
        "score": 0.915
      },
      {
        "word": "う",
        "start": 81.808,
        "end": 81.989,
        "score": 0.872
      },
      {
        "word": "で",
        "start": 81.989,
        "end": 93.247,
        "score": 0.998
      },
      {
        "word": "す",
        "start": 93.247,
        "end": 93.487,
        "score": 0.917
      },
      {
        "word": "ね",
        "start": 93.487,
        "end": 93.728,
        "score": 0.916
      },
      {
        "word": "。",
        "start": 93.728,
        "end": 93.848,
        "score": 0.986
      },
      {
        "word": "今",
        "start": 93.848,
        "end": 93.908,
        "score": 0.667
      },
      {
        "word": "日",
        "start": 93.908,
        "end": 93.948,
        "score": 0.5
      },
      {
        "word": "は",
        "start": 93.948,
        "end": 94.128,
        "score": 0.993
      },
      {
        "word": "2",
        "start": 94.128,
        "end": 94.268,
        "score": 0.853
      },
      {
        "word": "本",
        "start": 94.268,
        "end": 94.288,
        "score": 0.0
      },
      {
        "word": "目",
        "start": 94.288,
        "end": 94.369,
        "score": 0.623
      },
      {
        "word": "の",
        "start": 94.369,
        "end": 94.429,
        "score": 0.534
      },
      {
        "word": "撮",
        "start": 94.429,
        "end": 94.569,
        "score": 0.857
      },
      {
        "word": "影",
        "start": 94.569,
        "end": 94.689,
        "score": 0.833
      },
      {
        "word": "で",
        "start": 94.689,
        "end": 94.789,
        "score": 0.8
      },
      {
        "word": "す",
        "start": 94.789,
        "end": 95.05,
        "score": 0.92
      },
      {
        "word": "。",
        "start": 95.05,
        "end": 95.21,
        "score": 0.873
      },
      {
        "word": "緊",
        "start": 95.21,
        "end": 95.23,
        "score": 0.0
      },
      {
        "word": "張",
        "start": 95.23,
        "end": 95.29,
        "score": 0.665
      },
      {
        "word": "し",
        "start": 95.29,
        "end": 95.31,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 95.31,
        "end": 95.33,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 95.33,
        "end": 95.35,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 95.35,
        "end": 95.43,
        "score": 0.749
      },
      {
        "word": "か",
        "start": 95.43,
        "end": 95.45,
        "score": 0.0
      },
      {
        "word": "?",
        "start": 95.45,
        "end": 95.47,
        "score": 0.04
      },
      {
        "word": "こ",
        "start": 95.47,
        "end": 95.51,
        "score": 0.5
      },
      {
        "word": "の",
        "start": 95.51,
        "end": 95.53,
        "score": 0.0
      },
      {
        "word": "前",
        "start": 95.53,
        "end": 95.551,
        "score": 0.0
      },
      {
        "word": "よ",
        "start": 95.551,
        "end": 95.571,
        "score": 0.0
      },
      {
        "word": "り",
        "start": 95.571,
        "end": 95.591,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 95.591,
        "end": 95.651,
        "score": 0.665
      },
      {
        "word": "し",
        "start": 95.651,
        "end": 95.711,
        "score": 0.977
      },
      {
        "word": "て",
        "start": 95.711,
        "end": 95.731,
        "score": 0.0
      },
      {
        "word": "な",
        "start": 95.731,
        "end": 95.751,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 95.751,
        "end": 95.771,
        "score": 0.006
      },
      {
        "word": "ん",
        "start": 95.771,
        "end": 95.791,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 95.791,
        "end": 95.811,
        "score": 0.05
      },
      {
        "word": "す",
        "start": 95.811,
        "end": 95.831,
        "score": 0.0
      },
      {
        "word": "け",
        "start": 95.831,
        "end": 95.851,
        "score": 0.0
      },
      {
        "word": "ど",
        "start": 95.851,
        "end": 95.871,
        "score": 0.0
      },
      {
        "word": "、",
        "start": 95.871,
        "end": 95.891,
        "score": 0.003
      },
      {
        "word": "少",
        "start": 95.891,
        "end": 95.911,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 95.911,
        "end": 95.931,
        "score": 0.055
      },
      {
        "word": "し",
        "start": 95.931,
        "end": 96.051,
        "score": 0.858
      },
      {
        "word": "て",
        "start": 96.051,
        "end": 96.172,
        "score": 0.834
      },
      {
        "word": "ま",
        "start": 96.172,
        "end": 96.352,
        "score": 0.889
      },
      {
        "word": "す",
        "start": 96.352,
        "end": 96.392,
        "score": 0.492
      },
      {
        "word": "。",
        "start": 96.392,
        "end": 96.412,
        "score": 0.932
      }
    ],
    "avg_logprob": -0.06721047531155978
  },
  {
    "start": 111.794,
    "end": 136.685,
    "text": "そんなにないです。そんなにない。なんかね、見てるとそんなに緊張してない感じで。今日はリラックスして撮影できそうですか?はい。でね、2本目の今日は初めてのことをやりたいと思います。はい。それは、",
    "words": [
      {
        "word": "そ",
        "start": 111.794,
        "end": 111.814,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 111.814,
        "end": 111.854,
        "score": 0.5
      },
      {
        "word": "な",
        "start": 111.854,
        "end": 111.874,
        "score": 0.0
      },
      {
        "word": "に",
        "start": 111.874,
        "end": 111.894,
        "score": 0.0
      },
      {
        "word": "な",
        "start": 111.894,
        "end": 112.014,
        "score": 0.833
      },
      {
        "word": "い",
        "start": 112.014,
        "end": 112.054,
        "score": 0.5
      },
      {
        "word": "で",
        "start": 112.054,
        "end": 112.074,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 112.074,
        "end": 112.094,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 112.094,
        "end": 112.114,
        "score": 0.0
      },
      {
        "word": "そ",
        "start": 112.114,
        "end": 112.134,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 112.134,
        "end": 112.935,
        "score": 0.975
      },
      {
        "word": "な",
        "start": 112.935,
        "end": 113.355,
        "score": 0.952
      },
      {
        "word": "に",
        "start": 113.355,
        "end": 113.375,
        "score": 0.0
      },
      {
        "word": "な",
        "start": 113.375,
        "end": 113.395,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 113.395,
        "end": 113.415,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 113.415,
        "end": 113.475,
        "score": 0.667
      },
      {
        "word": "な",
        "start": 113.475,
        "end": 113.535,
        "score": 0.667
      },
      {
        "word": "ん",
        "start": 113.535,
        "end": 114.475,
        "score": 0.979
      },
      {
        "word": "か",
        "start": 114.475,
        "end": 116.716,
        "score": 0.991
      },
      {
        "word": "ね",
        "start": 116.716,
        "end": 116.736,
        "score": 0.0
      },
      {
        "word": "、",
        "start": 116.736,
        "end": 116.756,
        "score": 0.0
      },
      {
        "word": "見",
        "start": 116.756,
        "end": 118.617,
        "score": 0.989
      },
      {
        "word": "て",
        "start": 118.617,
        "end": 118.637,
        "score": 0.0
      },
      {
        "word": "る",
        "start": 118.637,
        "end": 118.657,
        "score": 0.0
      },
      {
        "word": "と",
        "start": 118.657,
        "end": 119.477,
        "score": 0.976
      },
      {
        "word": "そ",
        "start": 119.477,
        "end": 119.758,
        "score": 0.929
      },
      {
        "word": "ん",
        "start": 119.758,
        "end": 119.918,
        "score": 0.875
      },
      {
        "word": "な",
        "start": 119.918,
        "end": 120.078,
        "score": 0.875
      },
      {
        "word": "に",
        "start": 120.078,
        "end": 120.358,
        "score": 0.929
      },
      {
        "word": "緊",
        "start": 120.358,
        "end": 120.458,
        "score": 0.8
      },
      {
        "word": "張",
        "start": 120.458,
        "end": 120.578,
        "score": 0.833
      },
      {
        "word": "し",
        "start": 120.578,
        "end": 121.318,
        "score": 0.973
      },
      {
        "word": "て",
        "start": 121.318,
        "end": 121.338,
        "score": 0.0
      },
      {
        "word": "な",
        "start": 121.338,
        "end": 121.358,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 121.358,
        "end": 121.378,
        "score": 0.0
      },
      {
        "word": "感",
        "start": 121.378,
        "end": 121.398,
        "score": 0.0
      },
      {
        "word": "じ",
        "start": 121.398,
        "end": 123.799,
        "score": 0.992
      },
      {
        "word": "で",
        "start": 123.799,
        "end": 123.819,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 123.819,
        "end": 123.839,
        "score": 0.0
      },
      {
        "word": "今",
        "start": 123.839,
        "end": 123.859,
        "score": 0.0
      },
      {
        "word": "日",
        "start": 123.859,
        "end": 124.079,
        "score": 0.909
      },
      {
        "word": "は",
        "start": 124.079,
        "end": 124.099,
        "score": 0.0
      },
      {
        "word": "リ",
        "start": 124.099,
        "end": 124.119,
        "score": 0.0
      },
      {
        "word": "ラ",
        "start": 124.119,
        "end": 124.139,
        "score": 0.0
      },
      {
        "word": "ッ",
        "start": 124.139,
        "end": 124.34,
        "score": 0.9
      },
      {
        "word": "ク",
        "start": 124.34,
        "end": 124.36,
        "score": 0.0
      },
      {
        "word": "ス",
        "start": 124.36,
        "end": 124.38,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 124.38,
        "end": 124.52,
        "score": 0.857
      },
      {
        "word": "て",
        "start": 124.52,
        "end": 124.7,
        "score": 0.889
      },
      {
        "word": "撮",
        "start": 124.7,
        "end": 126.26,
        "score": 0.987
      },
      {
        "word": "影",
        "start": 126.26,
        "end": 126.3,
        "score": 0.5
      },
      {
        "word": "で",
        "start": 126.3,
        "end": 128.001,
        "score": 0.988
      },
      {
        "word": "き",
        "start": 128.001,
        "end": 128.201,
        "score": 0.9
      },
      {
        "word": "そ",
        "start": 128.201,
        "end": 128.381,
        "score": 0.889
      },
      {
        "word": "う",
        "start": 128.381,
        "end": 128.441,
        "score": 0.667
      },
      {
        "word": "で",
        "start": 128.441,
        "end": 128.461,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 128.461,
        "end": 128.521,
        "score": 0.667
      },
      {
        "word": "か",
        "start": 128.521,
        "end": 128.581,
        "score": 0.667
      },
      {
        "word": "?",
        "start": 128.581,
        "end": 128.601,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 128.601,
        "end": 128.802,
        "score": 0.9
      },
      {
        "word": "い",
        "start": 128.802,
        "end": 128.822,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 128.822,
        "end": 128.842,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 128.842,
        "end": 128.882,
        "score": 0.5
      },
      {
        "word": "ね",
        "start": 128.882,
        "end": 128.962,
        "score": 0.75
      },
      {
        "word": "、",
        "start": 128.962,
        "end": 128.982,
        "score": 0.0
      },
      {
        "word": "2",
        "start": 128.982,
        "end": 129.002,
        "score": 0.0
      },
      {
        "word": "本",
        "start": 129.002,
        "end": 129.022,
        "score": 0.0
      },
      {
        "word": "目",
        "start": 129.022,
        "end": 129.042,
        "score": 0.0
      },
      {
        "word": "の",
        "start": 129.042,
        "end": 129.062,
        "score": 0.0
      },
      {
        "word": "今",
        "start": 129.062,
        "end": 129.742,
        "score": 0.971
      },
      {
        "word": "日",
        "start": 129.742,
        "end": 130.022,
        "score": 0.929
      },
      {
        "word": "は",
        "start": 130.022,
        "end": 130.042,
        "score": 0.0
      },
      {
        "word": "初",
        "start": 130.042,
        "end": 131.963,
        "score": 0.99
      },
      {
        "word": "め",
        "start": 131.963,
        "end": 132.063,
        "score": 0.8
      },
      {
        "word": "て",
        "start": 132.063,
        "end": 132.123,
        "score": 0.667
      },
      {
        "word": "の",
        "start": 132.123,
        "end": 132.143,
        "score": 0.0
      },
      {
        "word": "こ",
        "start": 132.143,
        "end": 132.443,
        "score": 0.933
      },
      {
        "word": "と",
        "start": 132.443,
        "end": 134.204,
        "score": 0.989
      },
      {
        "word": "を",
        "start": 134.204,
        "end": 134.224,
        "score": 0.0
      },
      {
        "word": "や",
        "start": 134.224,
        "end": 134.244,
        "score": 0.0
      },
      {
        "word": "り",
        "start": 134.244,
        "end": 134.404,
        "score": 0.875
      },
      {
        "word": "た",
        "start": 134.404,
        "end": 135.224,
        "score": 0.976
      },
      {
        "word": "い",
        "start": 135.224,
        "end": 135.464,
        "score": 0.917
      },
      {
        "word": "と",
        "start": 135.464,
        "end": 135.785,
        "score": 0.938
      },
      {
        "word": "思",
        "start": 135.785,
        "end": 136.305,
        "score": 0.962
      },
      {
        "word": "い",
        "start": 136.305,
        "end": 136.325,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 136.325,
        "end": 136.345,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 136.345,
        "end": 136.365,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 136.365,
        "end": 136.385,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 136.385,
        "end": 136.445,
        "score": 0.667
      },
      {
        "word": "い",
        "start": 136.445,
        "end": 136.465,
        "score": 0.0
      },
      {
        "word": "。",
        "start": 136.465,
        "end": 136.485,
        "score": 0.0
      },
      {
        "word": "そ",
        "start": 136.485,
        "end": 136.505,
        "score": 0.0
      },
      {
        "word": "れ",
        "start": 136.505,
        "end": 136.545,
        "score": 0.5
      },
      {
        "word": "は",
        "start": 136.545,
        "end": 136.665,
        "score": 0.833
      },
      {
        "word": "、",
        "start": 136.665,
        "end": 136.685,
        "score": 1.0
      }
    ],
    "avg_logprob": -0.21270160857708223
  },
  {
    "start": 141.963,
    "end": 167.148,
    "text": "わかんないです今日は人生初の仲出しをしてみたいと思います仲出しって何かイメージあります?あんまりしちゃいけない",
    "words": [
      {
        "word": "わ",
        "start": 141.963,
        "end": 142.183,
        "score": 0.909
      },
      {
        "word": "か",
        "start": 142.183,
        "end": 142.203,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 142.203,
        "end": 142.443,
        "score": 0.917
      },
      {
        "word": "な",
        "start": 142.443,
        "end": 142.463,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 142.463,
        "end": 142.683,
        "score": 0.909
      },
      {
        "word": "で",
        "start": 142.683,
        "end": 144.864,
        "score": 0.991
      },
      {
        "word": "す",
        "start": 144.864,
        "end": 144.884,
        "score": 0.0
      },
      {
        "word": "今",
        "start": 144.884,
        "end": 144.904,
        "score": 0.0
      },
      {
        "word": "日",
        "start": 144.904,
        "end": 144.924,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 144.924,
        "end": 145.024,
        "score": 0.8
      },
      {
        "word": "人",
        "start": 145.024,
        "end": 148.724,
        "score": 0.995
      },
      {
        "word": "生",
        "start": 148.724,
        "end": 148.744,
        "score": 0.0
      },
      {
        "word": "初",
        "start": 148.744,
        "end": 165.047,
        "score": 0.999
      },
      {
        "word": "の",
        "start": 165.047,
        "end": 165.227,
        "score": 0.889
      },
      {
        "word": "仲",
        "start": 165.227,
        "end": 165.247,
        "score": 0.0
      },
      {
        "word": "出",
        "start": 165.247,
        "end": 165.267,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 165.267,
        "end": 165.287,
        "score": 0.0
      },
      {
        "word": "を",
        "start": 165.287,
        "end": 165.307,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 165.307,
        "end": 165.327,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 165.327,
        "end": 165.347,
        "score": 0.0
      },
      {
        "word": "み",
        "start": 165.347,
        "end": 165.527,
        "score": 0.889
      },
      {
        "word": "た",
        "start": 165.527,
        "end": 165.547,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 165.547,
        "end": 165.567,
        "score": 0.0
      },
      {
        "word": "と",
        "start": 165.567,
        "end": 165.587,
        "score": 0.0
      },
      {
        "word": "思",
        "start": 165.587,
        "end": 165.607,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 165.607,
        "end": 165.988,
        "score": 0.947
      },
      {
        "word": "ま",
        "start": 165.988,
        "end": 166.028,
        "score": 0.5
      },
      {
        "word": "す",
        "start": 166.028,
        "end": 166.048,
        "score": 0.0
      },
      {
        "word": "仲",
        "start": 166.048,
        "end": 166.068,
        "score": 0.0
      },
      {
        "word": "出",
        "start": 166.068,
        "end": 166.088,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 166.088,
        "end": 166.108,
        "score": 0.0
      },
      {
        "word": "っ",
        "start": 166.108,
        "end": 166.128,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 166.128,
        "end": 166.148,
        "score": 0.0
      },
      {
        "word": "何",
        "start": 166.148,
        "end": 166.168,
        "score": 0.0
      },
      {
        "word": "か",
        "start": 166.168,
        "end": 166.188,
        "score": 0.0
      },
      {
        "word": "イ",
        "start": 166.188,
        "end": 166.208,
        "score": 0.0
      },
      {
        "word": "メ",
        "start": 166.208,
        "end": 166.268,
        "score": 0.667
      },
      {
        "word": "ー",
        "start": 166.268,
        "end": 166.288,
        "score": 0.0
      },
      {
        "word": "ジ",
        "start": 166.288,
        "end": 166.308,
        "score": 0.0
      },
      {
        "word": "あ",
        "start": 166.308,
        "end": 166.328,
        "score": 0.0
      },
      {
        "word": "り",
        "start": 166.328,
        "end": 166.348,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 166.348,
        "end": 166.368,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 166.368,
        "end": 166.388,
        "score": 0.0
      },
      {
        "word": "?",
        "start": 166.388,
        "end": 166.408,
        "score": 0.0
      },
      {
        "word": "あ",
        "start": 166.408,
        "end": 166.428,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 166.428,
        "end": 166.448,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 166.448,
        "end": 166.528,
        "score": 0.75
      },
      {
        "word": "り",
        "start": 166.528,
        "end": 166.548,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 166.548,
        "end": 166.568,
        "score": 0.0
      },
      {
        "word": "ち",
        "start": 166.568,
        "end": 166.628,
        "score": 0.669
      },
      {
        "word": "ゃ",
        "start": 166.628,
        "end": 166.728,
        "score": 0.792
      },
      {
        "word": "い",
        "start": 166.728,
        "end": 166.868,
        "score": 0.862
      },
      {
        "word": "け",
        "start": 166.868,
        "end": 166.988,
        "score": 0.833
      },
      {
        "word": "な",
        "start": 166.988,
        "end": 167.128,
        "score": 0.857
      },
      {
        "word": "い",
        "start": 167.128,
        "end": 167.148,
        "score": 0.0
      }
    ],
    "avg_logprob": -0.1811266392469406
  },
  {
    "start": 172.544,
    "end": 198.59,
    "text": "そうそうしないじゃないですかはいただ今日はね初中出しコンドームをねしないおちんちんが入ってきてそして最後はね中に出したいと思いますはい大丈夫ですかなんか怖がってきてませんかうーんちょっと緊張してきました",
    "words": [
      {
        "word": "そ",
        "start": 172.544,
        "end": 172.644,
        "score": 0.8
      },
      {
        "word": "う",
        "start": 172.644,
        "end": 172.664,
        "score": 0.0
      },
      {
        "word": "そ",
        "start": 172.664,
        "end": 172.924,
        "score": 0.923
      },
      {
        "word": "う",
        "start": 172.924,
        "end": 173.444,
        "score": 0.962
      },
      {
        "word": "し",
        "start": 173.444,
        "end": 173.584,
        "score": 0.857
      },
      {
        "word": "な",
        "start": 173.584,
        "end": 177.425,
        "score": 0.995
      },
      {
        "word": "い",
        "start": 177.425,
        "end": 177.485,
        "score": 0.667
      },
      {
        "word": "じ",
        "start": 177.485,
        "end": 177.545,
        "score": 0.667
      },
      {
        "word": "ゃ",
        "start": 177.545,
        "end": 177.745,
        "score": 0.9
      },
      {
        "word": "な",
        "start": 177.745,
        "end": 180.106,
        "score": 0.992
      },
      {
        "word": "い",
        "start": 180.106,
        "end": 180.186,
        "score": 0.75
      },
      {
        "word": "で",
        "start": 180.186,
        "end": 180.646,
        "score": 0.957
      },
      {
        "word": "す",
        "start": 180.646,
        "end": 180.706,
        "score": 0.667
      },
      {
        "word": "か",
        "start": 180.706,
        "end": 181.246,
        "score": 0.963
      },
      {
        "word": "は",
        "start": 181.246,
        "end": 181.266,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 181.266,
        "end": 181.366,
        "score": 0.8
      },
      {
        "word": "た",
        "start": 181.366,
        "end": 181.686,
        "score": 0.938
      },
      {
        "word": "だ",
        "start": 181.686,
        "end": 182.586,
        "score": 0.978
      },
      {
        "word": "今",
        "start": 182.586,
        "end": 182.606,
        "score": 0.0
      },
      {
        "word": "日",
        "start": 182.606,
        "end": 182.626,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 182.626,
        "end": 182.646,
        "score": 0.0
      },
      {
        "word": "ね",
        "start": 182.646,
        "end": 182.666,
        "score": 0.0
      },
      {
        "word": "初",
        "start": 182.666,
        "end": 182.686,
        "score": 0.0
      },
      {
        "word": "中",
        "start": 182.686,
        "end": 182.746,
        "score": 0.667
      },
      {
        "word": "出",
        "start": 182.746,
        "end": 184.167,
        "score": 0.986
      },
      {
        "word": "し",
        "start": 184.167,
        "end": 190.488,
        "score": 0.997
      },
      {
        "word": "コ",
        "start": 190.488,
        "end": 190.568,
        "score": 0.75
      },
      {
        "word": "ン",
        "start": 190.568,
        "end": 190.588,
        "score": 0.0
      },
      {
        "word": "ド",
        "start": 190.588,
        "end": 190.608,
        "score": 0.0
      },
      {
        "word": "ー",
        "start": 190.608,
        "end": 190.648,
        "score": 0.5
      },
      {
        "word": "ム",
        "start": 190.648,
        "end": 190.948,
        "score": 0.933
      },
      {
        "word": "を",
        "start": 190.948,
        "end": 192.249,
        "score": 0.985
      },
      {
        "word": "ね",
        "start": 192.249,
        "end": 193.709,
        "score": 0.986
      },
      {
        "word": "し",
        "start": 193.709,
        "end": 194.129,
        "score": 0.952
      },
      {
        "word": "な",
        "start": 194.129,
        "end": 194.149,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 194.149,
        "end": 194.169,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 194.169,
        "end": 194.189,
        "score": 0.0
      },
      {
        "word": "ち",
        "start": 194.189,
        "end": 194.209,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 194.209,
        "end": 194.229,
        "score": 0.0
      },
      {
        "word": "ち",
        "start": 194.229,
        "end": 194.249,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 194.249,
        "end": 194.269,
        "score": 0.0
      },
      {
        "word": "が",
        "start": 194.269,
        "end": 194.289,
        "score": 0.0
      },
      {
        "word": "入",
        "start": 194.289,
        "end": 194.309,
        "score": 0.0
      },
      {
        "word": "っ",
        "start": 194.309,
        "end": 194.349,
        "score": 0.5
      },
      {
        "word": "て",
        "start": 194.349,
        "end": 194.489,
        "score": 0.857
      },
      {
        "word": "き",
        "start": 194.489,
        "end": 194.509,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 194.509,
        "end": 194.649,
        "score": 0.857
      },
      {
        "word": "そ",
        "start": 194.649,
        "end": 195.149,
        "score": 0.96
      },
      {
        "word": "し",
        "start": 195.149,
        "end": 195.529,
        "score": 0.947
      },
      {
        "word": "て",
        "start": 195.529,
        "end": 195.629,
        "score": 0.8
      },
      {
        "word": "最",
        "start": 195.629,
        "end": 195.789,
        "score": 0.875
      },
      {
        "word": "後",
        "start": 195.789,
        "end": 195.809,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 195.809,
        "end": 195.829,
        "score": 0.0
      },
      {
        "word": "ね",
        "start": 195.829,
        "end": 195.889,
        "score": 0.667
      },
      {
        "word": "中",
        "start": 195.889,
        "end": 196.429,
        "score": 0.963
      },
      {
        "word": "に",
        "start": 196.429,
        "end": 196.55,
        "score": 0.833
      },
      {
        "word": "出",
        "start": 196.55,
        "end": 197.07,
        "score": 0.962
      },
      {
        "word": "し",
        "start": 197.07,
        "end": 197.11,
        "score": 0.5
      },
      {
        "word": "た",
        "start": 197.11,
        "end": 197.13,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 197.13,
        "end": 197.17,
        "score": 0.5
      },
      {
        "word": "と",
        "start": 197.17,
        "end": 197.33,
        "score": 0.875
      },
      {
        "word": "思",
        "start": 197.33,
        "end": 197.43,
        "score": 0.8
      },
      {
        "word": "い",
        "start": 197.43,
        "end": 197.45,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 197.45,
        "end": 197.47,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 197.47,
        "end": 197.49,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 197.49,
        "end": 197.51,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 197.51,
        "end": 197.53,
        "score": 0.0
      },
      {
        "word": "大",
        "start": 197.53,
        "end": 197.55,
        "score": 0.0
      },
      {
        "word": "丈",
        "start": 197.55,
        "end": 197.57,
        "score": 0.0
      },
      {
        "word": "夫",
        "start": 197.57,
        "end": 197.59,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 197.59,
        "end": 197.61,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 197.61,
        "end": 197.65,
        "score": 0.5
      },
      {
        "word": "か",
        "start": 197.65,
        "end": 197.67,
        "score": 0.0
      },
      {
        "word": "な",
        "start": 197.67,
        "end": 197.69,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 197.69,
        "end": 197.71,
        "score": 0.0
      },
      {
        "word": "か",
        "start": 197.71,
        "end": 197.73,
        "score": 0.0
      },
      {
        "word": "怖",
        "start": 197.73,
        "end": 197.75,
        "score": 0.0
      },
      {
        "word": "が",
        "start": 197.75,
        "end": 197.77,
        "score": 0.0
      },
      {
        "word": "っ",
        "start": 197.77,
        "end": 197.79,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 197.79,
        "end": 197.81,
        "score": 0.0
      },
      {
        "word": "き",
        "start": 197.81,
        "end": 197.83,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 197.83,
        "end": 197.85,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 197.85,
        "end": 197.87,
        "score": 0.0
      },
      {
        "word": "せ",
        "start": 197.87,
        "end": 197.89,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 197.89,
        "end": 197.91,
        "score": 0.0
      },
      {
        "word": "か",
        "start": 197.91,
        "end": 197.93,
        "score": 0.0
      },
      {
        "word": "う",
        "start": 197.93,
        "end": 197.95,
        "score": 0.0
      },
      {
        "word": "ー",
        "start": 197.95,
        "end": 197.97,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 197.97,
        "end": 198.05,
        "score": 0.75
      },
      {
        "word": "ち",
        "start": 198.05,
        "end": 198.07,
        "score": 0.0
      },
      {
        "word": "ょ",
        "start": 198.07,
        "end": 198.17,
        "score": 0.8
      },
      {
        "word": "っ",
        "start": 198.17,
        "end": 198.19,
        "score": 0.0
      },
      {
        "word": "と",
        "start": 198.19,
        "end": 198.21,
        "score": 0.0
      },
      {
        "word": "緊",
        "start": 198.21,
        "end": 198.23,
        "score": 0.0
      },
      {
        "word": "張",
        "start": 198.23,
        "end": 198.25,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 198.25,
        "end": 198.39,
        "score": 0.857
      },
      {
        "word": "て",
        "start": 198.39,
        "end": 198.41,
        "score": 0.0
      },
      {
        "word": "き",
        "start": 198.41,
        "end": 198.43,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 198.43,
        "end": 198.45,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 198.45,
        "end": 198.57,
        "score": 0.833
      },
      {
        "word": "た",
        "start": 198.57,
        "end": 198.59,
        "score": 0.0
      }
    ],
    "avg_logprob": -0.10624999813735485
  },
  {
    "start": 216.065,
    "end": 245.165,
    "text": "大丈夫です今日はいろいろとね初めてのことをたくさんしていきたいと思いますのでよろしくお願いいたしますお願いします最後2本目の撮影の意気込みを撮影スタートしていきましょうはい楽しめたらいいなって思います",
    "words": [
      {
        "word": "大",
        "start": 216.065,
        "end": 216.105,
        "score": 0.5
      },
      {
        "word": "丈",
        "start": 216.105,
        "end": 216.125,
        "score": 0.0
      },
      {
        "word": "夫",
        "start": 216.125,
        "end": 216.145,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 216.145,
        "end": 216.265,
        "score": 0.833
      },
      {
        "word": "す",
        "start": 216.265,
        "end": 216.285,
        "score": 0.0
      },
      {
        "word": "今",
        "start": 216.285,
        "end": 216.305,
        "score": 0.0
      },
      {
        "word": "日",
        "start": 216.305,
        "end": 216.325,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 216.325,
        "end": 216.505,
        "score": 0.889
      },
      {
        "word": "い",
        "start": 216.505,
        "end": 216.525,
        "score": 0.0
      },
      {
        "word": "ろ",
        "start": 216.525,
        "end": 216.545,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 216.545,
        "end": 219.007,
        "score": 0.992
      },
      {
        "word": "ろ",
        "start": 219.007,
        "end": 220.128,
        "score": 0.982
      },
      {
        "word": "と",
        "start": 220.128,
        "end": 220.148,
        "score": 0.0
      },
      {
        "word": "ね",
        "start": 220.148,
        "end": 228.934,
        "score": 0.998
      },
      {
        "word": "初",
        "start": 228.934,
        "end": 228.954,
        "score": 0.0
      },
      {
        "word": "め",
        "start": 228.954,
        "end": 228.974,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 228.974,
        "end": 228.994,
        "score": 0.0
      },
      {
        "word": "の",
        "start": 228.994,
        "end": 229.014,
        "score": 0.0
      },
      {
        "word": "こ",
        "start": 229.014,
        "end": 229.094,
        "score": 0.75
      },
      {
        "word": "と",
        "start": 229.094,
        "end": 229.114,
        "score": 0.0
      },
      {
        "word": "を",
        "start": 229.114,
        "end": 229.134,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 229.134,
        "end": 229.154,
        "score": 0.0
      },
      {
        "word": "く",
        "start": 229.154,
        "end": 229.274,
        "score": 0.833
      },
      {
        "word": "さ",
        "start": 229.274,
        "end": 229.294,
        "score": 0.0
      },
      {
        "word": "ん",
        "start": 229.294,
        "end": 229.314,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 229.314,
        "end": 229.374,
        "score": 0.667
      },
      {
        "word": "て",
        "start": 229.374,
        "end": 229.394,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 229.394,
        "end": 229.494,
        "score": 0.8
      },
      {
        "word": "き",
        "start": 229.494,
        "end": 229.514,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 229.514,
        "end": 229.534,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 229.534,
        "end": 242.683,
        "score": 0.998
      },
      {
        "word": "と",
        "start": 242.683,
        "end": 242.863,
        "score": 0.889
      },
      {
        "word": "思",
        "start": 242.863,
        "end": 242.903,
        "score": 0.5
      },
      {
        "word": "い",
        "start": 242.903,
        "end": 242.923,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 242.923,
        "end": 242.943,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 242.943,
        "end": 242.963,
        "score": 0.0
      },
      {
        "word": "の",
        "start": 242.963,
        "end": 242.983,
        "score": 0.0
      },
      {
        "word": "で",
        "start": 242.983,
        "end": 243.003,
        "score": 0.0
      },
      {
        "word": "よ",
        "start": 243.003,
        "end": 243.023,
        "score": 0.0
      },
      {
        "word": "ろ",
        "start": 243.023,
        "end": 243.044,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 243.044,
        "end": 243.064,
        "score": 0.0
      },
      {
        "word": "く",
        "start": 243.064,
        "end": 243.084,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 243.084,
        "end": 243.104,
        "score": 0.0
      },
      {
        "word": "願",
        "start": 243.104,
        "end": 243.124,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 243.124,
        "end": 243.144,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 243.144,
        "end": 243.164,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 243.164,
        "end": 243.184,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 243.184,
        "end": 243.204,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 243.204,
        "end": 243.224,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 243.224,
        "end": 243.244,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 243.244,
        "end": 243.264,
        "score": 0.0
      },
      {
        "word": "願",
        "start": 243.264,
        "end": 243.284,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 243.284,
        "end": 243.304,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 243.304,
        "end": 243.324,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 243.324,
        "end": 243.344,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 243.344,
        "end": 243.364,
        "score": 0.0
      },
      {
        "word": "最",
        "start": 243.364,
        "end": 243.384,
        "score": 0.0
      },
      {
        "word": "後",
        "start": 243.384,
        "end": 243.404,
        "score": 0.0
      },
      {
        "word": "2",
        "start": 243.404,
        "end": 243.424,
        "score": 0.0
      },
      {
        "word": "本",
        "start": 243.424,
        "end": 243.444,
        "score": 0.0
      },
      {
        "word": "目",
        "start": 243.444,
        "end": 243.464,
        "score": 0.0
      },
      {
        "word": "の",
        "start": 243.464,
        "end": 243.484,
        "score": 0.0
      },
      {
        "word": "撮",
        "start": 243.484,
        "end": 243.504,
        "score": 0.0
      },
      {
        "word": "影",
        "start": 243.504,
        "end": 243.524,
        "score": 0.0
      },
      {
        "word": "の",
        "start": 243.524,
        "end": 243.544,
        "score": 0.0
      },
      {
        "word": "意",
        "start": 243.544,
        "end": 243.564,
        "score": 0.0
      },
      {
        "word": "気",
        "start": 243.564,
        "end": 243.584,
        "score": 0.0
      },
      {
        "word": "込",
        "start": 243.584,
        "end": 243.604,
        "score": 0.0
      },
      {
        "word": "み",
        "start": 243.604,
        "end": 243.624,
        "score": 0.0
      },
      {
        "word": "を",
        "start": 243.624,
        "end": 243.644,
        "score": 0.0
      },
      {
        "word": "撮",
        "start": 243.644,
        "end": 243.664,
        "score": 0.0
      },
      {
        "word": "影",
        "start": 243.664,
        "end": 243.684,
        "score": 0.0
      },
      {
        "word": "ス",
        "start": 243.684,
        "end": 243.704,
        "score": 0.0
      },
      {
        "word": "タ",
        "start": 243.704,
        "end": 243.724,
        "score": 0.0
      },
      {
        "word": "ー",
        "start": 243.724,
        "end": 243.744,
        "score": 0.0
      },
      {
        "word": "ト",
        "start": 243.744,
        "end": 243.764,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 243.764,
        "end": 243.784,
        "score": 0.0
      },
      {
        "word": "て",
        "start": 243.784,
        "end": 243.804,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 243.804,
        "end": 243.824,
        "score": 0.0
      },
      {
        "word": "き",
        "start": 243.824,
        "end": 243.844,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 243.844,
        "end": 243.864,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 243.864,
        "end": 243.884,
        "score": 0.0
      },
      {
        "word": "ょ",
        "start": 243.884,
        "end": 243.964,
        "score": 0.75
      },
      {
        "word": "う",
        "start": 243.964,
        "end": 243.984,
        "score": 0.0
      },
      {
        "word": "は",
        "start": 243.984,
        "end": 244.124,
        "score": 0.857
      },
      {
        "word": "い",
        "start": 244.124,
        "end": 244.144,
        "score": 0.0
      },
      {
        "word": "楽",
        "start": 244.144,
        "end": 244.164,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 244.164,
        "end": 244.244,
        "score": 0.75
      },
      {
        "word": "め",
        "start": 244.244,
        "end": 244.264,
        "score": 0.0
      },
      {
        "word": "た",
        "start": 244.264,
        "end": 244.284,
        "score": 0.0
      },
      {
        "word": "ら",
        "start": 244.284,
        "end": 244.304,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 244.304,
        "end": 244.324,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 244.324,
        "end": 244.344,
        "score": 0.0
      },
      {
        "word": "な",
        "start": 244.344,
        "end": 244.525,
        "score": 0.892
      },
      {
        "word": "っ",
        "start": 244.525,
        "end": 244.605,
        "score": 0.75
      },
      {
        "word": "て",
        "start": 244.605,
        "end": 244.725,
        "score": 0.833
      },
      {
        "word": "思",
        "start": 244.725,
        "end": 244.965,
        "score": 0.917
      },
      {
        "word": "い",
        "start": 244.965,
        "end": 244.985,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 244.985,
        "end": 245.145,
        "score": 0.875
      },
      {
        "word": "す",
        "start": 245.145,
        "end": 245.165,
        "score": 0.0
      }
    ],
    "avg_logprob": -0.04426081730769231
  },
  {
    "start": 249.857,
    "end": 250.38,
    "text": "よろしくお願いしますお願いします",
    "words": [
      {
        "word": "よ",
        "start": 249.857,
        "end": 249.877,
        "score": 0.0
      },
      {
        "word": "ろ",
        "start": 249.877,
        "end": 249.897,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 249.897,
        "end": 249.978,
        "score": 0.75
      },
      {
        "word": "く",
        "start": 249.978,
        "end": 249.998,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 249.998,
        "end": 250.018,
        "score": 0.002
      },
      {
        "word": "願",
        "start": 250.018,
        "end": 250.038,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 250.038,
        "end": 250.058,
        "score": 0.0
      },
      {
        "word": "し",
        "start": 250.058,
        "end": 250.078,
        "score": 0.0
      },
      {
        "word": "ま",
        "start": 250.078,
        "end": 250.099,
        "score": 0.0
      },
      {
        "word": "す",
        "start": 250.099,
        "end": 250.119,
        "score": 0.0
      },
      {
        "word": "お",
        "start": 250.119,
        "end": 250.139,
        "score": 0.0
      },
      {
        "word": "願",
        "start": 250.139,
        "end": 250.159,
        "score": 0.0
      },
      {
        "word": "い",
        "start": 250.159,
        "end": 250.179,
        "score": 0.003
      },
      {
        "word": "し",
        "start": 250.179,
        "end": 250.259,
        "score": 0.771
      },
      {
        "word": "ま",
        "start": 250.259,
        "end": 250.36,
        "score": 0.802
      },
      {
        "word": "す",
        "start": 250.36,
        "end": 250.38,
        "score": 0.983
      }
    ],
    "avg_logprob": -0.3095703050494194
  }
]
```
