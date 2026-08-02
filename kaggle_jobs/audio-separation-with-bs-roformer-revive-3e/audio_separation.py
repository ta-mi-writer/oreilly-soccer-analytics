import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 1. パスと設定
# ----------------------------------------------------------------------
INPUT_DIR = Path("/kaggle/input")
TARGET_VIDEO_NAME = "ref-raw.mp4"

WORKING_DIR = Path("/kaggle/working")
TEMP_INPUT_DIR = WORKING_DIR / "temp_input"
EXTRACTED_AUDIO_PATH = TEMP_INPUT_DIR / "extracted_speech.wav"

# 成果物の保存先
OUTPUT_DIR = WORKING_DIR / "separated_output"

# 推論用リポジトリおよびモデル
REPO_DIR = WORKING_DIR / "BS-Roformer-Repo"
MODEL_URL = "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt"
CONFIG_URL = "https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml"

CKPT_PATH = WORKING_DIR / "bs_roformer_revive3e.ckpt"
CONFIG_PATH = WORKING_DIR / "config.yaml"


# ----------------------------------------------------------------------
# 2. 処理関数
# ----------------------------------------------------------------------
def setup_environment():
    print("=== [1/4] 環境のセットアップ ===")

    # リポジトリの取得
    if not REPO_DIR.exists():
        print("推論リポジトリをクローン中...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ZFTurbo/Music-Source-Separation-Training.git",
                str(REPO_DIR),
            ],
            check=True,
        )

    # BS-Roformer の推論に必要なパッケージを指定
    required_packages = [
        "ml_collections",
        "librosa",
        "soundfile",
        "ptflops",
        "timm",
        "ftfy",
        "rotary-embedding-torch",
        "einops",
    ]
    print("必要なパッケージをインストール中...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + required_packages,
        check=True,
    )

    # モデルおよび設定ファイルの取得
    if not CKPT_PATH.exists():
        print(f"モデルをダウンロード中: {CKPT_PATH.name}")
        subprocess.run(["wget", "-q", MODEL_URL, "-O", str(CKPT_PATH)], check=True)
    if not CONFIG_PATH.exists():
        print(f"設定ファイルをダウンロード中: {CONFIG_PATH.name}")
        subprocess.run(["wget", "-q", CONFIG_URL, "-O", str(CONFIG_PATH)], check=True)

    print("[成功] セットアップが完了しました。")


def extract_audio():
    print("\n=== [2/4] 入力動画の探索と音声抽出 ===")
    found = list(INPUT_DIR.rglob(TARGET_VIDEO_NAME))
    if not found:
        raise FileNotFoundError(f"[エラー] {TARGET_VIDEO_NAME} が見つかりません。")

    print(f"動画を発見しました: {found[0]}")
    TEMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(found[0]),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(EXTRACTED_AUDIO_PATH),
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    print(f"[成功] 音声抽出完了: {EXTRACTED_AUDIO_PATH}")


def run_inference():
    print("\n=== [3/4] BS-Roformer Revive 3e による音源分離 ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_DIR / "inference.py"),
        "--model_type",
        "bs_roformer",
        "--config_path",
        str(CONFIG_PATH),
        "--start_check_point",
        str(CKPT_PATH),
        "--input_folder",
        str(TEMP_INPUT_DIR),
        "--store_dir",
        str(OUTPUT_DIR),
        "--extract_instrumental",
    ]

    print("推論を実行中...")
    subprocess.run(cmd, check=True)
    print("[成功] 音源分離が正常に完了しました！")


def cleanup():
    """途中でエラーが起きても確実に不要ファイル群を削除する"""
    print("\n=== [4/4] 不要ファイルの完全削除（クリーンアップ） ===")
    targets = [CKPT_PATH, CONFIG_PATH, TEMP_INPUT_DIR, REPO_DIR]

    for item in targets:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
            print(f"削除: {item.name}")

    print("\n[完了] 成果物以外のファイル削除が完了しました。")


# ----------------------------------------------------------------------
# 3. メインフロー
# ----------------------------------------------------------------------
def main():
    try:
        setup_environment()
        extract_audio()
        run_inference()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
