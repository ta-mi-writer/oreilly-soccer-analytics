import os
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 1. 設定エリア（生成したいテキストやモデルの指定）
# ----------------------------------------------------------------------
# 生成させたいセリフ（絵文字を入れると感情やニュアンスが乗ります😊）
TEXT_TO_SPEECH = "こんにちは！これは、背景からBGMを除去した音声をリファレンスにして生成された、AIの合成音声です😊"

# 使用する Irodori-TTS モデル（最新の v3 ベースモデル）
HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v3"

# 入力・出力ディレクトリ
INPUT_DIR = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORKING_DIR / "tts_output"
OUTPUT_WAV_PATH = OUTPUT_DIR / "generated_speech.wav"

# 推論用リポジトリ
REPO_DIR = WORKING_DIR / "Irodori-TTS"


# ----------------------------------------------------------------------
# 2. 処理関数
# ----------------------------------------------------------------------
def find_reference_audio() -> Path:
    """前工程（音源分離）から渡されたボーカル WAV ファイルを検索する"""
    print("=== [1/4] リファレンス音声（ボーカル）の探索 ===")

    # 分離されたボーカルファイル (*vocals*.wav や .wav ファイル) を探索
    wav_files = list(INPUT_DIR.rglob("*.wav"))

    # 'vocal' または 'instrumental' 以外（声トラック）を優先選択
    vocal_files = [
        f for f in wav_files if "vocal" in f.name.lower() or "speech" in f.name.lower()
    ]

    target_path = (
        vocal_files[0] if vocal_files else (wav_files[0] if wav_files else None)
    )

    if target_path and target_path.exists():
        print(f"[成功] リファレンス音声を検出しました: {target_path}")
        return target_path

    print("[エラー] リファレンス音声が見つかりませんでした。", file=sys.stderr)
    print("--- 現在の /kaggle/input/ 一覧 ---")
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            print(os.path.join(root, f))
    sys.exit(1)


def setup_environment():
    """Irodori-TTS リポジトリの取得と依存ライブラリのセットアップ"""
    print("\n=== [2/4] Irodori-TTS 推論環境のセットアップ ===")

    if not REPO_DIR.exists():
        print("Irodori-TTS リポジトリをクローン中...")
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/Aratako/Irodori-TTS.git",
                str(REPO_DIR),
            ],
            check=True,
        )

    # パッケージツール uv のインストールと環境同期
    print("必要なライブラリのセットアップ中...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)

    # リポジトリ内で uv sync を実行して環境構築
    subprocess.run(["uv", "sync"], cwd=str(REPO_DIR), check=True)
    print("[成功] 環境のセットアップが完了しました。")


def generate_tts(ref_wav_path: Path):
    """Irodori-TTS を呼び出して音声合成を実行"""
    print("\n=== [3/4] Irodori-TTS による音声合成（TTS）実行 ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"テキスト: 「{TEXT_TO_SPEECH}」")
    print(f"モデル: {HF_CHECKPOINT}")

    # Irodori-TTS の CLI 推論コマンド (infer.py)
    cmd = [
        "uv",
        "run",
        "python",
        "infer.py",
        "--hf-checkpoint",
        HF_CHECKPOINT,
        "--text",
        TEXT_TO_SPEECH,
        "--ref-wav",
        str(ref_wav_path),
        "--output-wav",
        str(OUTPUT_WAV_PATH),
    ]

    print("音声生成を実行中（初回はモデルダウンロードが行われます）...")
    res = subprocess.run(cmd, cwd=str(REPO_DIR), text=True)

    if res.returncode == 0 and OUTPUT_WAV_PATH.exists():
        print(f"[成功] 音声生成が完了しました！ 保存先: {OUTPUT_WAV_PATH}")
    else:
        print("[エラー] 音声生成に失敗しました。", file=sys.stderr)
        sys.exit(1)


def cleanup():
    """成果物 (OUTPUT_DIR) 以外の不要ファイルを確実に削除"""
    print("\n=== [4/4] クリーンアップ（不要ファイルの削除） ===")
    targets = [REPO_DIR]

    for item in targets:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
            print(f"削除完了: {item.name}")

    print(f"\n[完了] 成果物は {OUTPUT_DIR} に保存されています。")


# ----------------------------------------------------------------------
# 3. メインフロー
# ----------------------------------------------------------------------
def main():
    try:
        ref_wav_path = find_reference_audio()
        setup_environment()
        generate_tts(ref_wav_path)
    finally:
        # 成功・失敗に関わらず、モデルやコード等の不要データを確実に削除
        cleanup()


if __name__ == "__main__":
    main()
