import shutil
import subprocess
import sys
from pathlib import Path

# ======================================================================
# 1. 基本設定エリア（セリフ・モデル指定）
# ======================================================================
# 喋らせたいテキスト（絵文字で感情や雰囲気を表現可能）
TEXT_TO_SPEECH = "こんにちは！これは、BS-RoformerでBGMを除去した声をリファレンスにして、最新のイロドリTTSv4スモールで生成した音声です😊"

# 使用するモデル (最新の v4-Small)
HF_CHECKPOINT = "Aratako/Irodori-TTS-v4-Small"

# スタイル指示文（Caption / Style Prompt）※空欄でもOK
CAPTION_PROMPT = ""  # 例: "落ち着いた声で、明るく楽しそうに話す"


# ======================================================================
# 2. Sampling（サンプリング設定エリア）
# ※ デモ画面と同じデフォルト値です。自由に変更調整可能です！
# ======================================================================
SAMPLING_PARAMS = {
    # 音声の組み立てステップ数 (多いほど高精度。デフォルト: 40)
    "num_steps": 40,
    # 話すスピードの調整 (1.0=標準, 0.8=速い, 1.2=ゆっくり)
    "duration_scale": 1.0,
    # 【最重要】リファレンス音声への「声質の似せ具合」 (デフォルト: 5.0)
    "cfg_scale_speaker": 5.0,
    # テキスト通りにハッキリ喋らせる強さ (デフォルト: 3.0)
    "cfg_scale_text": 3.0,
    # スタイル指示（Caption）を守る強さ (デフォルト: 4.0)
    "cfg_scale_caption": 4.0,
    # 乱数シード (None=毎回ランダム, 数字を入れると再現可能。例: 42)
    "seed": None,
}


# ======================================================================
# 3. パス設定
# ======================================================================
INPUT_DIR = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORKING_DIR / "tts_output"
OUTPUT_WAV_PATH = OUTPUT_DIR / "generated_speech.wav"

REPO_DIR = WORKING_DIR / "Irodori-TTS"


# ======================================================================
# 4. 処理用関数
# ======================================================================
def find_reference_audio() -> Path:
    """前工程（音源分離）で作成したボーカル WAV ファイルを探す"""
    print("=== [1/4] リファレンス音声の自動探索 ===")

    wav_files = list(INPUT_DIR.rglob("*.wav"))

    # 'vocal' または 'speech' が含まれるファイルを優先
    vocal_files = [
        f for f in wav_files if "vocal" in f.name.lower() or "speech" in f.name.lower()
    ]

    target_path = (
        vocal_files[0] if vocal_files else (wav_files[0] if wav_files else None)
    )

    if target_path and target_path.exists():
        print(f"[成功] リファレンス音声を検出: {target_path}")
        return target_path

    print(
        "[エラー] 前処理の音声ファイルが見つかりませんでした。",
        file=sys.stderr,
    )
    sys.exit(1)


def setup_environment():
    """Irodori-TTS 環境の構築"""
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

    # パッケージ管理ツール uv の準備
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)

    # 依存ライブラリのセットアップ (CUDA対応)
    print("依存ライブラリのセットアップ中...")
    subprocess.run(["uv", "sync", "--extra", "cu128"], cwd=str(REPO_DIR), check=True)
    print("[成功] 環境セットアップ完了")


def run_v4_inference(ref_wav_path: Path):
    """v4-Small モデルによる音声合成を実行"""
    print("\n=== [3/4] Irodori-TTS-v4-Small による音声合成 ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"テキスト: 「{TEXT_TO_SPEECH}」")
    print(f"モデル: {HF_CHECKPOINT}")
    print(f"Samplingパラメータ: {SAMPLING_PARAMS}")

    # infer.py の呼び出しコマンド構築
    cmd = [
        "uv",
        "run",
        "--no-sync",
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
        "--num-steps",
        str(SAMPLING_PARAMS["num_steps"]),
        "--duration-scale",
        str(SAMPLING_PARAMS["duration_scale"]),
        "--cfg-scale-speaker",
        str(SAMPLING_PARAMS["cfg_scale_speaker"]),
        "--cfg-scale-text",
        str(SAMPLING_PARAMS["cfg_scale_text"]),
        "--cfg-scale-caption",
        str(SAMPLING_PARAMS["cfg_scale_caption"]),
    ]

    # Caption プロンプトがあれば追加
    if CAPTION_PROMPT:
        cmd.extend(["--caption", CAPTION_PROMPT])

    # シード値が指定されていれば追加
    if SAMPLING_PARAMS["seed"] is not None:
        cmd.extend(["--seed", str(SAMPLING_PARAMS["seed"])])

    print("\n音声生成中（初回は v4-Small モデルの自動ダウンロードが行われます）...")
    res = subprocess.run(cmd, cwd=str(REPO_DIR), text=True)

    if res.returncode == 0 and OUTPUT_WAV_PATH.exists():
        print(f"\n[成功] 音声生成が完了しました！ 出力先: {OUTPUT_WAV_PATH}")
    else:
        print("\n[エラー] 音声生成に失敗しました。", file=sys.stderr)
        sys.exit(1)


def cleanup():
    """成果物(OUTPUT_DIR)以外の不要ファイルを完全削除"""
    print("\n=== [4/4] 不要ファイルの完全削除（クリーンアップ） ===")
    targets = [REPO_DIR]

    for item in targets:
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
            print(f"削除完了: {item.name}")

    print(f"\n[完了] 成果物のみ {OUTPUT_DIR} に保存されています。")


# ======================================================================
# 5. メイン実行フロー
# ======================================================================
def main():
    try:
        ref_wav_path = find_reference_audio()
        setup_environment()
        run_v4_inference(ref_wav_path)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
