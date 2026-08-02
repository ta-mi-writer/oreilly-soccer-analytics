import os
import subprocess
import sys
from pathlib import Path

# ==========================================
# 1. 設定エリア（ご自身の環境に合わせて書き換えてください）
# ==========================================

# 元動画のパス（/kaggle/input/ 配下の動画ファイルを指定）
# ※データセット名やファイル名はご自身のKaggleデータセットに合わせて変更してください
INPUT_VIDEO_PATH = "/kaggle/input/cawd-255-english-subtitle/video.mp4"
# 出力先ディレクトリ（Kaggleで書き込み可能な作業領域）
OUTPUT_DIR = Path("/kaggle/working/trimmed_videos")

# 切り出したい時間範囲のリスト
# start / end は "hh:mm:ss" または "秒数"（文字列でも数値でも可）で指定します
CLIPS_TO_TRIM = [
    {
        "start": "01:56:57",
        "end": "01:57:13",
        "output_name": "ref-raw.mp4",
    },
    # {
    #     "start": "00:50:30",
    #     "end": "01:00:15",
    #     "output_name": "clip2_50m30s_60m15s.mp4",
    # },
    # 必要に応じて追加してください
]


# ==========================================
# 2. トリミング処理のロジック
# ==========================================


def check_input_file(video_path: str) -> None:
    """入力ファイルが存在するかチェックする"""
    if not os.path.exists(video_path):
        print(f"[Error] 入力動画が見つかりません: {video_path}", file=sys.stderr)
        print("Kaggleの /kaggle/input/ 配下を確認してください。")

        # 参考用に /kaggle/input 配下のファイル構造を出力
        print("\n--- 現在利用可能な入力ファイル一覧 ---")
        for root, _, files in os.walk("/kaggle/input"):
            for file in files:
                print(os.path.join(root, file))
        sys.exit(1)


def trim_video(input_path: str, start: str, end: str, output_path: Path) -> bool:
    """FFmpeg を使用してストリームコピー（再エンコードなし）で高速トリミングを実行する

    -ss を -i より前に配置することで、指定時間より手前のキーフレームから切り出し（＝長くなる方向）を行います。
    """
    cmd = [
        "ffmpeg",
        "-y",  # 同名ファイルが存在する場合は上書き
        "-ss",
        str(start),  # 開始時間（-iより前に置くことで直前のキーフレームを探す）
        "-to",
        str(end),  # 終了時間
        "-i",
        input_path,  # 入力ファイル
        "-c",
        "copy",  # ストリームコピー（再エンコードなし・無劣化・超高速）
        str(output_path),
    ]

    print(f"\n[処理開始] {start} ～ {end} ➔ {output_path.name} を切り出しています...")

    try:
        # コマンドの実行
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0:
            print(f"[成功] 保存完了: {output_path}")
            return True
        else:
            print("[失敗] FFmpeg エラー:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False

    except Exception as e:
        print(f"[例外発生] 実行中にエラーが起きました: {e}", file=sys.stderr)
        return False


def main():
    print("=== 動画トリミング処理を開始します ===")

    # 1. 入力ファイルの存在チェック
    check_input_file(INPUT_VIDEO_PATH)

    # 2. 出力用ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. トリミングの順次実行
    success_count = 0
    for i, clip in enumerate(CLIPS_TO_TRIM, 1):
        print(f"\n--- Clip {i}/{len(CLIPS_TO_TRIM)} ---")
        start_time = clip["start"]
        end_time = clip["end"]
        output_file = OUTPUT_DIR / clip["output_name"]

        is_success = trim_video(INPUT_VIDEO_PATH, start_time, end_time, output_file)
        if is_success:
            success_count += 1

    print("\n==========================================")
    print(f"全処理が完了しました: {success_count}/{len(CLIPS_TO_TRIM)} 件成功")
    print("==========================================")


if __name__ == "__main__":
    main()
