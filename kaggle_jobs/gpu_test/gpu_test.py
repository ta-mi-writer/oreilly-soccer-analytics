# %% [code]
import sys

import torch

def check_gpu_environment():
    print("==========================================")
    print("      Kaggle GPU & CUDA 環境チェック")
    print("==========================================")
    print(f"Python バージョン: {sys.version.split()[0]}")
    print(f"PyTorch バージョン: {torch.__version__}")

    # 1. GPUが認識されているか確認
    cuda_available = torch.cuda.is_available()
    print(
        f"\nCUDA (GPU) の認識状態: {'✅ 利用可能' if cuda_available else '❌ 不可 (CPUモード)'}"
    )

    if not cuda_available:
        print("\n[判定] GPUが割り当てられていません。設定を確認してください。")
        return

    # 2. 割り当てられたGPUの数と機種名の取得
    device_count = torch.cuda.device_count()
    print(f"認識された GPU の数: {device_count}")

    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f"\n--- GPU [{i}] 情報 ---")
        print(f"  機種名 (Device Name): {gpu_name}")
        print(f"  CUDA Capability   : {capability[0]}.{capability[1]}")

    # 3. 実際にテンソル計算が通るか動作テスト (sm_60 エラー等の判定)
    print("\n------------------------------------------")
    print("  CUDA 演算動作テストを実行中...")
    print("------------------------------------------")
    try:
        # GPU上で簡単なテンソル計算を実行
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2.0
        print(f"  GPUでの計算成功: {y.cpu().numpy()}")
        print("\n🎉 【成功】このGPU機種は、現在のPyTorch環境で問題なく動作します！")
    except Exception as e:
        print("\n❌ 【失敗】CUDAエラーが発生しました:")
        print(f"  エラー詳細: {e}")
        print(
            "\n[判定] このGPU（P100等）はPyTorchと互換性がありません。Web画面で T4 x2 に切り替えるか CPU モードを使用してください。"
        )

        print(f"  エラー詳細: {e}")
        print("\n[判定] このGPU（P100等）はPyTorchと互換性がありません。Web画面で T4 x2 に切り替えるか CPU モードを使用してください。")

if __name__ == "__main__":
    check_gpu_environment()
