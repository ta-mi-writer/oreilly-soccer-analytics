# Kaggle Agent Operations

本プロジェクトは、AIエージェントがKaggle環境を効率的に操作・管理するための仕組みを提供します。`uv`によるPython依存関係の管理と、Kaggle CLIを組み合わせることで、Notebook（Kernel）の初期化から実行、出力取得までを再現性のある形で行います。

---

## 1. 前提環境

- [uv](https://docs.astral.sh/uv/) がインストールされていること
- Kaggle API トークンが `.kaggle/access_token` に保存されていること
- `pyproject.toml` / `uv.lock` により、`kaggle` CLI が依存関係として管理されていること

---

## 2. Kaggle 認証・実行の基本ルール

Kaggle CLI は認証が必須です。すべての `kaggle` コマンドは、以下の接頭辞を必ず付けて実行してください。

```bash
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) uv run kaggle ...
```

- `KAGGLE_API_TOKEN` 環境変数に API トークンの中身を設定します。
- `uv run kaggle` で、仮想環境内の Kaggle CLI を実行します。
- **生の `kaggle` コマンドや、`uv` を使わない実行は避けてください。**

---

## 3. 主要な Kaggle CLI コマンド集

以下、すべての例で認証接頭辞 `KAGGLE_API_TOKEN=$(cat .kaggle/access_token)` を省略しています。実際には必ず付与してください。

### 3.1 カーネルの初期化

新しい Kaggle Notebook（Kernel）用の雛形を `./kaggle_jobs/<job_name>` 配下に作成します。

```bash
uv run kaggle kernels init -p ./kaggle_jobs/<job_name>
```

実行後、指定ディレクトリに `kernel-metadata.json` と `.ipynb` 雛形が生成されます。

### 3.2 既存カーネルのプル

Kaggle 上の既存カーネルをローカルに取得します。

```bash
uv run kaggle kernels pull <owner>/<kernel-slug> -m -p ./kaggle_jobs/<job_name>
```

- `-m` メタデータも一緒に取得します（`kernel-metadata.json` が含まれます）。
- ローカルの作業ディレクトリ名は、Kaggle 上のスラッグに合わせることを推奨します。

### 3.3 カーネルのプッシュ

ローカルの編集内容を Kaggle 上にアップロード・実行します。

```bash
uv run kaggle kernels push -p ./kaggle_jobs/<job_name>
```

プッシュ後、Kaggle 側でビルド・実行が始まります。

### 3.4 ステータス確認

プッシュしたカーネルの実行状況を確認します。

```bash
uv run kaggle kernels status <owner>/<kernel-slug>
```

実行完了までの進捗や、エラー発生時の確認に使用します。

### 3.5 出力ダウンロード

カーネル実行後に生成されたファイル（モデル、CSV、ログ等）をダウンロードします。

```bash
uv run kaggle kernels output <owner>/<kernel-slug> -p ./kaggle_jobs/<job_name>/output
```

**重要:** 出力先は必ず `./kaggle_jobs/<job_name>/output` のように、作業ディレクトリ配下に指定してください。指定しない場合、ファイルがプロジェクトルートに散らばります。

---

## 4. ディレクトリ規約

### 4.1 `./kaggle_jobs/` 配下の構造

すべての Kaggle 作業は `./kaggle_jobs/` 配下に閉じ込めます。

```text
./kaggle_jobs/
├── <job_name_1>/
│   ├── kernel-metadata.json
│   ├── <notebook>.ipynb
│   └── output/              # カーネル出力のダウンロード先
├── <job_name_2>/
│   ├── kernel-metadata.json
│   ├── <notebook>.ipynb
│   └── output/
└── ...
```

- 1つの Kaggle Kernel に対して、1つのサブディレクトリを割り当てます。
- ディレクトリ名は、Kaggle 上のカーネルスラッグと一致させることを推奨します。

### 4.2 GPU 設定

GPU を使用する場合は、`kernel-metadata.json` に以下の設定を追加してください。

```json
{
  "machine_shape": "NvidiaTeslaT4"
}
```

- 設定を省略すると CPU 環境で実行されます。
- 使用可能な GPU タイプは Kaggle の仕様に依存します。`NvidiaTeslaT4` は一般的な選択肢です。

---

## 5. まとめ

| 操作           | コマンド（認証接頭辞省略）                                                       |
| -------------- | -------------------------------------------------------------------------------- |
| カーネル初期化 | `uv run kaggle kernels init -p ./kaggle_jobs/<job_name>`                         |
| カーネル取得   | `uv run kaggle kernels pull <owner>/<slug> -m -p ./kaggle_jobs/<job_name>`       |
| カーネル実行   | `uv run kaggle kernels push -p ./kaggle_jobs/<job_name>`                         |
| ステータス確認 | `uv run kaggle kernels status <owner>/<slug>`                                    |
| 出力取得       | `uv run kaggle kernels output <owner>/<slug> -p ./kaggle_jobs/<job_name>/output` |

これらの規約に従うことで、AIエージェントや複数人の開発者が、Kaggle 環境への操作で混乱を起こすことを防ぎます。
