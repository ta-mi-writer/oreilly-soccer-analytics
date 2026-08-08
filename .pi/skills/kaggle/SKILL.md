---
name: kaggle
description: Instructions for using the kaggle CLI command in this project. Use when running kaggle commands for dataset downloads, kernel submission, or Kaggle API operations. Covers authentication, file organization in kaggle_jobs/, and GPU kernel configuration.
---

# Kaggle Command Usage

Use this skill when running any `kaggle` CLI command in this project.

## Authentication

Always prefix `kaggle` commands with the API token from `.kaggle/access_token`:

```bash
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) uv run kaggle <subcommand>
```

Never use the `kaggle` command without this prefix — authentication will fail.

## Working Files

All Kaggle-related work files are stored in the `kaggle_jobs/` directory:

```
kaggle_jobs/
├── <job-name>/
│   ├── dataset-metadata.json     # (for datasets)
│   └── kernel-metadata.json      # (for kernels)
```

When creating a new Kaggle job, create its subdirectory under `kaggle_jobs/`.

## GPU Kernels

When configuring a kernel that uses a GPU, set `machine_shape` to `NvidiaTeslaT4` in the `kernel-metadata.json`:

```json
{
  "machine_shape": "NvidiaTeslaT4"
}
```

This ensures the kernel runs on an NVIDIA T4 GPU instance.

## Kernel Output Downloads

When downloading kernel outputs with `kaggle kernels output`, always save to the job's `output/` directory using the `-p` flag:

```bash
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) uv run kaggle kernels output <owner>/<kernel-slug> -p ./kaggle_jobs/<job-name>/output
```

**Pattern:** The `<kernel-slug>` and `<job-name>` are typically the same (e.g., `trimming`). This keeps outputs organized alongside the job's metadata and logs.

**Example:**

```bash
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) uv run kaggle kernels output masashiki/trimming -p ./kaggle_jobs/trimming/output
```

This downloads output files (videos, CSVs, models, etc.) and the kernel log directly into `kaggle_jobs/trimming/output/` instead of the project root.
