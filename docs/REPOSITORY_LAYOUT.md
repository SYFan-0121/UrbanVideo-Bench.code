# Repository layout

This repository contains the reproducible benchmark code and the Windows-server deployment helpers. Dataset files, model weights, API keys, caches, and experiment outputs stay outside Git.

```text
UrbanVideo-Bench.code/
├── README.md                         # Project overview and quick start
├── run.py                            # Main benchmark inference loop
├── eval.py                           # Accuracy and Excel report generation
├── prompts.py                        # Category-specific prompts
├── confidence.py                     # Answer extraction and STRIVE scoring
├── agentic_retrieval.py              # Optional CLIP-based frame retrieval
├── video_locks.py                    # Safe concurrent video access
├── vlm_api.py                        # OpenAI-compatible VLM provider adapter
├── server.py                         # Local dashboard backend
├── templates/dashboard.html          # Dashboard frontend
├── question_generation/              # Gemini-based MCQ generation scripts
│   ├── MCQ_generation_basic.py
│   ├── MCQ_generation_goal.py
│   └── MCQ_generation_route.py
├── deployment/                       # Windows server setup and data utilities
│   ├── WINDOWS_NATIVE_DEPLOYMENT.md
│   ├── SERVER_FULL_RUN.md
│   ├── requirements-windows.txt
│   ├── requirements-benchmark-windows.txt
│   ├── windows_vlm_server.py
│   ├── smoke_test.py
│   ├── test_windows_vlm_server.py
│   ├── download_urbanvideo_dataset.py
│   ├── verify_urbanvideo_dataset.py
│   ├── upload_local_video_supplement.ps1
│   └── merge_supplemental_videos.py
└── docs/                             # Reproducibility and repository notes
    └── REPOSITORY_LAYOUT.md
```

## Keep outside Git

The following directories and files are intentionally ignored:

- `dataset/`: `MCQ.parquet` and MP4 videos.
- `result/`: predictions, SQLite dashboard data, and Excel reports.
- `cache/`, `hf-cache/`, and model caches.
- Python environments and generated logs.
- API keys, tokens, model weights, archives, and personal research material.

Download or copy the dataset on the server after cloning the code. Use a new `RUN_ID` for every baseline or ablation so that results remain separate.

## Recommended GitHub workflow

Keep the official repository as the read-only upstream remote. Add your own GitHub repository under a second remote such as `research`, then push a research branch:

```powershell
git switch -c codex/research-server-full-run
git remote add research git@github.com:<your-user>/<your-repository>.git
git add .gitignore README.md run.py eval.py prompts.py confidence.py `
  agentic_retrieval.py video_locks.py vlm_api.py server.py templates `
  question_generation deployment docs
git status --short
git diff --cached --stat
git commit -m "Prepare native Windows server experiments"
git push -u research codex/research-server-full-run
```

Before committing, confirm that the staged file list contains no `.mp4`, `.parquet`, API key, model weight, `dataset/`, `result/`, or personal backup directory.
