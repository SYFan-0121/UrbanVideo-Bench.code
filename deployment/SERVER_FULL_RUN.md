# 服务器端完整实验流程

本流程把代码、数据集、视频解码/抽帧、CLIP 语义召回、Qwen2.5-VL 推理、结果和评估都放在 Windows 服务器上。服务器上的 `syf` 账号使用自己的目录，不改动另一位用户的环境。

## 目录

```text
D:\VLM\syf\
├── python311\                    # syf 的便携式 Python
├── hf-cache\                     # Hugging Face 模型缓存
├── UrbanVideo-Bench.code\        # Git 工作副本
│   ├── dataset\                  # Hugging Face 全量数据，不提交 Git
│   │   ├── MCQ.parquet
│   │   └── videos\*.mp4
│   ├── cache\agentic_retrieval\  # 抽帧/候选帧/检索缓存，不提交 Git
│   ├── result\                   # CSV、SQLite、评估表，不提交 Git
│   └── deployment\
└── logs\                         # 服务日志
```

当前 Hugging Face `main` revision 有 995 个视频，合计约 21.1 GB（19.7 GiB），并提供 `MCQ.parquet`。该 parquet 有 5355 道题、1393 个唯一视频 ID；当前 revision 只包含其中 995 个视频，直接下载覆盖 3318 道题（61.96%）。服务器 `D:` 空间足够。

这不是代码可以自动修复的数据缺口。开始“全量 5355 题”实验前，必须从数据集作者或其他合法来源补齐验证脚本列出的缺失视频。没有这些视频时，实验只能称为“可用视频子集”，不能称为完整 benchmark。

## 1. 在 GitHub 管理代码

不要把视频、模型权重、缓存或密钥提交到 Git。`.gitignore` 已排除这些内容。建议在 GitHub 创建自己的代码仓库，或 fork 官方仓库，把本地研究改动推送到自己的仓库；官方 `origin` 只作为上游参考。公开项目的 fork 通常沿用上游可见性；如果需要私有仓库，应先确认上游许可证和学校规定，再创建独立仓库。

在本机项目根目录执行，逐条检查后再提交：

```powershell
git switch -c research/server-full-run
git remote add research git@github.com:<你的GitHub用户名>/<你的仓库名>.git
git add .gitignore run.py eval.py prompts.py confidence.py agentic_retrieval.py video_locks.py vlm_api.py server.py templates deployment
git status --short
git diff --cached --stat
git commit -m "Prepare native Windows server experiments"
git push -u research research/server-full-run
```

将占位符替换成自己的 GitHub 用户名和仓库名。不要使用 `git add .`，当前工作区还有本地视频、结果和其他研究资料。提交前检查 `git status` 和暂存差异，确认没有 `.mp4`、Parquet、密钥、压缩包或模型缓存。工作区里已有的删除记录不会被上面的路径列表自动暂存。

服务器先创建暂存目录，再克隆自己的代码分支：

```powershell
New-Item -ItemType Directory -Force 'D:\VLM\syf\incoming-local-videos'
git clone --branch research/server-full-run --single-branch git@github.com:<你的GitHub用户名>/<你的仓库名>.git D:\VLM\syf\UrbanVideo-Bench.code
cd D:\VLM\syf\UrbanVideo-Bench.code
```

如果仓库是私有仓库，服务器需要配置 GitHub SSH key 或使用一次性的 PAT；不要把 PAT 写入脚本或提交到仓库。

## 2. 安装完整 benchmark 依赖

```powershell
$PythonHome = 'D:\VLM\syf\python311'
$env:PIP_CACHE_DIR = 'D:\VLM\syf\pip-cache'
& "$PythonHome\python.exe" -m pip install -r .\deployment\requirements-benchmark-windows.txt
```

这会在服务器安装 OpenCV、pandas、pyarrow、Hugging Face Hub 和 `sentence-transformers`。CLIP 模型第一次使用时会下载到 `HF_HOME` 指定的缓存目录。

## 3. 下载并验证全量数据

```powershell
$env:HF_HOME = 'D:\VLM\syf\hf-cache'
$env:HUGGINGFACE_HUB_CACHE = 'D:\VLM\syf\hf-cache\hub'
& "$PythonHome\python.exe" .\deployment\download_urbanvideo_dataset.py --data-dir .\dataset --revision main
```

下载可中断后重跑，Hub 会复用已完成文件。下载完成后验证问题引用的视频是否全部存在：

```powershell
& "$PythonHome\python.exe" .\deployment\verify_urbanvideo_dataset.py --data-dir .\dataset --expected-videos 995
```

脚本会列出问题数、视频数、可运行题数和覆盖率，并把缺失视频写入 `dataset/missing_video_ids.txt`。只有 `missing_question_videos=0` 才能开始“全量”推理。当前 HF 数据集会显示 398 个缺失视频 ID；可以先生成缺失清单，但不要把它当作全量数据。需要把缺失视为硬错误时使用：

```powershell
& "$PythonHome\python.exe" .\deployment\verify_urbanvideo_dataset.py --data-dir .\dataset --expected-videos 995 --require-complete
```

把本地已有但 Hugging Face 没有的 MP4 合并到服务器后，再重新运行验证脚本；不要按文件名盲目覆盖而丢失本地唯一视频。

在**本机项目目录**另开 PowerShell，运行随仓库提供的上传脚本，只传本机有且当前 HF revision 没有的视频：

```powershell
cd D:\airsimproject\UrbanVideo-Bench.code
.\deployment\upload_local_video_supplement.ps1 -ServerIp 10.202.83.228
```

它会先显示将上传的文件数和总大小，再通过 SCP 上传到服务器 `incoming-local-videos`。本机当前 287 个视频里有 53 个文件不在 HF 当前 revision，约 8.0 GiB；合并后可覆盖约 3564/5355 道题（66.55%），仍有 345 个视频 ID 缺失。回到**服务器 SSH PowerShell**安全合并并校验：

```powershell
cd D:\VLM\syf\UrbanVideo-Bench.code
& "$PythonHome\python.exe" .\deployment\merge_supplemental_videos.py
& "$PythonHome\python.exe" .\deployment\verify_urbanvideo_dataset.py --data-dir .\dataset
```

合并脚本计算 SHA-256，只复制新文件，对同名但内容不同的文件拒绝覆盖，并把合并清单写入 `result\supplemental_video_manifest.csv`。当前 HF 数据集加上你本机补充视频仍不完整；要跑满 5355 题，需向作者取得其余 345 个视频，或从获准的原始来源补齐。任何可用子集实验都要保存固定的 `video_id` 清单，报告实际题数和覆盖率。

## 4. 启动本地 Qwen 服务

在第一个 SSH PowerShell 窗口运行，模型首次启动会下载权重：

```powershell
$env:HF_HOME = 'D:\VLM\syf\hf-cache'
$env:HUGGINGFACE_HUB_CACHE = 'D:\VLM\syf\hf-cache\hub'
$env:VLM_MAX_NEW_TOKENS = '256'
$env:VLM_MAX_IMAGES = '64'
$env:VLM_MIN_PIXELS = '12544'
$env:VLM_MAX_PIXELS = '112896'
$env:VLM_MAX_INPUT_TOKENS = '12288'
$env:VLM_CPU_THREADS = '4'
$env:VLM_ATTN_IMPLEMENTATION = 'sdpa'
& "$PythonHome\python.exe" .\deployment\windows_vlm_server.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

服务只监听 `127.0.0.1:8000`。在第二个 SSH 窗口检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5
```

看到 `status` 为 `ok` 后再继续。服务只允许一次 GPU 生成；这不会限制另一位用户的显卡使用，但你的实验必须约定时间窗口并观察 `nvidia-smi`。

## 5. 在服务器端运行 CLIP、抽帧和 benchmark

第二个 SSH 窗口切到仓库根目录。`run.py` 会在服务器上用 OpenCV 读取视频并抽帧；Agentic 类别会在服务器上用 CLIP 语义召回，缓存写入 `cache/agentic_retrieval`：

```powershell
cd D:\VLM\syf\UrbanVideo-Bench.code
$env:HF_HOME = 'D:\VLM\syf\hf-cache'
$env:HUGGINGFACE_HUB_CACHE = 'D:\VLM\syf\hf-cache\hub'
$env:VLM_PROVIDER = 'qwen'
$env:QWEN_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:QWEN_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
$env:OPENAI_API_KEY = 'local'
$env:VLM_TIMEOUT_SECONDS = '900'
$env:VLM_OUTER_CONCURRENCY = '1'
$env:AGENTIC_SEMANTIC_ALLOW_DOWNLOAD = '1'
$env:AGENTIC_RETRIEVAL_EMBED_MODEL = 'clip-ViT-B-32'
# 共享 GPU 时可把 Agentic CLIP 放在 CPU，减少额外显存；会增加 CPU 时间。
$env:AGENTIC_RETRIEVAL_DEVICE = 'cpu'
$env:RUN_ID = 'qwen25vl7b_full_server_agentic'
$env:RUN_MODE = 'new'
& "$PythonHome\python.exe" .\run.py
```

结果会留在服务器的 `result/`，包括结果 CSV、dashboard 数据库和 `eval.py` 生成的评估表。改变模型、帧数、分辨率、Agentic 开关或提示词时使用新的 `RUN_ID`，不要覆盖已有实验。当前 `run.py` 默认运行 Agentic 检索策略；它不是纯均匀抽帧基线，论文里应按实际策略命名。基线和消融应使用明确独立的参数/分支与 `RUN_ID`。

评估：

```powershell
& "$PythonHome\python.exe" .\eval.py
```

运行前后保存环境和 Git 提交号，论文中报告它们：

```powershell
git rev-parse HEAD | Set-Content .\result\git_commit.txt
& "$PythonHome\python.exe" -m pip freeze | Set-Content .\result\installed-packages.txt
nvidia-smi | Set-Content .\result\nvidia-smi.txt
```

`result/` 被 Git 忽略，但文件仍会保存在服务器；建议另行备份到实验存储或用 Git LFS/Release 管理小型结果表。
