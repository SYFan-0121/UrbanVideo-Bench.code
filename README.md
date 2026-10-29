# UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces

本文用于项目说明、版本入口与逐文件功能索引。

The benchmark is designed to evaluate whether video-large language models (Video-LLMs) can naturally process continuous first-person visual observations like humans, enabling recall, perception, reasoning, and navigation.

- **Arxiv**: https://arxiv.org/pdf/2503.06157
- **Project**: https://embodiedcity.github.io/UrbanVideo-Bench/
- **Dataset**: https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench

## 分支与实验版本

默认分支为 [`main`](https://github.com/SYFan-0121/UrbanVideo-Bench.code/tree/main)，用于维护当前 16 类 Agentic 代码。旧版源码与两个固定实验版本均可直接访问：

| 入口 | 用途 |
| --- | --- |
| [`main`](https://github.com/SYFan-0121/UrbanVideo-Bench.code/tree/main) | 当前维护版本：16 类均走 Agentic；支持 STRIVE 或 no_vote |
| [`codex/archive-agentic-12class`](https://github.com/SYFan-0121/UrbanVideo-Bench.code/tree/codex/archive-agentic-12class) | 旧版归档：12 类 Agentic 检索，其余 4 类沿用原始 STRIVE 路由 |
| [`agentic-12class-original-20261029`](https://github.com/SYFan-0121/UrbanVideo-Bench.code/tree/agentic-12class-original-20261029) | 旧版实验固定标签，对应 `bfea96b` |
| [`agentic-16class-20261029`](https://github.com/SYFan-0121/UrbanVideo-Bench.code/tree/agentic-16class-20261029) | 新版实验固定标签，对应 `30acd25`，不随文档维护移动 |

原 `codex/research-server-full-run` 已归档为 `codex/archive-agentic-12class`；原 `codex/agentic-16class` 与 `main` 重复，整理后统一使用 `main`。两个固定标签保留原始提交及完整历史。复现实验时选固定标签，日常维护使用 `main`，详细差异见 [版本记录](deployment/CODE_VERSIONS.md)。

## 文件功能索引

GitHub 文件列表右侧显示的是该路径最近一次提交的说明。本次用文件功能作为相应提交说明；以后修改文件时，该列会显示新的提交说明。下表提供持续维护的功能索引。

| 文件 | 功能 |
| --- | --- |
| [`.gitignore`](.gitignore) | 排除数据、权重、密钥、缓存和实验产物 |
| [`README.md`](README.md) | 项目说明、版本入口与逐文件功能索引 |
| [`agentic_retrieval.py`](agentic_retrieval.py) | 检索规划、多源候选池、可选 CLIP 召回、VLM 重排与帧预算融合 |
| [`confidence.py`](confidence.py) | STRIVE 多次投票、置信度计算与低置信度自适应重试 |
| [`eval.py`](eval.py) | 提取预测答案、统计总体及分类准确率并导出 Excel 报告 |
| [`prompts.py`](prompts.py) | 按 16 类生成问答提示，并根据实际选项确定答案范围 |
| [`run.py`](run.py) | 执行 16 类 Agentic 评测、抽帧、断点续跑与结果保存 |
| [`run_uniform32.py`](run_uniform32.py) | 独立执行均匀 32 帧单次回答基准，逐题保存原始输出并支持续跑 |
| [`server.py`](server.py) | 提供实时进度接口、Socket.IO 推送、SQLite 历史查询与视频回看 |
| [`video_locks.py`](video_locks.py) | 按视频路径共享线程锁，协调同一视频的并发解码 |
| [`vlm_api.py`](vlm_api.py) | 统一 MiMo、Qwen 与 OpenAI 兼容服务的请求及参数适配 |
| [`templates/dashboard.html`](templates/dashboard.html) | 展示实验队列、题目选项、证据帧、答案与历史结果 |
| [`question_generation/MCQ_generation_basic.py`](question_generation/MCQ_generation_basic.py) | 提取视频运动及物体信息，生成基础感知与推理选择题 |
| [`question_generation/MCQ_generation_goal.py`](question_generation/MCQ_generation_goal.py) | 结合视频与目标地点生成目标导航选择题 |
| [`question_generation/MCQ_generation_route.py`](question_generation/MCQ_generation_route.py) | 结合视频与移动指令生成路线导航选择题 |
| [`deployment/CODE_VERSIONS.md`](deployment/CODE_VERSIONS.md) | 记录分支用途、固定实验标签、版本差异与验证结果 |
| [`deployment/SERVER_FULL_RUN.md`](deployment/SERVER_FULL_RUN.md) | 说明服务器端数据准备、模型启动、完整评测与结果评估流程 |
| [`deployment/WINDOWS_NATIVE_DEPLOYMENT.md`](deployment/WINDOWS_NATIVE_DEPLOYMENT.md) | 说明 Windows 原生 VLM 环境、SSH 隧道与模型服务部署 |
| [`deployment/download_urbanvideo_dataset.py`](deployment/download_urbanvideo_dataset.py) | 下载 Hugging Face 指定版本的数据集并记录下载清单 |
| [`deployment/merge_supplemental_videos.py`](deployment/merge_supplemental_videos.py) | 校验补充视频的 SHA256 并合并入库，保留冲突文件记录 |
| [`deployment/requirements-benchmark-windows.txt`](deployment/requirements-benchmark-windows.txt) | 声明完整评测、CLIP 检索、数据处理与看板依赖 |
| [`deployment/requirements-windows.txt`](deployment/requirements-windows.txt) | 声明 Windows 本地 Qwen2.5-VL 推理服务依赖 |
| [`deployment/smoke_test.py`](deployment/smoke_test.py) | 向本地 VLM 服务发送 1–64 张合成图像，检查多图推理接口 |
| [`deployment/test_windows_vlm_server.py`](deployment/test_windows_vlm_server.py) | 使用模拟生成器检查 VLM 服务接口、并发与请求适配 |
| [`deployment/upload_local_video_supplement.ps1`](deployment/upload_local_video_supplement.ps1) | 筛选 Hugging Face 未提供的本地视频并通过 SSH 上传 |
| [`deployment/verify_urbanvideo_dataset.py`](deployment/verify_urbanvideo_dataset.py) | 对照题库统计视频覆盖率并导出缺失视频清单 |
| [`deployment/windows_vlm_server.py`](deployment/windows_vlm_server.py) | 加载本地 Qwen2.5-VL 并提供 OpenAI 兼容多图推理接口 |
| [`tests/dashboard.test.cjs`](tests/dashboard.test.cjs) | 回归检查题目选项解析、等待状态与答案高亮 |
| [`tests/test_agentic_modes.py`](tests/test_agentic_modes.py) | 检查全部 16 类的两种 Agentic 模式、32 帧预算与结果保存 |
| [`tests/test_question_options.py`](tests/test_question_options.py) | 检查 2–7 项题目的提示范围、答案提取与 G 选项评估 |
| [`tests/test_uniform32.py`](tests/test_uniform32.py) | 检查独立均匀基准的采样一致性、模型调用与失败题续跑 |

## 研究服务器使用

仓库保存可复现源码与部署工具；数据集、视频、模型权重、API 密钥、缓存和实验结果保存在 Git 之外，由部署脚本在服务器上准备。

For the complete Windows workflow, read [`deployment/WINDOWS_NATIVE_DEPLOYMENT.md`](deployment/WINDOWS_NATIVE_DEPLOYMENT.md) first, then [`deployment/SERVER_FULL_RUN.md`](deployment/SERVER_FULL_RUN.md). The latter covers Hugging Face download and verification, local video supplementation, local Qwen2.5-VL service startup, server-side OpenCV/CLIP/inference, evaluation, and saving the Git commit and environment metadata with each experiment.

The current inference code reads provider configuration from environment variables. For a local Qwen-compatible service, use `VLM_PROVIDER=qwen`, `QWEN_BASE_URL=http://127.0.0.1:8000/v1`, and `OPENAI_API_KEY=local`. For a cloud provider, set its API key in the shell; never write the key into a Python file or commit it.

Use a distinct `RUN_ID` for every baseline or ablation. This keeps CSV, dashboard SQLite, and evaluation outputs separate while the Git commit records the exact code version.

Use `python run.py` for Agentic experiments: all 16 categories use Agentic retrieval, with `AGENTIC_GLOBAL_MODE=strive` or `AGENTIC_GLOBAL_MODE=no_vote`. This file is restored exactly from commit `9f0b9cdb324bdc62ad3b46b632d46c73c9bb09dd`.

Use the separate `run_uniform32.py` for the full-video uniform 32-frame baseline. It adapts the supplied `latest_model_32frame_benchmark_test.py`: serial question processing, one answer call, per-question CSV saves and resume. It decodes only the 32 selected frames, uses the project prompts and A–G option handling, and reads API configuration from the environment. It does not import or modify `run.py`. Original responses are saved in `RawOutput`; failed questions remain eligible for resume.

```powershell
python run_uniform32.py --run-id qwen25vl7b_uniform32_baseline
# Resume this baseline; the model and input configuration must remain the same.
python run_uniform32.py --run-id qwen25vl7b_uniform32_baseline --resume
```

All modes use 32 base frames per answer. Fixed versions are documented in [`deployment/CODE_VERSIONS.md`](deployment/CODE_VERSIONS.md).

## News
🎉 Accepted as an oral presentation at ACL 2025!

✅ Dataset Upload

✅ Dataset generation code

✅ Example code for running the benchmark with Video-LLMs

## Dataset generation
The pipeline includes four steps: video curation, MCQ generation, blind filtering, and human refinement. 


We provide three seperated scripts: Basic, Goal, and Route for generating questions with Gemini, as the workflows are slightly different due to differences in the forms of input data and features of tasks:

- For data consists of videos with destinations, we use the **Goal** script for the generation. This script is capable of generating question of the following categories:
   ```python
   question_categories = [
      "Trajectory Captioning",
      "Landmark Position",
      "Goal Detection",
      "Association",
      "Cognitive Map",
      "High-level Planning",
      "Action Generation"
   ]
   ```

- For data that include videos collected with specific movement instructions, we use the **Route** script to generate the questions. This script is capable of generating question of the following categories:
   ```python
   question_categories = [
    "Progress Evaluation",
    "Landmark Position",
    "Action Generation"
   ]
   ```

- For some questions in the Recall and Perception category, details from the videos are important, so we introduce an extra chain of thought in the **Basic** script, where objects and movements from the videos are extracted in advance and fed into the model for final generation. This script is capable of generating question of the following categories:
   ```python
   question_categories = [
    "Trajectory Captioning",   
    "Start/End Position",    
    "Object Recall",    
    "Sequence Recall",
    "Scene Recall", 
    "Proximity",
    "Duration",
    "Causal",
    "Counterfactual"
   ]
   ```


Follow the steps below to configure and execute the script `question_generation/MCQ_generation_basic.py` (the other two are similar):

1. Set your Gemini API key and select the appropriate model version:
   ```python
   model = "gemini-1.5-flash"
   genai.configure(api_key="SET_YOUR_API_KEY_HERE")
   ```

2. Configure the input and output paths:
   - **Input Path**: Specify the folder path containing `video_list.json` and the `.MP4` videos to be processed.
   ```python
   video_path = rf"DIRECT\PATH\TO\YOUR\VIDEO" # Replace with your video path. This path should contain video files with video_list.json.
   ```
   
   ```
   DIRECT\PATH\TO\YOUR\VIDEO/
   ├── video_list.json
   ├── video_1.mp4
   ├── video_2.mp4
   └── ...              # All the videos recorded in video_list.json
   ```
   - **Output Path**: Specify the path to the `.CSV` file where the results will be saved.
   ```python
   MCQ_PATH = rf"DIRECT\PATH\TO\YOUR\MCQ\FILE.csv" # Set your output MCQ file here. 
   ```

3. Finally, if you have set the direct path to your input and output files, you can execute the script by simply running the following command in the terminal:
   ```bash
   python question_generation/MCQ_generation_basic.py
   ```
   The results will be saved to the specified file.

## Example code for running the benchmark
### Data Preparation

To get started, download the dataset from [Hugging Face](https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench) 
and place it in the `dataset` folder within the project directory. 
After downloading, ensure the folder structure matches the one described below.
```
UrbanVideo-Bench.code/
├── dataset/
│   ├── videos/          # Contains video files used as input for the model
│   ├── MCQ.parquet      # Contains multiple-choice questions
│   └── ...
├── run.py               # Script for running the model and generating predictions
├── eval.py              # Script for evaluating the model's predictions
├── README.md            # Documentation for the project
└── ...                  # Other potential files or subdirectories
```


### Running

We provide a sample script, `run.py`, to run the dataset using an OpenAI-style API. Follow the steps below to configure and execute the script:

1. Set the model name in `run.py`:
   ```python
   model = "your_model_name"
   ```

2. Configure OpenAI API credentials:
   ```python
   client = OpenAI(
       api_key='your_api_key',
       base_url='your_base_url'
   )
   ```

3. Run the script:
   ```bash
   python run.py
   ```
   Results will be saved to `result/%s_output.csv`.


### Evaluation

The `eval.py` script is provided to evaluate the model's predictions. It extracts the options from the model's output and calculates the accuracy by comparing them to the ground truth.

1. Modify the file path in `eval.py` to match the output file from `run.py`:
   ```python
   file_path = 'result/gpt-4o_output.csv'  # Replace with your output file path
   ```

2. Run the script:
   ```bash
   python eval.py
   ```

3. The script compares predictions to ground truth and calculates accuracy. Results are saved to: `result/%s_acc.xlsx`

*Note: The extraction method here is the simplest regular matching. However, the output of small-sized models often does not follow instructions. So it needs to be adjusted separately.*


## Acknowledgements

We would like to express our sincere gratitude to [ZhanxyR](https://github.com/ZhanxyR) for the valuable contribution to the VLMEvalKit library in the [project](https://github.com/ZhanxyR/UrbanVideo-to-tsv), which has further improved the usability of this project. Thank you for your efforts and dedication!


## Citation

If you use this project in your research, please cite the following paper:

```
@misc{zhao2025urbanvideobench,
      title={UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces}, 
      author={Baining Zhao and Jianjie Fang and Zichao Dai and Ziyou Wang and Jirong Zha and Weichen Zhang and Chen Gao and Yue Wang and Jinqiang Cui and Xinlei Chen and Yong Li},
      year={2025},
      eprint={2503.06157},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06157}, 
}
```
