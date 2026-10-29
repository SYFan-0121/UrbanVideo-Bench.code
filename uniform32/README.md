# 无水印均匀 32 帧基准

此文件夹独立运行均匀抽帧基准，不导入 Agentic/STRIVE 实现。超过 32 帧的视频无水印均匀抽取 32 帧；短视频上传所有帧。每题单次回答。结果和看板数据写入本目录 `result/`。

数据集从 `dataset` 目录联接读取共享的 `../dataset`。运行前配置 VLM 环境变量：

```powershell
python -m pip install -r requirements.txt
python launch.py run_uniform32.py --run-id uniform32_experiment_name
```

中断续跑：

```powershell
python launch.py run_uniform32.py --run-id uniform32_experiment_name --resume
```

验证：`python launch.py -m unittest discover -s tests -v`。
