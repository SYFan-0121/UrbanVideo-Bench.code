# Agentic 实验

此文件夹是独立的 Agentic 实验代码，包含检索、STRIVE/no_vote、评估、实时看板和测试。`run.py` 保持固定版本 `9f0b9cdb324bdc62ad3b46b632d46c73c9bb09dd`。

运行前先配置 VLM 环境变量。数据集通过 `dataset` 下的文件硬链接读取共享的 `../dataset`，结果独立保存在本目录的 `result/`，Agentic 检索缓存保存在本目录的 `cache/`。

```powershell
python -m pip install -r requirements.txt
$env:AGENTIC_GLOBAL_MODE = 'strive'  # 或 no_vote
$env:RUN_ID = 'agentic_experiment_name'
python launch.py run.py
```

验证：`python launch.py -m unittest discover -s tests -v`；前端看板测试：`node --test tests/dashboard.test.cjs`。
