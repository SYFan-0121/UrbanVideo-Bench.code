# Windows 服务器上的本地 Video-VLM 部署

本文针对一台只允许使用 Windows 的共享服务器：当前截图显示单张 Quadro RTX 8000（`46080 MiB`，约 45 GiB 可见显存）、128 GB 内存、Xeon 4215R。方案是 Windows 原生 Python + CUDA PyTorch + Transformers。服务器和你的电脑都运行 Windows。

优先跑通 Qwen2.5-VL-7B-Instruct 的 FP16 推理；资源紧张时可把模型名换成同架构的 Qwen2.5-VL-3B-Instruct。此脚本专门适配 Qwen2.5-VL，其他架构需要另外的加载代码。RTX 8000 属于 Turing，先使用 FP16，不按原生 BF16/FlashAttention-2 路线配置。这里给出的是部署候选配置，尚未在你的服务器上完成 GPU 实测。

服务器上只启动一个推理进程，并把服务绑定到 `127.0.0.1:8000`。你的电脑通过 OpenSSH 端口转发访问它；服务端口不会直接暴露给局域网。SSH 登录不会接管另一位用户的桌面会话，但模型运行会和对方争用同一张 GPU，不能保证对方完全无感。

## 1. 服务器管理员只做一次的配置

以下命令必须在服务器上以“管理员身份运行”的 PowerShell 执行。普通研究账号不需要管理员权限。

先查看并按需安装 OpenSSH Server。微软当前的能力名称是 `OpenSSH.Server~~~~0.0.1.0`：

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
Get-Service sshd
```

如果查询结果已经是 `Installed`，不要重复安装；直接执行 `Start-Service` 和 `Set-Service`。安装通常会自动创建 TCP 22 入站防火墙规则。若规则不存在，管理员可以只允许本地网段，例如把 `192.168.1.0/24` 改成你们实际网段：

```powershell
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (TCP 22)' `
    -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow -Profile Private `
    -RemoteAddress 192.168.1.0/24
}
```

为实验建立单独的普通 Windows 账号。不要把它加入 Administrators，除非管理员明确要求：

```powershell
$password = Read-Host 'researcher password' -AsSecureString
New-LocalUser -Name researcher -Password $password `
  -Description 'Video VLM experiments'
```

查服务器的局域网 IPv4 地址：

```powershell
ipconfig
```

记下实际的 `IPv4 Address`，下面用 `192.168.1.100` 作为示例。不要把示例地址原样使用。

## 2. 从本机 SSH 登录

在你的 Windows PowerShell 中执行：

```powershell
Test-NetConnection 192.168.1.100 -Port 22
ssh researcher@192.168.1.100
```

第一次连接会显示服务器主机密钥指纹。应让管理员在服务器执行下面的命令，把指纹与第一次 SSH 显示的指纹核对后再输入 `yes`：

```powershell
ssh-keygen -lf C:\ProgramData\ssh\ssh_host_ed25519_key.pub
```

SSH 密码输入时不会回显，这是正常行为。登录后如果看到的是 `cmd.exe`，输入：

```cmd
powershell -NoProfile
```

以后命令就在 PowerShell 中运行。打开第二个本机 PowerShell 窗口，再次 SSH 登录，专门用于检查服务或运行 benchmark；不要关闭第一个窗口中的前台模型服务。

**位置辨认**：SSH 登录后的窗口执行服务器命令；尚未 SSH 的窗口执行本机命令。可运行 `hostname` 确认当前操作的是哪台电脑。这里的“控制”是远程执行命令，不是远程桌面接管。

## 3. 在服务器创建 Python 环境

先确认 Python 和磁盘位置。不要假定一定使用 `C:`；模型缓存和虚拟环境应放在有足够空间的磁盘。Qwen2.5-VL-7B 权重、缓存和临时文件需要几十 GB 空间：

```powershell
py -0p
Get-PSDrive -PSProvider FileSystem
```

以下假设实验目录为 `C:\Users\researcher\UrbanVideo-Bench.code`。如果代码在 `D:` 或其他目录，把后续路径全部替换成实际路径。服务器先运行：

```powershell
New-Item -ItemType Directory -Force C:\Users\researcher\UrbanVideo-Bench.code\deployment
```

如果 `py -0p` 找不到 Python，先安装 python.org 提供的 Windows x64 CPython 3.10 或 3.11（只装在你的账号下，不覆盖对方环境）；只有 3.11 时把后续 `py -3.10` 改成 `py -3.11`。

不要先把整个项目用 `scp -r` 复制过去：当前本地项目包含虚拟环境、结果文件和约 23 GB 视频。第一阶段可以让本机 benchmark 抽帧，再通过 SSH 隧道调用服务器；只需把 `deployment` 目录和必要脚本复制到服务器。

如果代码尚未在服务器上，可从本机执行（按实际本机路径修改）：

```powershell
cd D:\airsimproject\UrbanVideo-Bench.code
scp deployment\windows_vlm_server.py deployment\requirements-windows.txt deployment\smoke_test.py `
  researcher@192.168.1.100:C:/Users/researcher/UrbanVideo-Bench.code/deployment/
```

在服务器 SSH PowerShell 中：

```powershell
cd C:\Users\researcher\UrbanVideo-Bench.code
py -3.10 -m venv .venv-vlm
& .\.venv-vlm\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-vlm\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cu126
& .\.venv-vlm\Scripts\python.exe -m pip install -r deployment\requirements-windows.txt
```

这里使用的是 CUDA 12.6 PyTorch wheel。`nvidia-smi` 中显示的 CUDA 版本是驱动兼容能力，不等于服务器已经安装了 CUDA Toolkit；普通推理不需要另外安装 Toolkit。不要在共享服务器上自行更新 NVIDIA 驱动或重启。

验证 GPU：

```powershell
& .\.venv-vlm\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
nvidia-smi
```

应看到 `True` 和 `Quadro RTX 8000`。如果是 `False`，先停止，不要下载模型；检查 Python 环境是否安装了 CUDA wheel，以及驱动是否允许该版本 PyTorch。

## 4. 启动 Qwen2.5-VL 服务

第一次启动会从 Hugging Face 下载 `Qwen/Qwen2.5-VL-7B-Instruct`。模型缓存默认在当前 Windows 用户目录；如果 C 盘空间不足，先设置到大磁盘，例如：

```powershell
$env:HF_HOME = 'D:\hf-cache'
```

启动服务（使用虚拟环境的 Python 可执行文件，不依赖 PowerShell 执行策略）：

```powershell
cd C:\Users\researcher\UrbanVideo-Bench.code
$env:VLM_MAX_NEW_TOKENS = '256'
$env:VLM_MAX_IMAGES = '64'
$env:VLM_MIN_PIXELS = '12544'
$env:VLM_MAX_PIXELS = '112896'
$env:VLM_MAX_INPUT_TOKENS = '12288'
$env:VLM_CPU_THREADS = '4'
$env:VLM_ATTN_IMPLEMENTATION = 'sdpa'
# 如需限制同一服务器上其他本地进程的调用，取消下一行注释并换成长随机字符串。
# $env:VLM_API_KEY = '换成随机长字符串'
& .\.venv-vlm\Scripts\python.exe deployment\windows_vlm_server.py `
  --model Qwen/Qwen2.5-VL-7B-Instruct
```

配置使用 FP16 和 `sdpa`，没有安装 FlashAttention-2。PyTorch 根据 GPU 选择可用的 SDPA 后端，必要时使用数学实现；长上下文仍可能消耗大量显存。服务固定单请求生成，默认最多 64 张图、单图 `336×336` 的像素预算（保持比例并对齐模型 patch）、输入最多 12288 token、输出最多 1024 token；未指定生成长度时默认 256 token。CPU 的 torch 计算线程限制为 4，但这不是所有 CPU/内存/显存资源的系统配额。

**请先做 1 张图的 HTTP/模型测试，再逐步测 8、16、32 帧及重试的 48/64 帧。不要第一次就直接跑全量。** `smoke_test.py --frames 8` 用合成图片测试输入规模，默认单张 336×336；它不是视频准确率实验，真实视频和长 prompt 的占用还要另测。测试过程中记录 `nvidia-smi` 的占用和耗时。32 帧可运行不代表 64 帧重试也可运行。超限请求会明确报错，不会偷偷截帧。正式实验需统一分辨率、帧预算并重新命名 RUN_ID；改变这些条件会改变论文的实验设定。

若约定最多给你的进程约 24 GiB PyTorch 分配器预算，可在启动前设置：

```powershell
$env:VLM_CUDA_MEMORY_GIB = '24'
```

这个值仅限制本进程 PyTorch 分配器，**不预留显存、不限制 GPU 算力利用率，也不能保证总占用或对方的性能**。不设置则不施加此分配器上限。

在第二个 SSH 窗口检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

看到 `status: ok` 后再运行 smoke test：

```powershell
& .\.venv-vlm\Scripts\python.exe deployment\smoke_test.py
```

smoke test 只发送一张合成 JPEG，验证 HTTP、图片解码、模型生成和返回格式；它不代表完整视频 benchmark 已经跑通。

设置 API key 后，调用者必须发送 `Authorization: Bearer <同一字符串>`；运行
`smoke_test.py` 的第二个 SSH 窗口也要设置同名 `$env:VLM_API_KEY`。服务仍然只绑定回环地址，SSH 隧道是主要访问方式；不要把服务改成 `0.0.0.0`，除非管理员另行配置认证和防火墙。

`VLM_REQUEST_TIMEOUT_SECONDS`（默认 900 秒）只限制等待空闲推理槽的时间，不是生成取消定时器；客户端关闭或超时也不保证正在进行的 GPU 生成立刻停止。

## 5. 用 SSH 隧道从本机访问

在本机另一个 PowerShell 窗口保持下面的命令运行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8000:127.0.0.1:8000 researcher@192.168.1.100
```

此时本机的 `127.0.0.1:8000` 会转发到服务器的 `127.0.0.1:8000`。不要再启动本机的另一个服务占用 8000；也可以把左侧端口改成 `18000`：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18000:127.0.0.1:8000 researcher@192.168.1.100
```

测试转发：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

如果使用 `18000`，把后续 `QWEN_BASE_URL` 改成 `http://127.0.0.1:18000/v1`。关闭 SSH 隧道窗口就会停止转发，不会停止服务器上的前台模型进程。

## 6. 接入 UrbanVideo-Bench

推荐第一阶段在本机运行 benchmark：视频和结果留在本机，抽出的 base64 JPEG 经过隧道送到服务器。这样不用复制 23 GB 数据，也不会在服务器上重复安装项目的其他依赖。本机 PowerShell 设置：

```powershell
cd D:\airsimproject\UrbanVideo-Bench.code
$env:VLM_PROVIDER = 'qwen'
$env:QWEN_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:QWEN_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
$env:OPENAI_API_KEY = 'local'
$env:VLM_TIMEOUT_SECONDS = '900'
$env:VLM_OUTER_CONCURRENCY = '1'
$env:RUN_ID = 'qwen25vl7b_windows_smoke'
$env:RUN_MODE = 'new'
& .\venv\Scripts\python.exe deployment\smoke_test.py
# 分级测试通过并准备正式运行后，再执行下面这行：
& .\venv\Scripts\python.exe run.py
```

若服务设置了 `VLM_API_KEY`，本机也需设置同名变量；此时适配器优先使用该 key。本地回环端点不会优先拿遗留的 Dashscope key。这里的 `VLM_PROVIDER=qwen` 只是复用项目的 Qwen 消息适配代码；URL 已经指向本地服务，请求不会发送给 Dashscope。

如果把 benchmark 也部署到了服务器，使用同样的环境变量，端点为服务器的 `http://127.0.0.1:8000/v1`，无需 SSH 隧道；还需复制视频/问答数据和项目代码，并另外安装 benchmark 的依赖，不能复制本机虚拟环境使用。

`VLM_OUTER_CONCURRENCY=1` 控制 QA 外层并发；项目中的 STRIVE `K=3` 是实验算法参数，不能为了省资源随意改成 `K=1` 后再把结果称为同一个 STRIVE 实验。若要做单投票基线，应明确记录为单独实验配置；若要保持 K=3，三个请求会排队串行，速度较慢但算法含义不变。

服务默认 temperature=0（贪心生成）；显式传入 temperature>0 时才采样，top_p 只在此时使用。接口拒绝未实现的请求字段。本方案把视频帧作为**有序多图片**输入，没有构造 Qwen 原生 video token/时间位置输入；论文应区分“多图片抽帧评估”和“原生视频接口评估”。现有 `run.py` 包含检索/定位/投票策略，不能把整套默认运行结果直接称为纯均匀抽帧基线。

每次实验记录：模型完整名称和 revision、Python 包版本、抽帧数量和分辨率、prompt、`max_tokens`、temperature、数据子集、运行时间和 GPU 显存峰值。当前本地数据与 MCQ 匹配的是一个子集，所有模型应固定同一题号集合。以下命令可保存服务器环境；`/health` 的 revision 在可读取时显示模型提交号，复现实验启动时用 `--revision <该提交号>` 固定模型和 processor：

```powershell
& .\.venv-vlm\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 deployment\installed-packages.txt
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5 |
  Set-Content -Encoding utf8 deployment\runtime-info.json
```

## 7. 共享服务器的使用边界

另一个用户可以继续使用其 Windows 桌面。你可以同时 SSH 登录和运行服务，但同一张 RTX 8000 的显存和计算资源是共享的；`CUDA_VISIBLE_DEVICES=0` 只是选择设备，不是资源预留。开始长实验前应约定时间窗口，运行期间观察：

```powershell
nvidia-smi -l 2
```

不要执行 `taskkill /IM python.exe /F`、重启服务器、结束未知 GPU 进程、修改显示驱动或改变全局电源设置。停止服务时只在启动它的窗口按 `Ctrl+C`。若 SSH 断开，前台进程可能随会话结束；初次部署建议保持 SSH 窗口打开。需要断线后继续运行时，使用管理员批准的 Windows Task Scheduler 任务，指定 `researcher` 账号、工作目录和 `.venv-vlm\Scripts\python.exe`，不要用 Linux 的 `tmux` 或 `nohup` 假设。

## 8. 常见故障

- `ssh: connect ... port 22`：检查服务器 IPv4、Private 网络配置、防火墙和 `Get-Service sshd`。
- `CUDA unavailable`：确认使用的是 `.venv-vlm` 中的 Python，重新检查 `torch.__version__` 是否带 CUDA 后缀；不要先下载权重。
- `/health` 暂时连接不上：本脚本完成模型下载和加载后才启动 HTTP 服务，等待并查看启动窗口，不要重复启动第二个服务。
- HTTP `429`：已有请求正在占用唯一 GPU；等待前一个请求完成。
- HTTP `507`：显存不足；减少帧数、`VLM_MAX_PIXELS` 或 `max_tokens`，同时确认没有另一进程占用显存。
- 请求超时：保持 `VLM_TIMEOUT_SECONDS=900`，并检查服务窗口是否仍在生成；不要通过增加 GPU 并发解决。
- `Only model ... is loaded`：客户端的 `QWEN_MODEL` 必须与服务启动参数完全一致。

## 参考文档

- [Microsoft OpenSSH Server for Windows](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [Qwen2.5-VL model repository](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
