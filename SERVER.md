# MindSurf 服务器使用说明

更新时间：2026-05-27  
用途：在本项目中复用 MindSurf 计算服务器进行训练、推理或评测。

## 1. 连接前提

先打开 WireGuard，并启用学校/科协提供的隧道：

```text
sast-public-mindsurf
```

如果 SSH 超时，先检查 WireGuard 日志是否反复出现 handshake 失败；这种情况通常不是项目问题，而是组网还没有连通。

## 2. SSH 配置

本机已经配置了 SSH alias：

```sshconfig
Host mindsurf mindsurf-oscar
    HostName 192.168.101.129
    User oscar
    Port 2222
    IdentityFile C:/Users/oscar/.ssh/mindsurf_oscar_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking yes
```

连接命令：

```bash
ssh mindsurf
```

说明：

- 不要把私钥内容、密码或 WireGuard 配置提交到项目仓库。
- `IdentityFile` 只记录本机私钥路径，不复制私钥。
- 服务器使用个人 Linux 用户 `oscar`，工作目录放在 `/home/oscar` 下。

## 3. 服务器当前信息

| 项 | 当前值 |
| --- | --- |
| 主机名 | `mindsurf` |
| 登录用户 | `oscar` |
| 默认 home | `/home/oscar` |
| GPU | NVIDIA GeForce RTX 4090 |
| 显存 | 24564 MiB |
| Driver | 595.71.05 |
| Python | 3.12.3 |
| 可用工具 | `git`、`tmux`、`nvidia-smi` |

快速检查：

```bash
ssh mindsurf "hostname; whoami; nvidia-smi"
```

## 4. 本项目建议放置位置

建议服务器侧路径使用英文，避免脚本和远程命令处理中文路径时出问题：

```bash
/home/oscar/global-campus-ai-algorithm
```

账号与目录规则：

- 本项目只使用个人账号 `oscar`，不要使用共享账号训练、推理或保存文件。
- 如果发现以前误把本项目放到了共享账号，先把需要保留的代码、配置和输出迁移到 `/home/oscar/global-campus-ai-algorithm`，验证无误后只删除共享账号下本项目对应目录。
- 不要删除或改动 `/home/oscar/minimind`，本项目和 MiniMind 必须使用不同目录、虚拟环境、日志和输出路径。

第一次创建：

```bash
ssh mindsurf "mkdir -p /home/oscar/global-campus-ai-algorithm"
```

也可以用项目脚本做账号断言和目录创建：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\remote_setup_mindsurf.ps1
```

从 Windows 上传当前项目：

```powershell
scp -r "D:\UserData\Desktop\全球校园人工智能算法\*" mindsurf:/home/oscar/global-campus-ai-algorithm/
```

如果项目后续接入 Git，优先用 Git 同步代码；大数据集、模型权重、日志不要直接进 Git。

## 5. 推荐运行方式

进入项目：

```bash
ssh mindsurf
cd /home/oscar/global-campus-ai-algorithm
```

创建项目自己的虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

后台任务用 `tmux`：

```bash
tmux new -s global_ai
# 运行训练或评测命令
# Ctrl-b 后按 d 可以退出但保持任务继续运行
tmux attach -t global_ai
```

查看资源：

```bash
nvidia-smi
ps -u oscar
tmux list-sessions
```

## 6. 使用边界

- 只写 `/home/oscar/...` 下自己的项目目录。
- 不修改其他用户目录。
- 不杀其他用户进程。
- 训练前看 `nvidia-smi`，确认剩余显存够用。
- 大文件、权重、日志建议放在项目内的 `outputs/`、`checkpoints/`、`logs/`，并加入 `.gitignore`。

## 7. 和 MiniMind 项目的关系

MiniMind 当前服务器项目路径是：

```bash
/home/oscar/minimind
```

本项目不要直接复用 MiniMind 的虚拟环境或输出目录。可以参考它的做法：

- 每个实验单独输出目录；
- 用 `tmux` 跑长任务；
- 固定评估集；
- 日志、权重、结果 JSON 分开保存；
- 报告只保留一份主 README，避免多个版本互相矛盾。
