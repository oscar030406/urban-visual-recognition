# 服务器运行说明

本项目可以在本地做代码检查和小样本测试，正式训练建议放到 GPU 服务器。

## 使用原则

1. 每个项目使用独立目录和独立 `.venv`。
2. 训练前先检查 `nvidia-smi` 和当前用户进程。
3. 可以尽量使用空闲显存，但不要 OOM。
4. 不杀其他用户进程，不改系统环境。
5. 数据、权重、日志、提交包不要提交到 Git。

## 建议服务器目录

```bash
/data/oscar/<project-name>
```

当前项目历史服务器目录：

```bash
/data/oscar/global-campus-ai-algorithm
```

## 基础环境

```bash
cd /data/oscar/<project-name>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

若需要公开预训练权重，先下载到本地路径，例如：

```bash
mkdir -p weights
python scripts/download_yolo_weights.py --out-dir weights
```

## 常用检查

```bash
nvidia-smi
ps -u "$USER" -o pid,stat,etime,pcpu,pmem,cmd
```

## 后台训练

```bash
mkdir -p logs
nohup .venv/bin/python scripts/train_yolo.py \
  --data datasets/city_rgb_guided_rdt/data.yaml \
  --model weights/yolo11m.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch auto-free \
  --workers 16 \
  --name rgb_guided_rdt_yolo11m \
  > logs/rgb_guided_rdt_yolo11m.log 2>&1 &
```

查看日志：

```bash
tail -f logs/rgb_guided_rdt_yolo11m.log
```

## 提交包校验

```bash
python scripts/validate_submission.py \
  --submission outputs/submission.zip \
  --raw-root data/测试集/AIC2026_PHASE_1_1000
```
