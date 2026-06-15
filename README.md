# 全球校园人工智能算法 - 城市场景视觉多模态目标检测

本项目实现“面向城市场景的视觉多模态目标检测”赛题的工程化方案。RGB YOLO 只作为链路验证和消融 baseline；正式提交默认使用 RGB/Infrared/Depth 三模态融合模型。

## 当前能力

- 自动发现并配对 `RGB`、`Infrared`、`Depth`、`labels` 目录。
- 已适配官方样例命名：`visible`、`infrared`、`depth`、`labels`。
- 校验 YOLO 标签：`class_id norm_center_x norm_center_y norm_w norm_h`。
- 生成 Ultralytics 标准训练目录和 `data.yaml`。
- 支持两种输入：
  - `rgb`：RGB-only baseline，仅用于链路验证和消融，不作为官方推荐提交；
  - `triad3`：RGB 亮度 + 红外增强 + 深度归一化三通道早期融合。
  - `cssa3`：基于红外/近距离深度显著图的轻量空间注意力融合。
- 支持 `--modality-dropout` 离线扩增，生成 `drop_rgb/drop_ir/drop_depth` 训练样本，提升模态质量下降时的鲁棒性。
- 训练、验证、推理和官方 TXT 提交文件生成。
- 每张测试图强制生成同名 TXT；无检测则空文件；最多 100 框。

## 环境

建议服务器使用 Python 3.10-3.12。服务器说明见 [docs/server_runbook.md](docs/server_runbook.md) 和本地 [SERVER.md](SERVER.md)。

远端只使用 `oscar` 账号和 `/home/oscar/global-campus-ai-algorithm`，不要使用共享账号，也不要复用 `/home/oscar/minimind`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 本地只建议做代码检查和小样本验证，正式训练放到 MindSurf 4090 服务器。

## 快速流程

没有官方数据时，可以先生成一个合成小样本做烟测：

```bash
python scripts/make_sample_dataset.py --out-root tmp/sample_official_dataset --count 8 --size 128
python scripts/inspect_dataset.py --raw-root tmp/sample_official_dataset
python scripts/prepare_dataset.py --raw-root tmp/sample_official_dataset --out-root tmp/prepared_triad3 --fusion triad3
```

检查数据：

```bash
python scripts/inspect_dataset.py --raw-root /path/to/official/train
```

准备 RGB baseline：

```bash
python scripts/prepare_dataset.py --raw-root /path/to/official/train --out-root datasets/city_rgb --fusion rgb
```

准备三模态早期融合：

```bash
python scripts/prepare_dataset.py --raw-root /path/to/official/train --out-root datasets/city_triad3 --fusion triad3
```

准备论文驱动的 `cssa3 + modality dropout` 实验：

```bash
python scripts/prepare_dataset.py \
  --raw-root /path/to/official/train \
  --out-root datasets/city_cssa3_dropout \
  --fusion cssa3 \
  --modality-dropout \
  --workers 16
```

训练：

```bash
python scripts/train_yolo.py --data datasets/city_triad3/data.yaml --model yolo11m.pt --epochs 120 --imgsz 960 --batch -1 --workers 16 --name triad3_yolo11m
```

验证：

```bash
python scripts/validate_yolo.py --weights outputs/runs/triad3_yolo11m/weights/best.pt --data datasets/city_triad3/data.yaml
```

推理并打包提交：

```bash
python scripts/predict_submit.py \
  --weights outputs/runs/triad3_yolo11m/weights/best.pt \
  --raw-root /path/to/official/test \
  --fusion triad3 \
  --out-dir outputs/submission_txt \
  --zip-path outputs/submission.zip
```

官方提交模型选择：

```bash
python scripts/select_best_run.py --project-dir outputs/runs --require-finished --json
```

该命令默认只在 `triad3/cssa3` 三模态候选中选择。若只是做 RGB baseline 消融分析，才显式加入：

```bash
python scripts/select_best_run.py --project-dir outputs/runs --allow-rgb-baseline --json
```

## 实验路线

1. RGB YOLO baseline：确认数据、标签、训练和提交全链路，仅作 baseline。
2. `triad3` 三模态早期融合：低风险引入红外和深度。
3. `cssa3 + modality dropout`：引入轻量空间注意力和模态退化增强。
4. 提高输入尺寸并优化小目标召回。
5. 调整 confidence、NMS 和 TTA。
6. 视排行榜反馈推进三分支特征级融合。当前已完成的是三模态早期融合和轻量空间注意力融合，不应在报告中写成已实现三分支 backbone。

## 服务器并行策略

MindSurf 是 4090/24GB，服务器训练默认按高并行配置使用：

- `scripts/prepare_dataset.py` 默认最多 16 个 workers 并行生成 triad3 图像。
- `scripts/train_yolo.py` 默认 `--batch -1`，由 Ultralytics 自动估计 batch size，目标是更充分利用显存。
- 默认 `--workers 16` 加速 dataloader。

正式长任务前仍运行 `nvidia-smi`，确认没有占满 GPU 的其他任务。

## 测试

当前测试覆盖图像预处理、标签解析、数据目录发现、提交格式和 Ultralytics 结果转换。

```powershell
$env:TEMP='D:\UserData\Desktop\全球校园人工智能算法\tmp'
$env:TMP=$env:TEMP
python -m pytest tests
```
