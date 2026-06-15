# 面向城市场景的视觉多模态目标检测

本项目用于全球校园人工智能算法赛“面向城市场景的视觉多模态目标检测”赛题。

当前主线方案是 `YOLO11M + RGB-guided-RDT`：保留 RGB 预训练检测器的稳定性，同时把红外和深度作为空间显著性引导，生成 YOLO 可直接训练和推理的三通道输入。

## 当前结果

| 项目 | 结果 |
|---|---:|
| 早期稳定提交 | 50.0380 |
| 当前最好提交 | 50.8190 |
| 提升 | +0.7810 |

当前最好提交包：

```text
artifacts/final/submission_best_50.8190_person003.zip
```

对应权重：

```text
artifacts/final/rgb_guided_rdt_yolo11m_1280_ft2_e70_best.pt
```

`artifacts/final/` 下的大文件默认不进入 Git，交接项目文件夹时需要一起保留。

## 目录结构

```text
configs/      主线实验配置
data/         官方原始数据，未纳入 Git
docs/         报告和实验摘要
scripts/      数据检查、训练、验证、推理、提交脚本
src/          项目公共代码
tests/        轻量单元测试
artifacts/    最终权重和最好提交包，未纳入 Git
references/   赛题文档和保留论文资料
```

## 环境

建议使用 Python 3.10 到 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

训练和推理全程离线。公开预训练权重允许使用，但应提前下载到本地，例如 `weights/yolo11m.pt`。

## 数据要求

官方数据目录应包含三模态图像和标签：

```text
RGB / visible
Infrared / infrared
Depth / depth
labels
```

训练标签格式：

```text
class_id norm_center_x norm_center_y norm_w norm_h
```

提交结果格式：

```text
class_id norm_center_x norm_center_y norm_w norm_h confidence
```

每张测试图必须有一个同名 TXT；无检测结果时生成空 TXT。

## 复现流程

检查官方训练数据：

```bash
python scripts/inspect_dataset.py --raw-root data/训练集/AIC2026_Train_2000
```

准备 RGB-guided-RDT 数据：

```powershell
python scripts/prepare_dataset.py `
  --raw-root data/训练集/AIC2026_Train_2000 `
  --out-root datasets/city_rgb_guided_rdt `
  --fusion rgb_guided_rdt
```

训练：

```powershell
python scripts/train_yolo.py `
  --data datasets/city_rgb_guided_rdt/data.yaml `
  --model weights/yolo11m.pt `
  --epochs 100 `
  --imgsz 1280 `
  --batch auto-free `
  --workers 16 `
  --name rgb_guided_rdt_yolo11m
```

验证：

```powershell
python scripts/validate_yolo.py `
  --weights outputs/runs/rgb_guided_rdt_yolo11m/weights/best.pt `
  --data datasets/city_rgb_guided_rdt/data.yaml `
  --imgsz 1280
```

生成普通提交包：

```powershell
python scripts/predict_submit.py `
  --weights artifacts/final/rgb_guided_rdt_yolo11m_1280_ft2_e70_best.pt `
  --raw-root data/测试集/AIC2026_PHASE_1_1000 `
  --fusion rgb_guided_rdt `
  --imgsz 1408 `
  --conf 0.0015 `
  --iou 0.65 `
  --augment `
  --out-dir outputs/submission_txt `
  --zip-path outputs/submission.zip
```

生成当前最好类别阈值提交包：

```powershell
python scripts/predict_class_threshold_submit.py `
  --weights artifacts/final/rgb_guided_rdt_yolo11m_1280_ft2_e70_best.pt `
  --raw-root data/测试集/AIC2026_PHASE_1_1000 `
  --fusion rgb_guided_rdt `
  --imgsz 1408 `
  --base-conf 0.0015 `
  --iou 0.65 `
  --augment `
  --class-conf person:0.003 `
  --out-dir outputs/submission_person003_txt `
  --zip-path outputs/submission_person003.zip
```

校验提交包：

```powershell
python scripts/validate_submission.py `
  --submission outputs/submission_person003.zip `
  --raw-root data/测试集/AIC2026_PHASE_1_1000
```

## 测试

```bash
python -m pytest tests
```

若 Windows 默认临时目录导致 pytest 异常，可临时指定：

```powershell
$env:TEMP="$PWD\tmp"
$env:TMP=$env:TEMP
python -m pytest tests
```

## 报告

- `docs/实验报告_给老师.md`
- `docs/赛题零基础导读.md`
- `docs/实验结果摘要.csv`
