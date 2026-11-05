# V4：候选人体确认

V4 从头训练。保留 VGA、80×60 ambient、最多 8 个传感器 ROI、四角 4 对齐。
ROI 输入提高为等效 96×96 mosaic，即 16×24×24 phase-packed 张量。
最深通道 128；检测分类与回归分支分开，新增候选级 person 分类头。
这是精度优先版本，尚未测 MCU 延迟、内存或 INT8 精度。

## 同步文件

已有的 v3 / hardneg 文件继续保留。新增或更新：

```text
mmdet/datasets/transforms/quad_roi_crop.py
mmdet/models/backbones/quad_roi_backbone_v4.py
mmdet/models/dense_heads/quad_roi_head_v4.py
mmdet/models/detectors/quad_roi_detector_v4.py
mmdet/evaluation/metrics/quad_roi_scene_metric.py
configs/ambient_train/quad_roi_detector_v4.py
configs/ambient_train/quad_roi_detector_v4_site.py
configs/ambient_train/quad_roi_detector_v4_scene_eval.py
configs/ambient_train/quad_roi_v4_README.md
tools/quad_roi_runtime.py
tools/quad_roi_runtime_v4.py
tools/export_quad_roi_v4.py
tools/live_quad_roi_v4.py
tools/mine_quad_roi_negatives.py
```

模型全部位于 mmdet 内，配置通过 custom_imports 注册 V4。
旧的 pack/align 函数增加了可选尺寸参数，默认值保持 v3 行为。

## 1. 用现有 v3 挖掘错误窗口

在仓库根目录运行。下面假设旧 ONNX 在 work_dirs/quad_roi_v3_onnx；替换为实际路径。
需要 numpy、opencv-python、onnxruntime，挖掘不依赖 PyTorch。

```powershell
python tools/mine_quad_roi_negatives.py work_dirs/quad_roi_v3_onnx D:/AI-Server/workspace/datasets/coco/annotations/instances_train2017.json D:/AI-Server/workspace/datasets/coco/train2017 D:/AI-Server/workspace/datasets/coco/annotations/instances_train2017_mined.json --max-images 5000
```

脚本固定随机抽取最多 5000 张训练图片：先 ambient-only，再用最多 8 个预测 ROI 运行高清分支，
收集与任何 person 标注完全不相交的预测窗口。输出新 JSON，所有原图、原标注仍保留。
窗口映射回原图 xyxy，写入 image.hard_negative_boxes，后续与 GT 一起做数据增强。
每张图最多添加 8 个窗口。这里同图两次推理仅用于离线挖掘，不是传感器运行时调度。

只挖训练集，不能用保留的验证场景。COCO 标注也可能漏人，建议抽查挖出的窗口。
中后期可将 model_dir 换为导出的 V4，输出另一个 JSON 后继续训练，以刷新错误窗口。
该脚本不会改写旧 JSON，除非手动把 output 指定为输入文件。

## 2. 从头训练

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v4.py --cfg-options train_dataloader.dataset.ann_file=annotations/instances_train2017_mined.json
```

默认 80 epoch、batch 12、AdamW 1e-3、500 iteration warmup、余弦下降至 1e-5。
不要 load_from v3，也不要首次使用 --resume。V4 的结构和部分张量尺寸已经改变。
输出 work_dirs/quad_roi_detector_v4，每轮验证，按 COCO AP50 保存最佳权重。
COCO 验证的 ROI 通过率仍为 100/80/30，保留原来的随机 ROI 和高度分组。

中断后恢复必须保留相同数据配置：

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v4.py --resume --cfg-options train_dataloader.dataset.ann_file=annotations/instances_train2017_mined.json
```

若暂时不挖掘，去掉 --cfg-options 可直接使用原 COCO；仍有物体标注引导的困难负 ROI，
但缺少上一版实际误报窗口，不是推荐的完整数据流程。

确认头前 2000 iteration 使用 GT、轻微扰动 GT、ROI 内随机窗口及完整 ROI 启动。
之后加入当前模型的 ambient/HD/fusion 预测候选，坐标不反传梯度，特征和确认分支正常训练。
每个 ROI 最多采 8 个正窗口与 8 个负窗口；BCE 分别按正负样本数量归一化。
负窗口中最多 4 个优先取模型中心分数≥0.15的高响应误报，其余随机采样，避免只学容易背景。
可见窗口与合格 GT 可见部分 IoU≥0.5 才为正，与所有人体/ignore 框完全不相交才为负。
其余窗口忽略；半身可为正，多人 ROI 中逐候选判断，不使用整 ROI 的“有人”标签确认所有框。
日志中的 loss_verify、verify_positive、verify_negative 用于观察分类学习和有效样本数。
verify_hard_negative 记录本批优先采入的在线误报窗口数（预热期为零）。
采样数是训练窗口数，不是传感器 ROI 数。

分类质量辅助损失 loss_quality_bg 只在有效区域、所有 GT/ignore 框以外的高响应位置计算，
每个特征图最多 16 个位置，权重 0.1。保留原有检测和定位损失。
困难物体 ROI 最多保留两个；原有无 ROI 概率、随机负 ROI、正 ROI 部分截断仍保留。
合并几何规则继续沿用 v3 的保守标准。

## 3. 可选：COCO + 现场数据

现场格式沿用 quad_roi_hardneg_README.md：原始图片、所有真人标注、人工确认的 hard_negative_boxes。
准备 data/quad_roi_site/train.json 与 images/，不要把未人工审查的误报直接当真负标签。
配置 quad_roi_detector_v4_site.py 中可以修改路径。

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v4_site.py --cfg-options train_dataloader.dataset.datasets.0.ann_file=annotations/instances_train2017_mined.json
```

混合配置使用 iteration 计划：80000 iteration，每 2000 iteration 验证；不是 80 个完整 COCO epoch。
每 batch 9 张 COCO、3 张现场。训练预算不同，可按实际曲线在配置中调整 max_iters。
若先训练了基础 V4，再切换现场配置，可用 --cfg-options load_from="实际V4权重.pth"
optim_wrapper.optimizer.lr=0.0001 开始新一轮较低学习率训练，不带 --resume。
中断同一轮再加 --resume，并保留相同的数据路径参数。

## 4. 导出

```powershell
python -m pip install onnx
python tools/export_quad_roi_v4.py work_dirs/quad_roi_detector_v4/你的checkpoint.pth --out-dir work_dirs/quad_roi_v4_onnx
```

使用现场配置训练时，可额外传 --config configs/ambient_train/quad_roi_detector_v4_site.py。
四个模型固定 batch=1、FP32、opset 13，没有做 INT8 量化。

| 模型 | 输入 NCHW | 输出 |
| --- | --- | --- |
| ambient | 1×1×60×80 | context 1×64×32×40；fine 1×6×32×40；coarse 1×6×16×20 |
| roi | 1×16×24×24 | hd 1×128×12×12；prediction 1×6×12×12；shallow 1×32×24×24 |
| fusion | hd 1×128×12×12；context 1×64×12×12；valid 1×1×12×12；scale 1×2×12×12 | 1×8×12×12 |
| verifier | crop 1×32×8×8 | 一个 logit |

CPU 按候选框与 ROI 的交集，在浅层特征上双线性采样 8×8，verifier 输出 sigmoid 后的 P。
只使用已经请求到的 ROI 数据。多个候选共享一次 ROI 特征提取，但分别调用小型 verifier。
选覆盖最多且可用的 ROI；可见面积至少占候选 25%，投影后至少宽 4、高 12 像素才验证。
未验证的 P=-1，不能解读为判负。最多处理 pre_verify_topk=128 个候选，之后 NMS 保留最多 64。
这不改变传感器的 8 ROI 上限，但确认头调用数可以超过 8；速度需要实际测量。

## 5. 摄像头与视频复查

复制四个 ONNX、quad_roi.json 和相关 Python 文件到运行电脑。运行端不需要 PyTorch。
live_v4 复用 live_v3 的几何合并/外观记忆函数，runtime_v4 复用 runtime 的采样/解码函数，
因此两个旧脚本也要一起保留。

```powershell
python tools/live_quad_roi_v4.py work_dirs/quad_roi_v4_onnx --camera 0
python tools/live_quad_roi_v4.py work_dirs/quad_roi_v4_onnx --video room_holdout.mp4 --log v4_holdout.jsonl
```

Q/Esc 退出。日志记录每帧检测、P、覆盖率、轨迹 ID/确认状态及实际 ROI，
可以用来统计错误 confirmed 事件、持续时间、真人确认延迟和 ROI 用量；它不是自动真值标注。
视频输入按视频帧时间计算位置记忆超时；摄像头使用实际时间。

初始阈值（尚未用实拍校准）：

- --confidence 0.25：新候选的中心分数 C 门槛；--track-confidence 0.10 用于已有轨迹关联。
- --person-high 0.70：连续两次几何一致的高 P 才确认。
- --person-low 0.20：覆盖≥80%时连续两次低 P 才进入位置记忆，不以单次未确认充当负证据。
- P 在中间或覆盖不足：不确定，复查间隔逐步增加至最多 15 帧，不写入背景记忆。
- 已确认近处目标，ambient 自己连续 3 帧 C≥--ambient-high 0.30 且 Q≥0.2 才关闭 ROI；
  连续 2 帧 C<--ambient-low 0.20 或 Q<0.15 会重新请求。融合高分不能代替 ambient 的可靠观测。
- 远近分界 130，高度在 120～140 之间保持状态；稳定远处目标每 5 帧申请一次，移动时每帧申请。
- 明确负证据优先触发重查；暂时漏检仍使用 alpha-beta，连续 5 帧无合格观测删除轨迹。
- 位置记忆默认 30 秒，ambient 外观明显变化提前解除；不按“静止”判背景。

屏幕 C=中心分数、Q=框质量、P=候选分类分数；age 是距最近匹配观测的帧数。
confirmed 是历史确认状态；每帧 P 可能不可用。所有参数都需要结合保留视频的真人召回调整。

Python API：将 tools 加入模块搜索路径后导入 QuadROIV4Onnx。
detect(rgb_vga, roi_list, confidence_threshold) 返回 bbox 和 scores；
predict(...) 额外返回 center_score、quality_score、person_score、verify_coverage、roi_id、source、ambient_center/quality。
调用方保证 RGB uint8 VGA、合法四角 4 对齐、最多 8 个 ROI，函数不检查输入 ROI 数量。
候选按 C 过滤；scores=C×P（已验证）或 C（未验证），NMS 按 scores×Q 排序。
detect 再按 scores 过滤；live 使用 predict 的独立 C/P 决策，不把乘积当成确认概率。
单帧 detect 返回的结果不是跟踪器的 confirmed 事件。

## 6. 验证与模型选择

原 COCO AP50 继续报告，但还应看场景召回和误报。准备独立的 data/quad_roi_site/val.json：

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v4_scene_eval.py work_dirs/quad_roi_detector_v4/你的checkpoint.pth
```

输出不同分数阈值下的 recall50、small_recall50、fp_per_image；verified 前缀另外要求 P≥0.7。
这是使用标签/物体引导 ROI 的静态诊断，不代表真实闭环跟踪召回。
没有可用高清的 ambient 检测不计入 verified，不能据此认定它是漏检。
完整系统效果请使用上面的保留视频和 JSONL，比较真人召回、误确认次数/时长、确认延迟和 ROI 用量。
没有自动承诺某个阈值或 epoch 能消除椅子误报。

本次只编写与静态审阅代码，没有执行训练、导出、摄像头推理或任何测试。
