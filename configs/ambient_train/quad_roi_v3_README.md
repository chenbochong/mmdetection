# Quad ROI v3

在训练电脑同步这些文件；保留原有 `quad_roi_detector.py` 基础配置和其他 Quad ROI 组件：

- `configs/ambient_train/quad_roi_detector_v3.py`
- `mmdet/models/backbones/quad_roi_backbone.py`、该目录的 `__init__.py`
- `mmdet/models/necks/quad_roi_fusion.py`
- `mmdet/models/dense_heads/quad_roi_head.py`
- `mmdet/models/detectors/quad_roi_detector.py`、`quad_roi_detector_v3.py`、该目录的 `__init__.py`
- `mmdet/datasets/transforms/quad_roi_crop.py`
- `mmdet/evaluation/metrics/quad_roi_coco_metric.py`

## 训练

在训练电脑的仓库根目录运行，第一次不要加 `--resume`：

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v3.py
```

默认从头训练，继承基础配置的 120 epoch、AdamW lr=0.001、500 iteration warmup 和 cosine schedule。
训练 batch size=12，数据路径继承基础配置；输出目录为 `work_dirs/quad_roi_detector_v3`。
最佳 checkpoint 按 `coco/bbox_mAP_50` 保存。中断后继续同一次 v3 训练：

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v3.py --resume
```

## 网络与监督

特征尺寸按宽 × 高 × 通道列出：

| 分支 | 特征与用途 |
| --- | --- |
| Ambient | 输入 80×60，底部补 4 行；A2/A3/A4 为 40×32×64、20×16×64、10×8×64 |
| Ambient FPN | 最近邻上采样 + 相加 + 深度可分离卷积，生成 P3、P2 |
| Ambient 检测 | P2 监督所有保留的 GT，P3 额外监督高度 ≥130 的 GT；两个头均先 64→32 再输出 6 通道 |
| ROI encoder | 保留 16×16×16 Quad Bayer packed 输入与 8×8×64 输出 |
| ROI 融合 | 从 P2 对齐 8×8 上下文，64→32；融合 HD、上下文、有效区域和 ROI 尺寸 |
| 融合检测头 | 8 通道：heatmap、LTRB 四边距离、二维 offset、完整框 IoU quality |
| HD 独立头 | 6 通道，监督 ROI 内可见部分的框；用于提供高清支持分数 |

Ambient 的 stride 相对于输入是 2/4/8，相对于 VGA 是 16/32/64。
补齐行不参与检测监督和候选输出；坐标最终仍对应原来的 640×480。
融合分支使用 GT 与 ROI 交集的中心作为预测点，回归完整 GT 的四条边。
可见面积达到 GT 的 25% 即可参与监督，不再要求完整 GT 中心位于 ROI 内。
融合框可以超出 ROI，只在最终输出时裁到 VGA 边界。HD 独立头的 quality 对应可见框。

训练保留原有随机平移、独立宽高缩放、ROI dropout、随机背景候选和合并规则；
另外对约 30% 的 jitter 后 ROI 随机保留上、下、左或右一侧，沿该轴保留 30%–60%。
`partial_prob`、`partial_fraction` 可配置；GT 完整框不随 ROI 截断，ROI 四角最后对齐到 4 的倍数。
常规验证不额外施加这种截断。

训练与验证共用 `(40, 130)` VGA 高度分档和 `min_gt_area=64`。
训练小、中、大 GT 候选保留概率仍为 `(0.85, 0.65, 0.3)`，验证改为 `(1.0, 0.8, 0.3)`；
这些是进入候选的概率，最终仍受 dropout（仅训练）、合并和 8 个名额限制。
合并后优先保合并过的 ROI，再按面积从小到大，最多保留 8 个。

训练和验证开启 `matched_negatives=True`，把原有随机候选改为背景负 ROI：
从本图已选的 GT 候选中抽取宽高模板；没有候选时从本图保留的 GT 中取，
没有普通 GT 时才使用原有随机宽高范围。保持模板宽高，将窗口移到与所有保留的人体
标注（包括 ignore/crowd）均无交集的位置；验证时还排除 VGA letterbox 的黑边。
四角仍对齐到 4 的倍数。负候选与其他候选一起合并、抖动和排序；
纯负候选组成的组在最终抖动后再次检查，有交集或越出有效图像时按最终宽高重新找背景。
负候选与正候选混合成组时，正常按组内的真实 GT 监督，不强行当作负样本。

候选数量仍为训练 0–3、验证 0–2；每个背景窗口最多尝试 `negative_attempts=32` 次，
找不到合适背景就跳过，不缩小模板。负 ROI 不预留名额，因此拥挤图中仍可能被截断。
面积小于 64 而已删除的标签仍按此前约定视为背景。负样本沿用检测头的背景分类损失，
不修改网络或 GT，也不增加推理分支。

本次数据策略调整只需同步 transform 和 v3 配置，重启训练进程后可用 `--resume`
继续现有 v3 checkpoint。验证 ROI 策略已经改变，新旧 AP 不能直接归因于模型提升。

## 输出与指标

输出保留 `bboxes`、`scores`、`labels`，以及：

- `source`：最终框来自 ambient（0）还是融合头（1）。
- `ambient_score`：与最终框匹配的独立 ambient 检测分数。
- `hd_support`：匹配的 HD 独立头分数；对比完整框在对应 ROI 内的可见区域，允许半身支持全身框。
- `hd_available`：是否有 ROI 与最终框相交。
- `hd_coverage`、`roi_id`：支持分数所对应 ROI 的覆盖比例和槽位；没有匹配的支持时选择覆盖最多的 ROI，无覆盖时槽位为 -1。

`hd_support` 来自可训练的 HD 检测分支与几何匹配，不表示去掉 HD 后检测一定消失。
匹配阈值为可见区域 IoU≥0.5、HD 框位于可见区域内的比例≥0.8、覆盖完整框面积≥0.25。

日志新增 `coco/bbox_mAP_50_s`、`_m`、`_l` 和 `coco/bbox_AR_50`、`_s`、`_m`、`_l`。
AR50 沿用 COCO 的匹配与数量上限，固定 IoU=0.5，不是某个部署 score 阈值下的召回率。
原有 `bbox_mAP_l` 等指标仍平均 IoU=0.50:0.95。只修改自定义 metric。

需要区分 ambient 能力和部分 ROI 情况时，可自行对同一个 checkpoint 运行以下命令。
把 `PATH_TO_V3_CHECKPOINT.pth` 换成实际路径；不必每轮都运行这两种诊断。

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v3.py PATH_TO_V3_CHECKPOINT.pth --work-dir work_dirs/quad_roi_v3_ambient --cfg-options test_dataloader.dataset.pipeline.2.pseudo_rois=False
python tools/test.py configs/ambient_train/quad_roi_detector_v3.py PATH_TO_V3_CHECKPOINT.pth --work-dir work_dirs/quad_roi_v3_partial --cfg-options test_dataloader.dataset.pipeline.2.partial_prob=1.0 test_dataloader.dataset.pipeline.2.partial_fraction="(0.3,0.6)"
```

第二种诊断使用固定逐图种子；指标仍对完整 GT 框计算，ambient 也参与输出，
因此它衡量部分 ROI 条件下的整套系统表现。它不改变原有的大目标候选策略。

静态卷积算量约为 ambient 19.64M MAC，每个 ROI 7.10M MAC；
ambient +3 ROI 约 40.94M MAC，+8 ROI 约 76.43M MAC。
最大单张特征为 40×32×64，INT8 下 80KiB，不等于运行时总内存。
部署时逐个处理 ROI 并复用内存；实际延迟和内存需后续由板端转换结果确认。

此次仅编写代码并做静态审阅，未运行模型、训练或测试。
