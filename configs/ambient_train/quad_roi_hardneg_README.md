# v3 困难负 ROI 微调

网络、损失、ONNX 接口和推理计算量不变。新增 dataset 只读取辅助框来选 ROI，
输出类别仍然只有 person。原始 CocoDataset 和通用 COCO metric 没有修改。

同步到训练电脑的文件：

```text
mmdet/datasets/__init__.py
mmdet/datasets/quad_roi_coco.py
mmdet/datasets/transforms/quad_roi_crop.py
configs/ambient_train/quad_roi_detector_v3_hardneg.py
configs/ambient_train/quad_roi_detector_v3_hardneg_site.py
configs/ambient_train/quad_roi_hardneg_README.md
```

## 先用现有 COCO 继续训练

在仓库根目录运行，将 checkpoint 替换为实际的 v3 权重路径：

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v3_hardneg.py --cfg-options load_from="C:/OculiProjects/mmdetection/work_dirs/quad_roi_detector_v3/你的checkpoint.pth"
```

首次启动不要加 `--resume`：这是加载 v3 权重开始新一轮微调，重新建立优化器和学习率计划。
默认训练 10 个 epoch，AdamW 学习率 1e-4，200 iteration warmup，余弦下降至 1e-5。
输出目录是 `work_dirs/quad_roi_detector_v3_hardneg`，不覆盖原 v3。
中断后继续这一轮，才使用：

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v3_hardneg.py --resume
```

本轮验证仍使用原 v3 的 COCO、ROI 通过率和指标，保证与原配置可比。
默认最佳权重仍按 AP50 保存；这不保证它是实际场景误报最少的一份权重。

## 训练变化

- `QuadROICocoDataset` 额外读取 chair、couch、potted plant、vase、backpack、
  suitcase、teddy bear 的标注框；`negative_classes` 可配置。
- 辅助框与人体框一起做 crop、VGA resize、水平翻转，然后按 VGA 面积筛选。
  人体标签仍采用原先的面积至少 64、边长至少 2 的规则。
- `hard_negative_slots=2`：有合适困难负 ROI 时预留最多两个名额，其余名额按原规则分配；
  总量始终不超过 8。没有合适困难负 ROI 时，名额回到普通候选。
- 每张图最多随机选 8 个不与保留人体框相交的困难候选。
  困难候选之间沿用原来的合并几何条件，再按合并优先、小面积优先选出最多两个。
  它们单独成组，不与正候选混合，以保留真正的负 ROI。
- 困难 ROI 保留独立宽高缩放、平移、四角对齐到 4；不做额外的单边截断。
  合并和抖动后再次检查与人体（含 crowd/ignore）的相交；不合适就跳过，
  不把困难窗口搬到另一块容易背景。与人重叠的椅子框不会被强制当负样本。
- 普通随机负 ROI、正 ROI 保留概率、正 ROI 的部分截断仍然保留。
  原本 15% 的全帧无 ROI 概率也保留，此时不预留困难负 ROI，继续训练 ambient 独立检测。
- 三个检测分支仍按真实人体标签计算原有损失；没有为物体新增检测类别，
  没有修改质量分支的监督。

原配置中 `hard_negative_slots` 默认为 0，不启用新采样策略。
COCO 没有专门的挂衣服类别；要针对这类误报，需要下面的现场数据。

## 可选：加入现场原图及人工确认的误报框

准备 `data/quad_roi_site/images/` 和 COCO 格式的 `data/quad_roi_site/train.json`。
图片必须是未叠加检测框/文字的原始画面，尺寸不要求已经是 VGA。
每个 image 项可增加 `hard_negative_boxes`，坐标为**原图 xyxy**。
这是已人工确认的非人体区域，不是未经审查的模型输出。

一个纯负样本标注文件示例：

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "empty_room_001.jpg",
      "width": 640,
      "height": 480,
      "hard_negative_boxes": [[430, 180, 600, 470]]
    }
  ],
  "annotations": [],
  "categories": [{"id": 1, "name": "person"}]
}
```

如果画面有人，必须在 `annotations` 中标出所有真实人体，采用标准 COCO xywh、
area、iscrowd、category_id、image_id 和唯一 id。不能把“误报框是假的”等同于整张图无人。
没有显式误报框的图片仍可参与常规训练，但不会凭空生成物体引导的困难框。
已有 COCO 非人类别标注也可以通过 `negative_classes` 自动提供辅助框。

保留多场景、多角度及不同距离的真人正样本，特别是坐姿、遮挡和半身，
避免只教会模型排除整个房间。训练与验证按场景或拍摄片段划分，避免相邻帧分到两侧。

准备好数据后，使用独立的混合配置：

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v3_hardneg_site.py --cfg-options load_from="C:/OculiProjects/mmdetection/work_dirs/quad_roi_detector_v3/你的checkpoint.pth"
```

也可以加载上一轮 hardneg 的权重。现场路径在此配置的 `site_dataset.update(...)` 中修改。
每个 batch 默认 12 张，其中 9 张 COCO、3 张现场图片；现场数据会循环采样。
配置使用 IterBasedTrainLoop 配合 MultiSourceSampler，默认 15000 iteration，
每 1000 iteration 验证并保存，输出至 `work_dirs/quad_roi_detector_v3_hardneg_site`。
若改 batch size，需要同时修改 sampler.batch_size。
中断续训同样在相应配置后加 `--resume`。

混合配置的自动验证也仍是 COCO；现场视频中的误报、误确认持续时间和人体召回
需用单独保留的场景对比。当前没有自动采集、自动标注或自动运行现场评估。

微调完成后，用新的 checkpoint 再运行原 ONNX 导出脚本即可。
本次仅编写和静态审阅代码，没有运行训练、测试或导出。
