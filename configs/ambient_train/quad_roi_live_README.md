# v3 ONNX 导出与摄像头验证

同步 `tools/export_quad_roi_v3.py`、`tools/quad_roi_runtime.py`、
`tools/live_quad_roi_v3.py`。导出在已有 MMDetection 训练环境中运行；
摄像头运行端只需要后两个文件、模型目录和 NumPy、OpenCV、ONNX Runtime。

## 导出

在仓库根目录运行，将 checkpoint 路径替换为实际文件：

```powershell
python -m pip install onnx
python tools/export_quad_roi_v3.py work_dirs/quad_roi_detector_v3/epoch_3.pth --out-dir work_dirs/quad_roi_v3_onnx
```

`--config` 默认 `configs/ambient_train/quad_roi_detector_v3.py`。
脚本从指定 checkpoint 加载权重，使用 CPU、eval 模式和固定 batch=1 导出。
三个模型都是 FP32、opset 13；没有做 INT8 校准或 MCU 转换。

所有张量采用 NCHW：

| 文件 | 输入 | 输出 |
| --- | --- | --- |
| `ambient.onnx` | `ambient`: 1×1×60×80 | `context`: 1×64×32×40；`fine`: 1×6×32×40；`coarse`: 1×6×16×20 |
| `roi.onnx` | `roi`: 1×16×16×16 | `hd`: 1×64×8×8；`prediction`: 1×6×8×8 |
| `fusion.onnx` | `hd`、`context`: 各 1×64×8×8；`valid`: 1×1×8×8；`scale`: 1×2×8×8 | `prediction`: 1×8×8×8 |

表中通道数对应默认 v3 配置。`scale` 是 ROI 宽/640、高/480 在 8×8 上的广播，
由 CPU 准备；融合图包含 64→32 的上下文投影。每帧先运行一次 ambient，
然后逐个运行 ROI 和 fusion，共享该帧的 ambient 特征。
RGB/CFA 采样、ROI 对齐、解码、证据匹配和 NMS 在模型图外完成。
这三个边界可用于后续 MCU 分阶段推理；实际 INT8 算子支持、内存和延迟需后续转换确认。

导出同时生成 `quad_roi.json`，保存后处理参数与训练使用的合并参数。
将三个 ONNX 和此 JSON 作为同一套模型文件复制到运行电脑。

## 摄像头

```powershell
python -m pip install numpy opencv-python onnxruntime
python tools/live_quad_roi_v3.py work_dirs/quad_roi_v3_onnx --camera 0 --confidence 0.25 --hd-confidence 0.25
```

按 Q 或 Esc 退出。默认 CPUExecutionProvider、单线程；可用 `--threads` 调整。
摄像头请求 640×480，实际获取的画面先居中裁成 4:3 再 resize 到 VGA，随后 BGR→RGB。
显示坐标、检测坐标、ROI 坐标均对应这张 VGA 画面。

- 第 1 帧没有 ROI。第 N 帧检测产生的轨迹，只安排第 N+1 帧的 ROI。
- 新目标先进入 pending；`hd_support >= --hd-confidence` 后进入 confirmed。
  该分数来自独立 HD 分支与最终人体框可见区域的匹配。
- pending 目标每帧申请 ROI；confirmed 目标高度 ≤`--roi-height`（默认 200）时申请 ROI。
  超过此高度后依靠 ambient；只要当前帧漏检，下一帧重新申请 ROI。
- 每次匹配检测都重置漏检计数，连续第 5 帧漏检时删除；pending 和 confirmed 均如此。
- alpha-beta 状态为中心 x/y、宽/高，各自带速度；时间单位是处理帧，dt=1。
  默认 `--alpha 0.7 --beta 0.1`，以预测框与检测框的 IoU 做一对一贪心匹配，
  默认 `--match-iou 0.2`。
- 下一帧预测框按照导出的训练规则合并：尺寸比、间距、面积、最小投影高度、中心距离、
  最多 3 个成员。随后确定性外扩 `--roi-scale 1.15`，四角对齐到 4 的倍数。
  合并过的组优先，再按最终面积从小到大，最多取 8 个；申请 ROI 不保证一定取得名额。
  实时流程不施加训练用的随机抖动、随机负 ROI 或 GT 保留概率。

蓝色细框是本帧实际使用的 ROI；黄色轨迹为 pending，绿色为 confirmed，橙色为漏检预测。
confirmed 表示曾得到 HD 确认，不表示本帧一定有 HD。
画面上的 score 是该轨迹最近一次匹配检测的分数；漏检时显示的是预测位置。
FPS 是当前帧预处理、推理、跟踪和绘制的处理速度，不包含摄像头等待和窗口显示时间。
只靠 ambient 启动的流程，需要 ambient 先发现目标，才能在后续帧为它安排高清 ROI。

## 在其他 Python 程序中调用

从仓库根目录导入，或将 `quad_roi_runtime.py` 放在调用程序的模块路径中。

```python
from tools.quad_roi_runtime import QuadROIOnnx

detector = QuadROIOnnx("work_dirs/quad_roi_v3_onnx")
boxes, confidence = detector.detect(rgb_vga, roi_list, confidence_threshold=0.25)
```

`rgb_vga` 为 RGB 顺序、uint8、480×640×3。`roi_list` 为 VGA 中有效的
`[x1, y1, x2, y2]` 列表，右下角不包含在 crop 内；调用方保证四角是 4 的倍数、
坐标在图内且最多 8 个，空列表允许。函数不检查或截断外部 ROI 数量。
返回 `boxes` 形状 N×4、`confidence` 形状 N，空检测为 0×4 和 0。
模型 session 在构造对象时加载一次，逐帧调用 `detect` 即可。

需要证据字段时调用 `detector.predict(rgb_vga, roi_list, threshold)`，返回字典：
`bboxes`、`scores`、`source`、`ambient_score`、`hd_support`、`hd_coverage`、
`hd_available`、`roi_id`。`roi_id` 对应本次传入列表的下标，`source` 的 0/1
对应 ambient/融合分支。摄像头主流程每帧只调用一次这个详细接口。

预处理使用绿色 `rgb[::8, 2::8, 1]` 和逐 CFA phase resize，输入除以 255。
ROI 上下文采用修正后的 `(j+0.25)/8` 和 ambient 原点 `(2.5, 0.5)`，
以 NumPy 双线性插值复现 border padding；ambient 检测屏蔽底部 padding 行。
检测头沿用 heatmap×quality、3×3 局部极大值、top-k、坐标还原与统一 NMS。
浮点运算和同分候选的排序可能与 PyTorch 有微小差异。

脚本已编写并静态审阅；尚未执行导出、摄像头推理或测试。
