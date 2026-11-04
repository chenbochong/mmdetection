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

现有 ONNX 可以继续使用，无需重新训练或导出。以下阈值是初始调度参数，尚未用实际视频调优。

### 发现与确认

第 1 帧没有 ROI。第 N 帧的结果只决定第 N+1 帧 ROI。
`--confidence 0.25` 是新候选门槛；新目标先进入 pending，并申请高清检查。
最终检测分数和 `hd_support` 分别通过 confidence 和 `--hd-confidence 0.25`，
才算一次高清确认。默认 `--confirm-hits 2`：需要不同帧的两次支持，后一次检测与
轨迹预测框 IoU ≥0.5；几何不一致时从一次重新计数。未安排 ROI 的帧不重置计数，
充分覆盖的 ROI 检查失败会重置。pending 是候选，不应作为已确认人体事件。

新候选最多进行 `--confirm-attempts 3` 次有效高清检查，仍未获得两次一致的支持便移入位置记忆。
这里的“失败”只是没有达到确认条件，不代表模型证明该区域没人。
阳性支持会计一次检查；阴性结果只有实际 ROI 覆盖目标框 ≥80%、
且按 ROI 长边缩放到 64 后目标高度 ≥24，才计一次失败。
没有取得 ROI 名额、ROI 没框准或合并后目标太小，都不算失败。

位置记忆保存该框内 ambient 的 16×16 外观摘要，不读取额外高清像素。
新候选与记忆框 IoU ≥0.4 且外观未改变时，暂不建立轨迹、也不再为它申请 ROI。
外观变化用相对保存基准的平均绝对差衡量，扣除整体亮度偏移；
达到 `--memory-change 0.04`（约 10/255）便清除该条记忆。
即使画面没有变化，`--memory-seconds 30` 秒后也解除抑制，允许仍被 ambient 检出的候选重查。
记忆最多保留最近 64 个区域；在其他目标的 ROI 中意外获得合格 HD 支持，也可以提前解除。
这是暂缓检查机制，不是永久背景标签；外观变化检测不可能保证发现每个新进入的人。

### 已确认目标的 ROI 调度

| 情况 | 下一帧申请 |
| --- | --- |
| pending | ROI，直到确认或进入位置记忆 |
| confirmed、近处、检测正常 | 不申请 ROI，用 ambient 更新轨迹 |
| confirmed、远处、移动或框不稳定 | 每帧申请 ROI |
| confirmed、远处、连续稳定且检测正常 | 默认每 5 帧一次 ROI |
| 任意轨迹漏检，或只有低分检测 | 申请 ROI 恢复可靠观测 |

远近初始分界为 `--roi-height 130` 像素，沿用训练中的约 3m 分档。
这是人体像素高度代理，不是真实测距；普通摄像头视角与 120° 传感器不同，需要调整此参数。
`--height-hysteresis 10` 使已有轨迹低于 120 才转远处、高于 140 才转近处，区间内保持原状态。
稳定要求连续 3 帧有合格检测、预测误差小于宽高的 8%、状态速度小于宽高的 4%/帧。
`--stable-roi-interval 5` 控制稳定远处目标的高清间隔；设为 1 可恢复每帧申请。
即使目标静止，若 ambient 无法可靠检测到它，也会继续申请 ROI。

跟踪使用 alpha-beta，状态为中心 x/y、宽/高，各自带速度；时间单位为处理帧，dt=1。
默认 `--alpha 0.7 --beta 0.1 --match-iou 0.2`，先匹配高分检测，再匹配剩余低分检测。
`--track-confidence 0.10` 允许弱检测以一半增益修正已有轨迹，但不能建立新轨迹、
不能完成高清确认，也不重置可靠观测缺失计数。默认 `--max-misses 5` 帧没有合格检测便删除轨迹。
已确认轨迹如果连续 3 次充分覆盖的高清检查失败，会降回 pending，重新确认；
重新给予 3 次检查额度，仍不能确认才进入位置记忆；期间也可能因缺失观测而删除。
未执行 ROI 的帧不会被当成高清检查失败。

下一帧预测框按照导出的训练规则合并：尺寸比、间距、面积、最小投影高度、中心距离、
最多 3 个成员。随后确定性外扩 `--roi-scale 1.15`，四角对齐到 4 的倍数。
合并过的组优先，再按最终面积从小到大，最多取 8 个；申请不保证取得名额。
实时流程不施加训练用的随机抖动、随机负 ROI 或 GT 保留概率。

### 误检与画面状态

确认要求最终人体检测与独立 HD 支持同时达标，并有跨帧几何一致性，
用于减少瞬时误检和证据矛盾；没有使用“静止就是假人”的规则。
两分支共享信息，多帧结果也相关，不能把它们当作统计独立的证据相乘。
如果衣服、人形物体持续让这些输出都达到条件，现有输出仍可能误确认，
此调度不能替代针对这类负样本的模型训练。

蓝色细框是本帧实际 ROI；黄色轨迹为 pending，绿色为 confirmed，橙色为 coast。
confirmed 表示通过高清确认且尚未被撤销，不表示本帧一定有 HD。
score 是最近匹配的检测分数，coast 时可能是弱检测分数或之前的分数。
memory 显示当前暂缓区域数；FPS 包含预处理、推理、调度、跟踪和框绘制，
不包含摄像头等待和窗口显示时间。
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
