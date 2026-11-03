# Quad ROI person detector

Training entry: `configs/ambient_train/quad_roi_detector.py`, through the existing
`tools/train.py`. Set `data_root` in that config. Components are registered directly under mmdet. No previous model implementations or
training configs are changed.

## Data and network

- Training: random 4:3 crop, VGA resize, optional horizontal flip, then CFA sampling.
- Ambient: `image[::8, 2::8, 1]`, 1x60x80, normalized to [0, 1].
- CFA: RGGB Quad Bayer, red top left, blue bottom right. The 16 channels are
  **row-major 4x4 phase order**, unlike the old transform's grouped-color order.
- ROIs: jittered scale and position, outward alignment to multiples of 4,
  size-dependent sampling, random regions, label-independent layouts, and
  whole-frame ROI dropout. When over budget, keep the smallest eight regions.
- Each ROI is phase-packed before resize, aspect-preserving letterboxed to
  16x16x16 (the 64x64 mosaic budget). Rounded resize scales and padding are stored
  explicitly in `roi_affine`. Padded ROI slots never enter the encoder.
- Ambient encoder: 30x40x32 features with 15x20x64 context. ROI encoder: 8x8x64.
  Ambient uses a separable second stem layer. ROI downsampling happens in its
  first convolution, avoiding a 16x16x64 activation while preserving 8x8 output.
  The fusion neck combines aligned ambient features, ROI features, a valid-region
  mask, and ROI width/height. All three detection heads use center heatmaps,
  width/height, offsets, and learned IoU quality.
- Ambient targets include all people. Local heads include all people with centers
  inside the ROI and at least 60% box coverage, including multiple people per ROI.
  Partially observed excluded people and COCO crowd regions are ignored, not
  treated as background. Local regression targets are full person boxes.
- Local boxes map directly back to VGA. Ambient and fused detections share NMS;
  the HD-only head supplies independently supervised evidence.

Validation uses fixed per-image random seeds and label-derived pseudo ROIs. Its
COCO AP measures detection with simulated ROIs, not motion/tracker proposal recall.
Set `pseudo_rois=False` in the validation transform for ambient-only evaluation.

For annotation-free inference, use `inference_pipeline` explicitly; the default
validation pipeline requires annotations. Optionally provide `roi_boxes` in the
transform input, in VGA sensor coordinates. Without them, this pipeline uses no
ROIs. A real RAW acquisition frontend should supply the same preprocessed tensor
dictionary rather than resample a RAW image as BGR.

## Output contract

`pred_instances` contains:

| Field | Meaning |
| --- | --- |
| `bboxes`, `scores`, `labels` | Standard MMDetection detections; score = person heatmap probability x learned box quality |
| `source` | 0: selected ambient candidate; 1: selected fused candidate |
| `ambient_score` | Best independent ambient score matched to this box at `support_iou` |
| `hd_support` | Best independent HD person x quality score matched at `support_iou` |
| `hd_available` | At least one acquired ROI intersects the detection |
| `hd_coverage` | Largest fraction of the box covered by a single acquired ROI |
| `roi_id` | ROI slot supplying HD evidence, or best-coverage slot if evidence is absent; -1 without overlap |

`hd_support` is learned from HD pixels without ambient input; its classification
target is person presence and its quality target is detached predicted-box IoU.
It is **independent corroborating evidence**, not a causal attribution or an
estimated gain. `source` describes the selected decoder, not whether HD is needed.
Missing observation (`hd_available=False`) and weak evidence despite observation
remain distinguishable. Evidence matching is geometric and can be ambiguous in
crowds. Ambient/fused union NMS does not implement HD veto of ambient false positives.

Coordinates are VGA with `predict(..., rescale=False)`; normal MMDetection
prediction restores original-image coordinates for COCO evaluation. ROI slot IDs
refer to the area-sorted, budgeted `inputs['roi_boxes']`. Coverage is measured in
VGA before coordinate restoration.

## MCU boundary

Convolutional paths use Conv, depthwise Conv, BN (foldable), ReLU, add, concat,
and nearest upsampling. CFA resize, ambient feature alignment (`grid_sample` in
training), heatmap decoding, evidence matching, and NMS are host/MCU operations,
not intended NPU graph operators. Sigmoid and box decode are outside the head's
raw convolution output. The ambient pass runs once; ROI encoding/fusion can run
one slot at a time after its fixed context tensor is prepared.

## Ambient + three ROI budget

The 100ms configuration uses 32 ambient feature channels, 64 ambient context
channels, and 64 ROI/fusion channels. The ambient detection head takes 32 channels;
the HD and fused heads take 64. Increasing width mainly at small spatial sizes
preserves capacity without keeping a 64-channel global feature map in SRAM.

Static counts include all convolutional detection/evidence heads and assume
folded BN. They exclude activation, interpolation, memory copies, decoding and
NMS. One MAC counts as two operations.

| Component | Previous 50ms configuration MAC | Current MAC |
| --- | ---: | ---: |
| Ambient encoder + head, once/frame | 3,232,800 | 8,284,800 |
| ROI encoder, once/ROI | 986,112 | 5,906,432 |
| Fusion neck + HD/fused heads, once/ROI | 198,144 | 1,052,672 |
| Ambient + 3 ROIs | 6,785,568 | 29,162,112 |

Current total: `8,284,800 + N * 6,959,104` MAC for N ROIs. Eight ROI slots remain
supported; the 100ms budget applies to three active ROIs, not to eight. At eight,
the convolution count is 63,957,632 MAC. Retrain this architecture; channel widths
and parameter shapes are incompatible with the earlier checkpoint.

For three ROIs this is 58.32M operations. At 4.8 GOPS peak, arithmetic-only time
is 12.15ms. Use 20% effective model throughput as a planning assumption, plus
20ms for input preparation, dispatch, feature alignment and postprocessing:

| Assumed model throughput / peak | Arithmetic estimate + 20ms overhead |
| --- | ---: |
| 10% | 141.5ms |
| 20% | 80.8ms |
| 30% | 60.5ms |

With that overhead allowance, meeting 100ms requires at least 15.2% of peak.
Neither the previous 10% nor the current 20% is a measurement of this model.
Timing starts with pixels in memory and excludes sensor exposure/readout.
Resize/PSRAM transfer cost depends on source ROI area even with fixed neural
inputs. Operator mapping and dispatch costs can invalidate this estimate.

User-provided calibration: `face_detection_yunet_2023mar_int8.onnx`, resized to
160x128, takes about 25-28ms including preprocessing, inference and postprocessing
on their setup. Static inspection of the supplied graph gives 53 QLinearConv
nodes and 17,211,200 convolution MAC at that size (input shape in the file is
1x3x640x640; convolution spatial shapes were scaled to 1x3x128x160). This includes
the box, class, objectness and landmark heads; no inference was run.

That corresponds to 1.23-1.38 effective GOPS, or 25.6-28.7% of the 4.8 GOPS peak,
using convolution operations over the measured end-to-end time. It is not NPU
core utilization. Simple MAC-ratio extrapolation puts the new network at
42.4-47.4ms, but differing layer shapes, memory access, three ROI preprocessing
paths and multiple invocations prevent treating this as a latency prediction.
The 80.8ms planning case above deliberately retains a lower throughput assumption
and a separate overhead allowance. YuNet timing already includes its own overhead;
do not describe the MAC-ratio estimate as pure NPU time.

Postprocessing keeps 64 ambient candidates and 16 per ROI per local head. With
three ROIs NMS sees at most 112 candidates, and HD matching sees at most 48.
Only selected candidates have box coordinates decoded. The final output is
capped at 64 detections. These caps also limit dense-scene recall and independent
evidence matching; they are configurable in the detector config.

For sequential INT8 MCU inference:

| Buffer | Bytes |
| --- | ---: |
| Ambient input | 4,800 |
| Retained ambient features, 30x40x32 | 38,400 |
| Ambient raw head output, 30x40x6 | 7,200 |
| One packed ROI input, 16x16x16 | 4,096 |
| One ROI feature, 8x8x64 | 4,096 |
| One sampled ambient context, 8x8x32 | 2,048 |
| Fusion concatenation, 8x8x99 | 6,336 |
| Two local raw head outputs | 768 |

There are 129,440 convolution weights (~126.4 KiB INT8), 5.08x the prior version,
and 1,426 folded biases (5,704 bytes INT32), excluding quantization tables and
compiled graph metadata. These counts do not grow with the number of ROIs.
The ambient upsample/add can require three simultaneous 38,400-byte buffers;
with its input/output this is about 127.2KB of live feature storage under a reused
buffer schedule, not a measured tensor arena size. Budget 160KiB for feature/I/O
buffers and target at most 384KiB total CNN SRAM, including any weights staged
to RAM, scratch, quantization metadata and bounded postprocessing buffers. Prefer
internal flash for persistent weights and reuse ROI workspace across invocations.
Actual converter arena allocation remains unverified. Raw ROI/frame storage
belongs outside this budget in PSRAM or streaming input buffers. PyTorch training
batches ROIs and does not implement the MCU buffer allocator.

This remains a float training pipeline. INT8 conversion/QAT, model-interface
scales, actual NPU mapping, SRAM allocation and 100ms latency have not been verified.
No training or tests have been run.
