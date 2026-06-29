# 标准件契约(Artifact Contract)

每个步骤都读写**固定格式、固定路径**的"标准件"。只要某步骤的实现(本地 or 付费产品)
产出同样的标准件,就能被任意替换,链路不变。这是"每步可选现成产品 / 本地、统一输入输出"的基础。

## 总原则

- **帧序列**:统一为零填充 PNG `f00000.png, f00001.png, …`,BGR/RGBA,放在 `seg_<id>/` 下。
- **每段一目录**:`data/work/<artifact>/seg_<id>/`。多主体再下分 `p0/ p1/`。
- **每个产物带 manifest**:同级 `manifest.json` 描述 `{step, segment, kind, fps, width, height, count, schema}`,供 WebUI 与下游校验。
- **JSON 标准件**:UTF-8、2 空格缩进。
- **schema 版本**:`schema: "1"`,不兼容变更时递增。

## 标准件清单(数据流顺序)

| 名称 | 路径 | 格式 | 产出步骤 |
|---|---|---|---|
| clean(去水印) | `data/work/clean/<bg>.mp4` | H.264 视频(整段去水印) | 去水印 tab |
| clips(片段) | `data/work/clips/<clipid>.mp4` | H.264 视频(裁剪出的子片段) | 裁剪 tab |
| beats | `data/work/beats.json` | JSON(cut_times + segments[start/end 秒&帧]) | S1 |
| src frames | `data/work/src/seg_<id>/f*.png` | BGR PNG(原始机位帧) | S2(抽帧) |
| camera | `data/work/camera/seg_<id>.json` | JSON(每帧 2x3 仿射 A_t + meta) | S2 |
| locked | `data/work/locked/seg_<id>/f*.png` | BGR PNG(反稳定到锁定域) | S2 |
| alpha | `data/work/alpha/seg_<id>/f*.png` | 灰度 PNG 0-255(主体并集) | S3 |
| alpha/person | `data/work/alpha/seg_<id>/p<k>/f*.png` | 灰度 PNG(按人身份,k=0,1…) | S3 |
| plates | `data/work/plates/seg_<id>/f*.png` | BGR PNG(干净锁定背景) | S4 |
| light | `data/work/plates/seg_<id>.light.json` | JSON(方位/色温/强度) | S4 |
| garment | `data/work/garment/seg_<id>/f*.png` | BGR PNG(换装后主体合并帧) | S5 |
| relit | `data/work/relit/seg_<id>/f*.png` | BGR PNG(重打光后主体) | S6 |
| comp | `data/work/comp_locked/seg_<id>/f*.png` | BGR PNG(锁定域合成) | S7 |
| final | `data/output/final.mp4` | H.264+AAC | S8 |

> 约定:所有 `seg_<id>/` 内帧数一致、尺寸一致(= 该段锁定域尺寸),按帧号对齐。

## 每步即时运行(自带工具 + 产物)

每个 tab 既是配置面板,也是一个**可独立运行**的步骤:配好该步的 provider/参数后点 ▶运行,
立即产出上表对应的标准件。例如「去水印」运行 → `clean/<bg>.mp4`;「裁剪」运行 → 每个
`clips/<clipid>.mp4`。下游步骤(plates 等)优先消费这些已生成的标准件。

## Provider:本地 or 现成产品

每个步骤的实现由 `config.providers.<step>` 选择:

```
providers:
  matte:   local            # 本地实现
  garment: product:kling    # 现成产品(走 handoff)
  ...
```

- `local`:调用项目内的本地实现,直接产出标准件。
- `product:<name>`:走 **handoff(人工/半自动交接)**:
  1. **导出**该步骤的标准**输入**到 `data/work/handoff/<step>/seg_<id>/`,附 `HANDOFF.json`(说明该用什么产品、期望产出什么)与 `README.txt`。
  2. 你在该产品里处理,把结果放进 `data/work/ingest/<step>/seg_<id>/`。
  3. **ingest** 把结果归一化成该步骤的标准**输出**路径,链路继续。
  - 产品若有 API,可在 provider 实现里自动完成 1–3,无需人工。

### handoff 包结构

```
data/work/handoff/<step>/seg_<id>/
├── HANDOFF.json      # {step, segment, provider, inputs[], expect, ingest_dir, fps, size}
├── README.txt        # 人类可读操作指引
├── in_*/             # 标准输入(帧序列/参考图/mask 等)
data/work/ingest/<step>/seg_<id>/
└── f*.png            # 你从产品导出的结果(帧序列),由 ingest 归一化
```

只要遵守"输入在 handoff、输出放 ingest、命名按帧号",任何产品都能接。
