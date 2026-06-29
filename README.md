# transition-video-generator 转场视频生成器

一个通用的 **AI 转场视频生成管线**:给定一段源视频 + 一段音乐 + 一批背景/外观素材,
按音乐**节拍卡点**自动切换背景、(可选)切换人物服装与光影,目标是合成出
**像在每个场景当场实拍的**真实转场视频。

> 完整技术方案、风险与落地分工见 **[PLAN.md](PLAN.md)**。

## 定位

- 这是一个**通用工具**,核心能力是「卡点 + 背景替换 + 重打光 + 真实感合成」。
- 各环节**可配置/可选**:主体数量、是否换装、是否重打光、转场方式(硬切/交叉/遮罩)等由 `config.yaml` 驱动。
- **首个用例(driving case)**:固定机位拍摄的双人舞,按音乐卡点换背景 + 换装 + 换光影。

## 现状

- 阶段:**全管线 S1–S8 已实现,可离线端到端跑通**(合成素材已验证出片)。
  - **纯 CV 环节真实可用**:S1 卡点、S2 运镜跟踪/反稳定、S3 抠像(时序中值)、S4 背景稳定+光照估计、S7 真实度合成、S8 贴回运镜+转场+混音。
  - **GPU 环节惰性降级**:S5 换装(SD+ControlNet)、S6 重打光(IC-Light)目前走 passthrough/soft_match,**装上模型即升级**为真实效果。
- 环境:已建 `.venv`(Python 3.13)并装好 CV 依赖(见 requirements.txt)。
- 硬件:中低端 N 卡(8–12GB);GPU 步骤接入后按需启用。

## 运行

```bash
# 1) 准备素材到 data/input/(source.mp4 / music.wav / backgrounds/ / garments/),并在 config.yaml 配置 segments
# 2) 单段 POC(推荐先跑通):
.venv/Scripts/python src/pipeline.py --segment 0
# 3) 全片:
.venv/Scripts/python src/pipeline.py
# 也可分步:--steps 2-7;单步:python src/s2_camera.py --segment 0
```

产物在 `data/work/`(各步骤中间帧)与 `data/output/final.mp4`。

## 管线概览

```
music ──► beats.json(卡点)
source.mp4 ──运镜跟踪 T_t ──► 反稳定到锁定域(支持源片带视角移动)
          └──SAM2──► 主体 alpha 序列
backgrounds/ ──稳定化──► 每段 locked plate
   每段独立(可选,锁定域): 换装(ControlNet+参考图) → 重打光(IC-Light) → 合成(light wrap/接触阴影/颗粒/景深)
把 T_t 贴回成片(前景背景同运镜) ──► 按 beats 转场拼接 + 配乐 ──► 成片
```

> 源视频**可以有运镜**:先反稳定到锁定域做合成,最后把源片运镜整体贴回,使前景与替换背景共享同一摄像机运动。

## 目录

```
data/input/     源素材:source.mp4 / music.wav / backgrounds/ / garments/
data/work/      中间产物(帧/mask/alpha/打光)
data/output/    成片
models/         模型权重(SAM2 / IC-Light / ControlNet / 试穿)
src/            s1_beats … s7_assemble + pipeline.py
config.yaml     中枢配置:每个节拍段 ↔ 背景素材 ↔ 服装参考图 ↔ 光照 ↔ 转场方式
```

## 下一步

1. 准备素材到 `data/input/`(源视频、音乐、背景视频、服装参考图)。
2. 搭建 venv(建议 Python 3.10/3.11)+ 安装依赖(见 PLAN「环境搭建」)。
3. 先跑通**单段 3–5 秒 POC**(`pipeline.py --segment N`),验收换装稳定性与人景融合度,再批量。
