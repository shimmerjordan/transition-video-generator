# transition-video-generator 转场视频生成器

把一段**跳舞(或任意主体)视频**,按音乐**节拍卡点**切换背景,并按需**按人按段换装**、
重打光,合成出**像在每个场景当场实拍**的转场视频。

项目做成一个**可配置编排器**:每个步骤可自由选择「**本地工具** or **现成付费AI产品**」,
两者通过**统一的标准件(artifact)**对接,可随时互换。配套一个 **Web 配置前端**。

> 技术方案见 [PLAN.md](PLAN.md);标准件格式见 [docs/ARTIFACTS.md](docs/ARTIFACTS.md)。

---

## 1. 三个核心概念

| 概念 | 含义 |
|---|---|
| **步骤(step)** | 管线共 8 步:`beats→camera→matte→plates→garment→relight→composite→assemble` |
| **Provider** | 每步可选 `local`(本地实现)或 `product:<名>`(付费产品,走交接流程)。在 `config.yaml` 的 `providers` 配置 |
| **标准件(artifact)** | 每步读写固定格式/路径的中间产物(帧序列 PNG + JSON)。只要格式一致,本地/付费可互换 |

数据流(每段独立处理,最后拼接):

```
音乐 ─► 卡点(beats.json)
跳舞视频 ─► 运镜跟踪 T_t ─► 反稳定到「锁定域」─┐         (源片可有运镜)
                         └─► 抠像(按人 alpha) │
背景视频 ─► 选子片段 ─► 去水印/路人 ─► 稳定 ─► 补地面 ─► 干净背景 plate
   每段(锁定域): 换装(按人按段) ─► 重打光 ─► 合成(光包裹/接触阴影/颗粒/调色)
把 T_t 贴回(前景背景同运镜)─► 按卡点转场拼接 + 配乐 ─► 成片 final.mp4
```

---

## 2. 安装

已自带 `.venv`(Python 3.13)与基础 CV 依赖。若要在新机器重建:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

- 编码/混音用内置 ffmpeg(`imageio-ffmpeg`),**无需另装系统 ffmpeg**。
- 抠像/换装/重打光的**本地重型模型**(SAM2、SD+ControlNet、IC-Light)按需后装;
  不装也能用付费 `product:*` 路径完成这些步骤。

---

## 3. 准备素材

把素材放进 `data/input/`,或在 `config.yaml` 的 `input` 段指向现有路径(支持相对路径,
当前默认指向 `e:\data\workspace\source\`):

| 素材 | 配置项 | 说明 |
|---|---|---|
| 跳舞视频 | `input.source` | 主体视频(可带运镜) |
| 卡点音乐 | `input.music` | 决定切换节拍(**必需**,否则 S1 无法卡点) |
| 背景视频 | `input.backgrounds_dir` + `backgrounds` | 可多个、可切多段 |
| 服装参考图 | `input.garments_dir` + `garments` | 每件一张图 |

---

## 4. 配置(在网页里完成,无需手改 YAML)

**推荐全程用 Web 前端配置**(见第 5 节),`config.yaml` 只是后台存储。网页提供结构化表单:

- **常规/Provider**:fps、分辨率、转场;每步选本地或付费产品。
- **人物**:增删人物,「取点」按钮可直接在源片画面上单击设定 `seed_point`。
- **服装**:从素材图片下拉选,带缩略图。
- **背景**:选视频(自动显示时长)、定义多个子片段(起止秒)、「框选添加」点两角即生成去水印/字幕矩形。
- **分段(EDL)**:下拉选背景子片段、地面策略、给每个人选该段穿哪件。
- **运行**:填段/步骤一键运行,看产物状态与实时日志。

下面是这些表单背后写入 `config.yaml` 的字段含义(供理解/高级直接编辑):

### config.yaml 字段参考

```yaml
project:
  fps: 30
  resolution: [1280, 720]
  transition: hard_cut          # hard_cut | crossfade | mask_wipe

providers:                      # ★每步:本地 or 付费产品
  matte:   local                #   local | product:runway ...
  cleanup: local
  ground:  local
  garment: local                #   换装难,常用 product:kling
  relight: local

input:
  source: ../source/dance.mp4
  music:  data/input/music.wav
  backgrounds_dir: ../source
  garments_dir: ../source

persons:                        # 人物身份(换装按人指定)
  - { id: p0, name: 男舞者, seed_point: [820, 360] }
  - { id: p1, name: 女舞者, seed_point: [470, 380] }

garments:                       # 服装库
  cloth1: cloth1.jpg
  cloth2: cloth2.jpg

backgrounds:                    # 背景库:每个视频可切多个命名子片段
  meadow:
    file: bg1.mp4
    clips:
      - { id: meadow_a, range: [10, 18] }     # 取 10~18 秒
      - { id: meadow_b, range: [60, 70] }
    cleanup:                                   # 静态去元素(原分辨率矩形 [x0,y0,x1,y1])
      watermarks: [[1120, 20, 1260, 70]]
      subtitles:  [[0, 640, 1280, 720]]
      movers: []                               # 路人留空 → 交付费视频修复

segments:                       # ★EDL:每个卡点段怎么配(段数 = 卡点数)
  - id: 0
    background_clip: meadow_a    # 用哪段背景
    ground: as_is               # as_is | generate(补地面) | virtual_plane
    garments: { p1: cloth1 }     # 谁换什么(省略=不换),与背景切换同步
    light: { direction: auto, color_temp: auto, intensity: auto }
```

要点:
- **背景子片段**:在 `backgrounds.*.clips` 定义时间段,`segments[*].background_clip` 引用。
- **按人按段换装**:`segments[*].garments` 用 `人物id: 服装id`,可只换某人。
- **去元素**:`cleanup.watermarks/subtitles` 是静态矩形,本地 `cv2.inpaint` 去除;`movers`(走动路人)留空,交给 `providers.cleanup: product:*`。
- **地面**:背景没拍到地面时设 `ground: generate`(付费生成)或 `virtual_plane`(本地铺虚拟地面)。

---

## 5. 运行

### 方式 A:Web 前端(推荐,配置+运行全在网页)

```bash
.venv/Scripts/python src/webui.py
# 打开 http://127.0.0.1:8800
```

网页按**创作流程分 6 个阶段**,右侧常驻运行状态与日志。顶部「💾 保存全部」写回配置:

| 阶段 | 做什么 | 运行的底层步骤 |
|---|---|---|
| **去水印**(第1步) | 画面**拖拽框选**水印/字幕(红框可见可删),选工具(本地/付费),**▶运行**→产出去水印整段视频 `clean/<bg>.mp4` | dewatermark |
| **裁剪/分段** | **剪映式时间轨道**:拖指针看画面、拖片段块/手柄改起止,**▶生成片段**→产出 `clips/<clipid>.mp4` | clip |
| **1 时间线** | **🎵 生成卡点建议**→手动增删切换点;给**每段指定背景**;选转场 | beats |
| **2 换背景** | 设人物(**画面取点**)、选 matte provider、每段地面策略 → **▶运行** | camera·matte·plates·composite·assemble |
| **3 换装** | 服装库 + **按段按人**选穿哪件 + garment provider → **▶运行** | garment |
| **4 灯光** | 重打光参数/provider → **▶运行** | relight |
| **5 细节/出片** | 光包裹/接触阴影/颗粒/调色 + **成片预览** → **▶合成出片** | composite·assemble |

无需手动编辑 YAML(「高级」标签页提供原始 YAML 兜底)。

### 方式 B:命令行

```bash
# 单段 POC(强烈建议先跑通一段)
.venv/Scripts/python src/pipeline.py --segment 0

# 全片
.venv/Scripts/python src/pipeline.py

# 只跑某些步骤(1 beats … 8 assemble)
.venv/Scripts/python src/pipeline.py --steps 3,5     # 只 matte、garment
```

产物在 `data/work/<步骤>/seg_<段>/`,成片在 `data/output/final.mp4`。

---

## 6. 用付费产品做某一步(handoff 流程)

把某步 provider 改成 `product:<名>`(如 `garment: product:kling`)后运行,该步会:

1. **导出**标准输入到 `data/work/handoff/<步骤>/seg_<段>/`
   (含 `in_locked/`、`in_alpha/`、换装还会附 `in_garment_refs/` 与 `swap_plan.json`,
   以及 `README.txt` 操作指引)。
2. 你在该付费产品里处理这些输入。
3. 把结果(帧序列 `f00000.png…` 或一个视频)放进 `data/work/ingest/<步骤>/seg_<段>/`。
4. **再次运行该步**,系统自动把结果归一化成标准产物,链路继续。

> 产品若有 API,可在对应 provider 实现里自动完成 1–4,无需手动搬运。

---

## 7. 推荐工作流程(对应网页 6 阶段)

0. **素材**:填好 source/music/目录;给背景定义子片段,框选去掉水印/字幕(动态路人把 `cleanup` provider 设 `product`)。
1. **时间线**:点「生成卡点建议」拿到切换点,手动调整;给每段选背景;选转场。
2. **换背景**:人物「取点」;难抠的把 matte provider 设 `product`;每段定地面策略;**先把右侧「段」填 0 跑通一段**,在 `data/work/` 看 `locked→alpha→plates→comp_locked`,再清空段跑全片。
3. **换装**:配服装库与每段每人穿什么;换装难,通常 garment 用 `product:*`(走 handoff)。
4. **灯光**:重打光(真实级建议 `product:*` 或本地 IC-Light)。
5. **细节/出片**:调真实感细节,合成出片,网页内直接预览。

> 难点步骤(换装/去路人/补地面/重打光)切到 `product:*` 后,运行会在 `data/work/handoff/` 生成交接包;详见第 6 节。

---

## 8. 现状(哪些已就绪)

| 步骤 | 本地实现 | 备注 |
|---|---|---|
| beats / camera / plates / composite / assemble | ✅ 真实可用 | 纯 CV,运镜匹配是真实感核心 |
| cleanup(静态去水印/字幕) | ✅ 本地 inpaint | 动态路人需 `product:*` |
| matte 抠像 | ⏳ 目前中值法(合成测试用) | 真实素材需 SAM2(本地)或付费,**按人**输出 p0/p1 |
| garment 换装 | ⏳ 本地透传 | 推荐 `product:*`(如 Kling 试衣);本地 SD 接口预留 |
| relight 重打光 | ⏳ soft_match 近似 | 本地 IC-Light 接口预留 |

> 即:整条链路现在就能端到端出片;`matte/garment/relight` 三步换成付费 `product:*` 或装本地模型后,效果升级为真实级。

---

## 9. 排错

- **S1 报找不到音乐**:`input.music` 指向的文件不存在,放入 `data/input/music.wav`。
- **某段报缺少 locked/alpha**:该段前置步骤没跑,先按顺序跑或用 `--steps` 补。
- **去水印没干净/误伤**:`cleanup` 的矩形是按背景**原分辨率**的 `[x0,y0,x1,y1]`,按实际位置调整。
- **WebUI 打不开**:确认 `webui.host/port`(默认 127.0.0.1:8800)未被占用。
- **中文日志乱码(Windows)**:命令行加 `PYTHONUTF8=1` 或用 WebUI 看日志。
