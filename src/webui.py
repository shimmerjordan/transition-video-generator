"""Web 配置前端(FastAPI):全部配置与操作在网页完成,config.yaml 仅作后台存储。

阶段:0 裁剪/分段(时间轨道) · 去水印(画面框选) · 1 时间线 · 2 换背景 · 3 换装 · 4 灯光 · 5 出片
另有「高级」原始 YAML 兜底。

启动:.venv/Scripts/python src/webui.py   →  http://127.0.0.1:8800
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import contract, pipeline, s1_beats  # noqa: E402
from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, project_root, resolve_path  # noqa: E402

try:
    import cv2
    import yaml
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
    import uvicorn
except ImportError as e:  # pragma: no cover
    raise SystemExit("缺少 web 依赖,请安装:pip install fastapi uvicorn") from e

ROOT = project_root()
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
app = FastAPI(title="转场视频生成器 · 配置前端")
RUN = {"running": False, "log": []}


def _clean(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _media_dirs(cfg: dict) -> list[str]:
    ds = []
    src = get(cfg, "input.source", "")
    if src:
        ds.append(os.path.dirname(resolve_path(cfg, src)))
    for key in ("input.backgrounds_dir", "input.garments_dir"):
        d = get(cfg, key, "")
        if d:
            ds.append(resolve_path(cfg, d))
    seen, out = set(), []
    for d in ds:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def _resolve_media(cfg: dict, file: str) -> str | None:
    if not file:
        return None
    if os.path.isabs(file) and os.path.isfile(file):
        return file
    p = resolve_path(cfg, file)
    if os.path.isfile(p):
        return p
    for d in _media_dirs(cfg):
        cand = os.path.join(d, os.path.basename(file))
        if os.path.isfile(cand):
            return cand
    return None


def _segment_ids(cfg: dict) -> list[int]:
    return [s.get("id") for s in get(cfg, "segments", []) or []]


# ---------------- 结构化配置 API ----------------

@app.get("/api/config.json")
def config_json():
    return JSONResponse(_clean(load_config(CONFIG_PATH)))


@app.post("/api/config.json")
async def save_config_json(req: Request):
    try:
        data = _clean(await req.json())
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/api/assets")
def assets():
    cfg = load_config(CONFIG_PATH)
    images, videos, seen = [], [], set()
    for d in _media_dirs(cfg):
        for name in sorted(os.listdir(d)):
            ext = os.path.splitext(name)[1].lower()
            if name in seen:
                continue
            seen.add(name)
            if ext in IMG_EXT:
                images.append(name)
            elif ext in VID_EXT:
                info = {}
                try:
                    vi = video.video_info(os.path.join(d, name))
                    info = {"duration": round((vi["count"] / vi["fps"]) if vi["fps"] else 0, 1),
                            "w": vi["width"], "h": vi["height"]}
                except Exception:
                    pass
                videos.append({"name": name, **info})
    return {"images": images, "videos": videos}


@app.get("/api/frame_at")
def frame_at(file: str, t: float = 0.0):
    cfg = load_config(CONFIG_PATH)
    path = _resolve_media(cfg, file)
    if not path:
        return JSONResponse({"error": f"找不到 {file}"}, status_code=404)
    if os.path.splitext(path)[1].lower() in IMG_EXT:
        return FileResponse(path)
    vi = video.video_info(path)
    idx = int(t * (vi["fps"] or 30))
    frames = list(video.read_frames(path, start=idx, count=1)) or \
        list(video.read_frames(path, start=0, count=1))
    ok, buf = cv2.imencode(".jpg", frames[0])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/suggest_beats")
def suggest_beats(subdivide: int = 1, source: str = "music"):
    cfg = load_config(CONFIG_PATH)
    audio = _resolve_media(cfg, get(cfg, "input.music", "")) if source == "music" else None
    if not audio:
        src = _resolve_media(cfg, get(cfg, "input.source", ""))
        if not src:
            return JSONResponse({"error": "找不到音乐,也找不到源片"}, status_code=404)
        audio = os.path.join(contract.work_root(ROOT), "_audio.wav")
        if not video.extract_audio(src, audio):
            return JSONResponse({"error": "源片无音轨,请配置 input.music"}, status_code=404)
    bpm, beats, dur = s1_beats.detect_beats(audio)
    cuts = s1_beats.build_cut_times(beats, dur, subdivide=subdivide, override=None,
                                    include_start_end=False)
    return {"cut_times": cuts, "duration": round(dur, 2), "bpm": round(bpm, 1)}


# ---------------- 原始 YAML(高级)----------------

@app.get("/api/config", response_class=PlainTextResponse)
def get_config_raw():
    return open(CONFIG_PATH, encoding="utf-8").read() if os.path.isfile(CONFIG_PATH) else ""


@app.post("/api/config")
async def save_config_raw(req: Request):
    text = (await req.body()).decode("utf-8")
    try:
        yaml.safe_load(text)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"YAML 解析失败: {e}"})
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return {"ok": True}


# ---------------- 状态 / 运行 / 成片 ----------------

@app.get("/api/status")
def status():
    cfg = load_config(CONFIG_PATH)
    art = contract.artifact_status(ROOT, _segment_ids(cfg))
    out_map = {"beats": "beats", "camera": "locked", "matte": "alpha", "plates": "plates",
               "garment": "garment", "relight": "relit", "composite": "comp", "assemble": "final"}
    rows = []
    for i, step in enumerate(pipeline.STEP_ORDER, 1):
        prov = get(cfg, f"providers.{step}", "local") if step in (
            "matte", "ground", "garment", "relight") else "local"
        a = out_map[step]
        if a in ("beats", "final"):
            st = "✓" if art.get(a) else "—"
        else:
            st = " ".join(f"s{k}:{v}" for k, v in (art.get(a, {}) or {}).items()) or "—"
        rows.append({"idx": i, "step": step, "provider": prov, "status": st})
    bgs = get(cfg, "backgrounds", {}) or {}
    clean = {n: os.path.isfile(contract.clean_path(ROOT, n)) for n in bgs}
    clips = {}
    for b in bgs.values():
        for c in b.get("clips", []) or []:
            if c.get("id"):
                clips[c["id"]] = os.path.isfile(contract.clip_path(ROOT, c["id"]))
    return {"steps": rows, "clean": clean, "clips": clips}


@app.post("/api/run")
async def run_pipeline(req: Request):
    if RUN["running"]:
        return {"ok": False, "error": "已有任务在运行"}
    body = await req.json()
    seg = str(body.get("segment", "")).strip()
    stp = str(body.get("steps", "")).strip()
    segment = int(seg) if seg else None
    steps = pipeline.parse_steps(stp or None)

    def worker():
        RUN.update(running=True, log=[])
        try:
            pipeline.run(CONFIG_PATH, segment, steps, log=lambda m: RUN["log"].append(str(m)))
        except Exception:
            RUN["log"].append("ERROR:\n" + traceback.format_exc())
        finally:
            RUN["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


@app.post("/api/run_tool")
async def run_tool(req: Request):
    if RUN["running"]:
        return {"ok": False, "error": "已有任务在运行"}
    body = await req.json()
    tool = body.get("tool")
    name = body.get("name") or None
    cfg = load_config(CONFIG_PATH)

    def worker():
        RUN.update(running=True, log=[])
        lg = lambda m: RUN["log"].append(str(m))
        try:
            from src import preprocess
            if tool == "dewatermark":
                preprocess.run_dewatermark(cfg, ROOT, name, log=lg)
            elif tool == "clips":
                preprocess.run_clips(cfg, ROOT, name, log=lg)
            else:
                lg(f"未知工具 {tool}")
        except Exception:
            lg("ERROR:\n" + traceback.format_exc())
        finally:
            RUN["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


@app.get("/api/run/log")
def run_log():
    return {"running": RUN["running"], "log": RUN["log"][-500:]}


@app.get("/api/video")
def video_file(path: str):
    full = os.path.abspath(os.path.join(ROOT, path))
    base = os.path.abspath(os.path.join(ROOT, "data"))
    if not full.startswith(base) or not os.path.isfile(full):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(full, media_type="video/mp4")


@app.get("/api/final")
def final():
    p = contract.artifact_path(ROOT, "final")
    if os.path.isfile(p):
        return FileResponse(p, media_type="video/mp4")
    return JSONResponse({"error": "尚无成片"}, status_code=404)


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html><html lang=zh><meta charset=utf-8>
<title>转场视频生成器 · 配置</title>
<style>
body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{padding:10px 18px;background:#171a21;border-bottom:1px solid #2a2f3a;display:flex;align-items:center;gap:12px}
header b{font-size:16px}
nav{display:flex;gap:4px;padding:8px 14px;background:#13161c;border-bottom:1px solid #2a2f3a;flex-wrap:wrap}
nav button{background:#222733;color:#ccd;border:1px solid #2a2f3a;border-radius:6px;padding:6px 12px;cursor:pointer}
nav button.on{background:#2d6cdf;color:#fff}
.layout{display:flex;align-items:flex-start}main{padding:16px;flex:1;min-width:0}
aside{width:340px;border-left:1px solid #2a2f3a;padding:12px;background:#13161c;min-height:90vh;position:sticky;top:0}
#outs div{font-size:12px;padding:1px 0}#outs .ok{color:#4ad07a}#outs .no{color:#667}
aside video{max-height:120px}
.tab{display:none}.tab.on{display:block}
h2{margin:4px 0 12px;font-size:18px}h3{margin:14px 0 6px}
.row{display:flex;align-items:center;gap:8px;margin:6px 0;flex-wrap:wrap}
.row label{min-width:90px;color:#9aa}
.sub{border:1px solid #2a2f3a;border-radius:8px;padding:10px;margin:10px 0;background:#13161c}
.phasehint{color:#8a93a6;font-size:13px;margin-bottom:8px}
button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:6px 11px;cursor:pointer;margin:2px}
button.alt{background:#3a4150}button.go{background:#2a9d5b;font-size:15px;padding:8px 16px}
input,select{background:#0b0d12;color:#cfe;border:1px solid #2a2f3a;border-radius:5px;padding:5px}
table{border-collapse:collapse;margin:6px 0}td,th{border-bottom:1px solid #2a2f3a;padding:4px 8px;text-align:left}
.thumb{height:42px;border-radius:4px;margin-left:6px;vertical-align:middle}
.tag{padding:1px 7px;border-radius:10px;font-size:11px}.local{background:#244a2a}.product{background:#5a3a1a}
#log{height:220px;overflow:auto;background:#0b0d12;border:1px solid #2a2f3a;border-radius:6px;padding:8px;
font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap}
textarea{width:100%;height:430px;background:#0b0d12;color:#cfe;border:1px solid #2a2f3a;border-radius:6px;
font-family:Consolas,monospace;font-size:12px;padding:8px}
.hint{color:#8a93a6;font-size:12px}video{max-width:100%;border-radius:6px}
/* 时间轨道 */
.track{position:relative;height:48px;background:#0b0d12;border:1px solid #2a2f3a;border-radius:6px;margin:8px 0 4px;overflow:hidden;user-select:none}
.tick{position:absolute;top:0;font-size:10px;color:#566;border-left:1px solid #1d2530;padding-left:2px;height:100%;pointer-events:none}
.blk{position:absolute;top:9px;height:30px;background:rgba(45,108,223,.55);border:1px solid #2d6cdf;border-radius:4px;cursor:grab;display:flex;align-items:center;justify-content:center;min-width:14px}
.blk .lbl{font-size:11px;color:#fff;pointer-events:none;white-space:nowrap}
.hl,.hr{position:absolute;top:0;width:9px;height:100%;cursor:ew-resize;background:#5b8def}
.hl{left:0;border-radius:4px 0 0 4px}.hr{right:0;border-radius:0 4px 4px 0}
.blk .del{position:absolute;top:-8px;right:-6px;background:#a33;border:0;color:#fff;border-radius:9px;padding:0 5px;cursor:pointer;font-size:11px}
.prev{width:100%;max-height:340px;object-fit:contain;border:1px solid #2a2f3a;border-radius:6px;background:#000;display:block}
.playhead{position:absolute;top:0;width:0;border-left:2px solid #ffd400;height:100%;z-index:5;pointer-events:none}
#wmcanvas{border:1px solid #2a2f3a;border-radius:6px;cursor:crosshair;max-width:100%;touch-action:none}
#picker{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9}
#pkbox{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:14px;max-width:92vw;margin:3vh auto;overflow:auto;max-height:92vh}
#pkimg{max-width:100%;cursor:crosshair;border:1px solid #2a2f3a}
</style>
<header><b>🎬 转场视频生成器</b><button onclick=save()>💾 保存全部</button><span id=msg class=hint></span></header>
<nav id=nav></nav>
<div class=layout>
<main>
 <div class=tab id=t_clip></div>
 <div class=tab id=t_wm></div>
 <div class=tab id=t_p1></div>
 <div class=tab id=t_p2></div>
 <div class=tab id=t_p3></div>
 <div class=tab id=t_p4></div>
 <div class=tab id=t_p5></div>
 <div class=tab id=t_adv>
   <h2>高级:原始 YAML</h2><p class=hint>「保存全部」会重写本文件并丢失注释。</p>
   <textarea id=rawcfg></textarea>
   <div><button onclick=saveRaw()>保存 YAML</button><button class=alt onclick=loadRaw()>重载</button></div>
 </div>
</main>
<aside>
 <h3>运行 / 状态</h3><div class=row>段(留空=全部):<input id=seg size=4 placeholder=all></div>
 <table id=steps></table>
 <div id=outs></div>
 <h3>日志 <span id=runflag class=hint></span></h3><div id=log></div>
</aside>
<datalist id=provlist><option>local</option><option>product:kling</option><option>product:runway</option></datalist>
</div>
<div id=picker><div id=pkbox>
 <div class=row>文件:<input id=pkfile size=26> 时间(秒):<input id=pktime type=number value=2 size=5>
  <button class=alt onclick=loadPickFrame()>载入帧</button>
  <button class=alt onclick="G('picker').style.display='none'">关闭</button></div>
 <p class=hint id=pkhint></p><img id=pkimg onclick=pkClick(event)>
</div></div>
<script>
const G=id=>document.getElementById(id);
let CFG={},ASSETS={images:[],videos:[]},_picker=null;
let WMV={name:null,kind:'watermarks',t:2,img:null,scale:1};
const TABS=[['wm','1·去水印'],['clip','2·裁剪分段'],['p1','3·时间线'],['p2','4·换背景'],
 ['p3','5·换装'],['p4','6·灯光'],['p5','7·出片'],['adv','高级']];

function buildNav(){G('nav').innerHTML=TABS.map(([k,n])=>`<button id=nav_${k} onclick=showTab('${k}')>${n}</button>`).join('');}
function showTab(k){TABS.forEach(([t])=>{G('t_'+t).classList.toggle('on',t==k);G('nav_'+t).classList.toggle('on',t==k);});
 if(k=='adv')loadRaw();if(k=='wm')wmInit();}
function msg(s){G('msg').textContent=s;setTimeout(()=>G('msg').textContent='',4000);}
function sel(opts,val,bind){return `<select onchange="${bind}=this.value">`+
 opts.map(o=>`<option ${o==val?'selected':''}>${o}</option>`).join('')+`</select>`;}
function selFn(opts,val,fn){return `<select onchange="${fn}(this.value)">`+
 opts.map(o=>`<option ${o==val?'selected':''}>${o}</option>`).join('')+`</select>`;}
function vidOpts(val,bind){let o=['',...ASSETS.videos.map(v=>v.name)];
 return `<select onchange="${bind}=this.value">`+o.map(n=>`<option ${n==val?'selected':''}>${n}</option>`).join('')+`</select>`;}
function imgOpts(val,bind){let o=['',...ASSETS.images];
 return `<select onchange="${bind}=this.value">`+o.map(n=>`<option ${n==val?'selected':''}>${n}</option>`).join('')+`</select>`;}
function num(val,bind,sz){return `<input type=number value="${val}" onchange="${bind}=+this.value" size=${sz||5}>`;}
function r1(x){return Math.round(x*10)/10;}
function durOf(name){let f=(CFG.backgrounds[name]||{}).file;let v=ASSETS.videos.find(x=>x.name==f);return (v&&v.duration)||30;}

async function boot(){buildNav();
 CFG=await (await fetch('/api/config.json')).json();
 ASSETS=await (await fetch('/api/assets')).json();
 renderAll();showTab('wm');setInterval(()=>{if(G('runflag').textContent.includes('运行'))poll();},2500);refresh();}
function renderAll(){
 G('t_clip').innerHTML=secInputs()+secClips();
 G('t_wm').innerHTML=secWatermark();
 G('t_p1').innerHTML=secTimeline();
 G('t_p2').innerHTML=secPersons()+secProvider('matte','抠像')+secGround()+runBtn('▶ 运行换背景','2,3,4,7,8');
 G('t_p3').innerHTML=secProvider('garment','换装')+secGarmentSched()+runBtn('▶ 运行换装','5');
 G('t_p4').innerHTML=secProvider('relight','灯光')+secRelight()+runBtn('▶ 运行灯光','6');
 G('t_p5').innerHTML=secComposite()+runBtn('▶ 合成出片','7,8')+secFinal();
 clipsInit();wmInit();}
async function save(){let r=await fetch('/api/config.json',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(CFG)});let j=await r.json();
 if(j.ok){msg('✓ 已保存');CFG=await (await fetch('/api/config.json')).json();ASSETS=await (await fetch('/api/assets')).json();renderAll();}
 else msg('✗ '+j.error);}

/* ---- 输入路径 ---- */
function secInputs(){let inp=CFG.input=CFG.input||{};return `<h2>0 · 裁剪 / 分段</h2>
 <p class=phasehint>拖动片段块移动、拖两端手柄改起止;拖动时上方预览该时刻画面,不用回视频里数秒。</p>
 <div class=sub><h3>素材路径</h3>
 <div class=row><label>source</label><input value="${inp.source||''}" onchange="CFG.input.source=this.value" size=42></div>
 <div class=row><label>music</label><input value="${inp.music||''}" onchange="CFG.input.music=this.value" size=42></div>
 <div class=row><label>背景目录</label><input value="${inp.backgrounds_dir||''}" onchange="CFG.input.backgrounds_dir=this.value" size=42></div>
 <div class=row><label>服装目录</label><input value="${inp.garments_dir||''}" onchange="CFG.input.garments_dir=this.value" size=42></div></div>`;}

/* ---- 裁剪:时间轨道 ---- */
function secClips(){CFG.backgrounds=CFG.backgrounds||{};
 let h=`<h2>2 · 裁剪 / 分段</h2><p class=phasehint>点轨道任意处移动黄色指针查看该时刻画面;拖片段块移动、拖两端手柄改起止。画面始终显示当前指针处。</p>`;
 for(const name of Object.keys(CFG.backgrounds)){let b=CFG.backgrounds[name];b.clips=b.clips||[];let dur=durOf(name);
  h+=`<div class=sub><div class=row><b>${name}</b> 文件:${vidOpts(b.file,`CFG.backgrounds['${name}'].file`)} <span class=hint>时长 ${dur}s</span>
   <button class=alt onclick="delete CFG.backgrounds['${name}'];renderAll();showTab('clip')">删背景</button></div>
   <img id="pv_${name}" class=prev src=""><div class=hint>指针 <span id="pvl_${name}">0</span>s</div>
   <div class=track id="trk_${name}" onpointerdown="trackDown(event,'${name}')"><div class=playhead id="ph_${name}"></div>`;
  let step=Math.max(1,Math.round(dur/10));
  for(let s=0;s<=dur;s+=step)h+=`<span class=tick style=left:${s/dur*100}%>${s}</span>`;
  b.clips.forEach((c,j)=>{c.range=c.range||[0,5];let l=c.range[0]/dur*100,w=(c.range[1]-c.range[0])/dur*100;
   h+=`<div class=blk id="blk_${name}_${j}" style="left:${l}%;width:${w}%" onpointerdown="startDrag(event,'${name}',${j},'move')">
    <div class=hl onpointerdown="event.stopPropagation();startDrag(event,'${name}',${j},'l')"></div>
    <span class=lbl>${r1(c.range[0])}~${r1(c.range[1])}s</span>
    <div class=hr onpointerdown="event.stopPropagation();startDrag(event,'${name}',${j},'r')"></div>
    <button class=del onclick="event.stopPropagation();CFG.backgrounds['${name}'].clips.splice(${j},1);renderAll();showTab('clip')">×</button></div>`;});
  h+=`</div><table>`;
  b.clips.forEach((c,j)=>{h+=`<tr><td>id <input value="${c.id||''}" onchange="CFG.backgrounds['${name}'].clips[${j}].id=this.value" size=10></td>
   <td>起 <input type=number value="${c.range[0]}" onchange="CFG.backgrounds['${name}'].clips[${j}].range[0]=+this.value;renderAll();showTab('clip')" size=5></td>
   <td>止 <input type=number value="${c.range[1]}" onchange="CFG.backgrounds['${name}'].clips[${j}].range[1]=+this.value;renderAll();showTab('clip')" size=5></td></tr>`;});
  h+=`</table><button class=alt onclick="addClip('${name}')">+ 片段</button>
   <div class=row style=margin-top:6px><button class=go onclick="runTool('clips','${name}')">▶ 生成片段</button>
    <span class=hint>把每个子片段裁成独立视频(优先用去水印后的版本)</span></div>`;
  b.clips.forEach(c=>{if(c.id)h+=`<div class=hint>${c.id}.mp4</div>
   <video controls width=260 src="/api/video?path=data/work/clips/${c.id}.mp4&_=${Date.now()}"></video>`;});
  h+=`</div>`;}
 h+=`<div class=sub><div class=row>新增背景名:<input id=nbgname size=8> 文件:${vidOpts('','window._nbg')}<button onclick=addBg()>+ 背景</button></div></div>`;
 return h;}
function addClip(name){let cs=CFG.backgrounds[name].clips,dur=durOf(name);
 let st=cs.length?Math.min(dur-2,cs[cs.length-1].range[1]||0):0;
 cs.push({id:name+'_'+cs.length,range:[r1(st),r1(Math.min(dur,st+5))]});renderAll();showTab('clip');}
function addBg(){let n=G('nbgname').value.trim();if(!n||!window._nbg){msg('填名并选文件');return;}
 CFG.backgrounds[n]={file:window._nbg,clips:[],cleanup:{watermarks:[],subtitles:[],movers:[]}};window._nbg='';renderAll();showTab('clip');}
let CURT={},_pvt={};
function clipsInit(){for(const name in CFG.backgrounds){let cs=CFG.backgrounds[name].clips||[];
  if(CURT[name]==null)CURT[name]=cs.length?cs[0].range[0]:0;
  layoutPlayhead(name);setPrev(name,true);}}
function layoutPlayhead(name){let ph=G('ph_'+name);if(ph)ph.style.left=(CURT[name]/durOf(name)*100)+'%';
 let lbl=G('pvl_'+name);if(lbl)lbl.textContent=r1(CURT[name]);}
function setPrev(name,force){let now=Date.now();if(!force&&now-(_pvt[name]||0)<110)return;_pvt[name]=now;
 let pv=G('pv_'+name);if(pv)pv.src='/api/frame_at?file='+encodeURIComponent(CFG.backgrounds[name].file)+'&t='+CURT[name]+'&_='+now;}
function trackDown(e,name){scrub(e,name);
 let mv=ev=>scrub(ev,name),up=()=>{document.removeEventListener('pointermove',mv);document.removeEventListener('pointerup',up);};
 document.addEventListener('pointermove',mv);document.addEventListener('pointerup',up);}
function scrub(e,name){let trk=G('trk_'+name),r=trk.getBoundingClientRect(),dur=durOf(name);
 CURT[name]=r1(Math.max(0,Math.min(dur,(e.clientX-r.left)/r.width*dur)));layoutPlayhead(name);setPrev(name);}
function layoutBlk(name,j){let dur=durOf(name),c=CFG.backgrounds[name].clips[j],b=G('blk_'+name+'_'+j);
 if(b){b.style.left=(c.range[0]/dur*100)+'%';b.style.width=((c.range[1]-c.range[0])/dur*100)+'%';
  let lbl=b.querySelector('.lbl');if(lbl)lbl.textContent=r1(c.range[0])+'~'+r1(c.range[1])+'s';}}
function startDrag(e,name,j,mode){e.preventDefault();e.stopPropagation();
 let track=G('trk_'+name),dur=durOf(name),pxs=track.clientWidth/dur;
 let clip=CFG.backgrounds[name].clips[j],sx=e.clientX,o=[clip.range[0],clip.range[1]];
 function mv(ev){let d=(ev.clientX-sx)/pxs;
  if(mode=='move'){let len=o[1]-o[0],ns=Math.max(0,Math.min(o[0]+d,dur-len));clip.range=[r1(ns),r1(ns+len)];CURT[name]=clip.range[0];}
  else if(mode=='l'){clip.range[0]=Math.max(0,Math.min(r1(o[0]+d),clip.range[1]-0.2));CURT[name]=clip.range[0];}
  else if(mode=='r'){clip.range[1]=Math.min(dur,Math.max(r1(o[1]+d),clip.range[0]+0.2));CURT[name]=clip.range[1];}
  layoutBlk(name,j);layoutPlayhead(name);setPrev(name);}
 function up(){document.removeEventListener('pointermove',mv);document.removeEventListener('pointerup',up);renderAll();showTab('clip');}
 document.addEventListener('pointermove',mv);document.addEventListener('pointerup',up);}

/* ---- 去水印:画面框选 + 显示矩形 ---- */
function secWatermark(){let names=Object.keys(CFG.backgrounds||{});
 if(!WMV.name||!names.includes(WMV.name))WMV.name=names[0]||null;
 if(!WMV.name)return `<h2>去水印 / 字幕</h2><p class=hint>先在「裁剪/分段」添加背景。</p>`;
 let dur=durOf(WMV.name);
 return `<h2>去水印 / 字幕</h2>
 <p class=phasehint>选背景与类型,在画面上按住拖出矩形;红框=已选区域(可删)。运行时这些区域用本地 inpaint 抹除;动态路人请把 cleanup 设为 product。</p>
 <div class=sub><div class=row>背景:${selFn(names,WMV.name,'wmSetVideo')} 类型:${selFn(['watermarks','subtitles'],WMV.kind,'wmSetKind')}</div>
  <div class=row>时间:<input type=range id=wmt min=0 max=${Math.max(1,Math.floor(dur))} step=0.5 value=${WMV.t} oninput="WMV.t=+this.value;wmLoad()"> <span id=wmtl>${WMV.t}s</span></div>
  <canvas id=wmcanvas width=720 height=405></canvas>
  <div id=wmlist></div>
  <div class=row style=margin-top:10px>${provInline('cleanup','去水印工具')}
   <button class=go onclick="runTool('dewatermark','${WMV.name}')">▶ 去水印(本背景)</button>
   <button class=alt onclick="runTool('dewatermark','')">全部背景</button></div>
  <div class=hint>输出(去水印后整段视频):</div>
  <video controls src="/api/video?path=data/work/clean/${WMV.name}.mp4&_=${Date.now()}"></video>
  </div>`;}
function wmSetVideo(v){WMV.name=v;renderAll();showTab('wm');}
function wmSetKind(k){WMV.kind=k;renderAll();showTab('wm');}
function wmInit(){if(!WMV.name||!CFG.backgrounds[WMV.name])return;
 let b=CFG.backgrounds[WMV.name];b.cleanup=b.cleanup||{};b.cleanup[WMV.kind]=b.cleanup[WMV.kind]||[];
 if(G('wmcanvas'))wmLoad();}
function wmLoad(){if(!G('wmcanvas')||!WMV.name)return;if(G('wmtl'))G('wmtl').textContent=WMV.t+'s';
 let img=new Image();img.onload=()=>{WMV.img=img;let cv=G('wmcanvas');let CW=Math.min(760,img.naturalWidth);
  WMV.scale=CW/img.naturalWidth;cv.width=CW;cv.height=Math.round(img.naturalHeight*WMV.scale);wmAttach(cv);wmDraw();wmList();};
 img.src='/api/frame_at?file='+encodeURIComponent(CFG.backgrounds[WMV.name].file)+'&t='+WMV.t+'&_='+Date.now();}
function wmDraw(temp){let cv=G('wmcanvas');if(!cv)return;let ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
 if(WMV.img)ctx.drawImage(WMV.img,0,0,cv.width,cv.height);
 let arr=(CFG.backgrounds[WMV.name].cleanup[WMV.kind]||[]);ctx.lineWidth=2;ctx.strokeStyle='#ff4444';ctx.fillStyle='rgba(255,68,68,.25)';
 arr.forEach(r=>{let x=r[0]*WMV.scale,y=r[1]*WMV.scale,w=(r[2]-r[0])*WMV.scale,hh=(r[3]-r[1])*WMV.scale;ctx.fillRect(x,y,w,hh);ctx.strokeRect(x,y,w,hh);});
 if(temp){ctx.strokeStyle='#44ff88';ctx.lineWidth=2;ctx.strokeRect(temp.x,temp.y,temp.w,temp.h);}}
function wmAttach(cv){if(cv._a)return;cv._a=1;let st=null;
 cv.addEventListener('pointerdown',e=>{let r=cv.getBoundingClientRect();st={x:e.clientX-r.left,y:e.clientY-r.top};});
 cv.addEventListener('pointermove',e=>{if(!st)return;let r=cv.getBoundingClientRect();let x=e.clientX-r.left,y=e.clientY-r.top;
  wmDraw({x:Math.min(st.x,x),y:Math.min(st.y,y),w:Math.abs(x-st.x),h:Math.abs(y-st.y)});});
 let fin=e=>{if(!st)return;let r=cv.getBoundingClientRect();let x=e.clientX-r.left,y=e.clientY-r.top;
  let x0=Math.min(st.x,x)/WMV.scale,y0=Math.min(st.y,y)/WMV.scale,x1=Math.max(st.x,x)/WMV.scale,y1=Math.max(st.y,y)/WMV.scale;st=null;
  if(Math.abs(x1-x0)<5||Math.abs(y1-y0)<5){wmDraw();return;}
  CFG.backgrounds[WMV.name].cleanup[WMV.kind].push([Math.round(x0),Math.round(y0),Math.round(x1),Math.round(y1)]);wmDraw();wmList();};
 cv.addEventListener('pointerup',fin);cv.addEventListener('pointerleave',fin);}
function wmList(){let arr=(CFG.backgrounds[WMV.name].cleanup[WMV.kind]||[]);let h='<table>';
 arr.forEach((r,k)=>{h+=`<tr><td>${r.join(', ')}</td><td><button class=alt onclick="CFG.backgrounds['${WMV.name}'].cleanup['${WMV.kind}'].splice(${k},1);wmDraw();wmList()">删</button></td></tr>`;});
 h+='</table>';if(G('wmlist'))G('wmlist').innerHTML=h;}

/* ---- 阶段1:时间线 ---- */
function allClipIds(){let ids=[];for(const n in CFG.backgrounds)for(const c of (CFG.backgrounds[n].clips||[]))if(c.id)ids.push(c.id);return ids;}
function secTimeline(){let b=CFG.beats=CFG.beats||{};b.override_seconds=b.override_seconds||[];CFG.segments=CFG.segments||[];
 let cuts=[...b.override_seconds].sort((x,y)=>x-y);
 let h=`<h2>1 · 时间线(何时切到哪个背景)</h2>
 <p class=phasehint>切换点=背景切换时刻。可生成卡点建议再手动增删;N 个切换点 = N+1 段,给每段选背景。</p>
 <div class=sub><div class=row>每 N 拍切:${num(window._subdiv||1,'window._subdiv',3)}
   <button class=alt onclick=suggestBeats()>🎵 生成卡点建议</button><span class=hint id=beatinfo></span></div>
  <div class=row><b>切换点(秒)</b></div><table>`;
 cuts.forEach((t,i)=>{h+=`<tr><td>${num(t,`CFG.beats.override_seconds[${i}]`,6)}</td>
  <td><button class=alt onclick="CFG.beats.override_seconds.splice(${i},1);renderAll();showTab('p1')">删</button></td></tr>`;});
 h+=`</table><button class=alt onclick="CFG.beats.override_seconds.push(0);renderAll();showTab('p1')">+ 切换点</button>
  <button onclick=applyTimeline()>↻ 按切换点重建分段</button></div>
  <div class=sub><h3>每段背景(共 ${CFG.segments.length} 段)</h3>`;
 let clips=['',...allClipIds()];
 CFG.segments.forEach((s,i)=>{let a=i==0?0:cuts[i-1],z=i<cuts.length?cuts[i]:'结尾';
  h+=`<div class=row><label>段${s.id} [${a}~${z}s]</label>${sel(clips,s.background_clip||'',`CFG.segments[${i}].background_clip`)}</div>`;});
 h+=`</div><div class=row><label>转场</label>${sel(['hard_cut','crossfade','mask_wipe'],(CFG.project||{}).transition||'hard_cut','CFG.project.transition')}</div>`;
 return h;}
async function suggestBeats(){let sd=window._subdiv||1;let j=await (await fetch('/api/suggest_beats?subdivide='+sd)).json();
 if(j.error){msg('✗ '+j.error);return;}CFG.beats=CFG.beats||{};CFG.beats.detect='manual';CFG.beats.override_seconds=j.cut_times;
 applyTimeline();if(G('beatinfo'))G('beatinfo').textContent=`BPM≈${j.bpm}, ${j.duration}s, ${j.cut_times.length} 点`;}
function applyTimeline(){let cuts=[...(CFG.beats.override_seconds||[])].sort((x,y)=>x-y);
 CFG.beats.override_seconds=cuts;CFG.beats.detect='manual';CFG.beats.include_start_end=true;
 let n=cuts.length+1,old=CFG.segments||[],ns=[];
 for(let i=0;i<n;i++){let o=old[i]||{};ns.push({id:i,background_clip:o.background_clip||'',ground:o.ground||'as_is',
  garments:o.garments||{},light:o.light||{direction:'auto',color_temp:'auto',intensity:'auto'}});}
 CFG.segments=ns;renderAll();showTab('p1');}

/* ---- 阶段2:换背景 ---- */
function secPersons(){CFG.persons=CFG.persons||[];
 let h=`<h2>2 · 换背景</h2><p class=phasehint>设人物(「取点」在源片上单击)。运行:运镜对齐→抠像→背景平面→合成→出片。</p>
 <div class=sub><h3>人物</h3><table><tr><th>id</th><th>名称</th><th>seed_point</th><th></th></tr>`;
 CFG.persons.forEach((p,i)=>{h+=`<tr><td><input value="${p.id||''}" onchange="CFG.persons[${i}].id=this.value" size=4></td>
  <td><input value="${p.name||''}" onchange="CFG.persons[${i}].name=this.value" size=8></td>
  <td><input value="${(p.seed_point||[]).join(',')}" onchange="CFG.persons[${i}].seed_point=this.value.split(',').map(Number)" size=10>
   <button class=alt onclick=pickPoint(${i})>取点</button></td>
  <td><button class=alt onclick="CFG.persons.splice(${i},1);renderAll();showTab('p2')">删</button></td></tr>`;});
 h+=`</table><button onclick="CFG.persons.push({id:'p'+CFG.persons.length,name:'',seed_point:[0,0]});renderAll();showTab('p2')">+ 人物</button></div>`;
 return h;}
function secGround(){CFG.segments=CFG.segments||[];let h='<div class=sub><h3>每段地面策略</h3>';
 if(!CFG.segments.length)h+='<span class=hint>先在「时间线」建分段</span>';
 CFG.segments.forEach((s,i)=>{h+=`<div class=row><label>段${s.id}</label>${sel(['as_is','generate','virtual_plane'],s.ground||'as_is',`CFG.segments[${i}].ground`)}</div>`;});
 return h+'</div>';}

/* ---- 阶段3:换装 ---- */
function secGarmentSched(){CFG.garments=CFG.garments||{};CFG.segments=CFG.segments||[];
 let persons=(CFG.persons||[]).map(p=>p.id),gars=['(不换)',...Object.keys(CFG.garments)];
 let h='<div class=sub><h3>服装库</h3><table>';
 for(const id of Object.keys(CFG.garments)){let f=CFG.garments[id];
  h+=`<tr><td>${id}</td><td>${imgOpts(f,`CFG.garments['${id}']`)}`+(f?`<img class=thumb src="/api/frame_at?file=${encodeURIComponent(f)}">`:'')+
  `</td><td><button class=alt onclick="delete CFG.garments['${id}'];renderAll();showTab('p3')">删</button></td></tr>`;}
 h+=`</table><div class=row>新增 id:<input id=ngid size=6> 图:${imgOpts('','window._ng')}<button onclick=addGarment()>+ 服装</button></div></div>`;
 h+='<div class=sub><h3>按段按人换装</h3>';
 if(!persons.length)h+='<span class=hint>先在阶段2加人物</span>';
 CFG.segments.forEach((s,i)=>{s.garments=s.garments||{};h+=`<div class=row><label>段${s.id}</label>`;
  persons.forEach(pid=>{let cur=s.garments[pid]||'(不换)';
   h+=`${pid}:<select onchange="setSegG(${i},'${pid}',this.value)">`+gars.map(g=>`<option ${g==cur?'selected':''}>${g}</option>`).join('')+`</select> `;});
  h+=`</div>`;});
 return h+'</div>';}
function setSegG(i,pid,val){if(val=='(不换)')delete CFG.segments[i].garments[pid];else CFG.segments[i].garments[pid]=val;}
function addGarment(){let id=G('ngid').value.trim();if(!id||!window._ng){msg('填 id 并选图');return;}CFG.garments[id]=window._ng;window._ng='';renderAll();showTab('p3');}

/* ---- 阶段4 / 5 ---- */
function secRelight(){let r=CFG.relight=CFG.relight||{};return `<div class=sub><h3>重打光</h3>
 <div class=row><label>本地回退</label>${sel(['passthrough','soft_match'],r.fallback||'soft_match','CFG.relight.fallback')}</div>
 <div class=row><label>soft 强度</label>${num(r.soft_strength??0.25,'CFG.relight.soft_strength')}</div>
 <p class=hint>真实重打光建议 provider=product 或本地接 IC-Light。</p></div>`;}
function secComposite(){let c=CFG.compositing=CFG.compositing||{};let cb=(k,v)=>`<input type=checkbox ${v?'checked':''} onchange="CFG.compositing['${k}']=this.checked">`;
 return `<h2>5 · 细节 / 出片</h2><p class=phasehint>调真实感细节后合成出片。</p><div class=sub>
 <div class=row><label>光包裹</label>${cb('light_wrap',c.light_wrap!==false)} 强度${num(c.light_wrap_amount??0.5,'CFG.compositing.light_wrap_amount')}</div>
 <div class=row><label>接触阴影</label>${cb('contact_shadow',c.contact_shadow!==false)} 浓度${num(c.shadow_strength??0.45,'CFG.compositing.shadow_strength')}</div>
 <div class=row><label>颗粒</label>${cb('grain',c.grain!==false)} 强度${num(c.grain_sigma??3.0,'CFG.compositing.grain_sigma')}</div>
 <div class=row><label>调色</label>${cb('match_color',c.match_color!==false)} 强度${num(c.match_strength??0.25,'CFG.compositing.match_strength')}</div>
 <div class=row><label>背景虚化</label>${num(c.bg_blur??0,'CFG.compositing.bg_blur')}</div></div>`;}
function secFinal(){return `<div class=sub><h3>成片预览</h3><video controls src="/api/final?_=${Date.now()}"></video>
 <div><button class=alt onclick="renderAll()">刷新预览</button></div></div>`;}

/* ---- 通用 ---- */
function secProvider(step,label){let pv=CFG.providers=CFG.providers||{};
 return `<div class=sub><div class=row><label>${label} provider</label>
  <input value="${pv[step]||'local'}" onchange="CFG.providers['${step}']=this.value" list=provlist>
  <span class=hint>local 或 product:kling / product:runway</span></div>
  <datalist id=provlist><option>local</option><option>product:kling</option><option>product:runway</option></datalist></div>`;}
function provInline(step,label){let pv=CFG.providers=CFG.providers||{};
 return `${label}:<input value="${pv[step]||'local'}" onchange="CFG.providers['${step}']=this.value" list=provlist size=12>`;}
function runBtn(label,steps){return `<div class=row><button class=go onclick="runSteps('${steps}')">${label}</button>
 <span class=hint>(步骤 ${steps};段见右侧)</span></div>`;}
async function runSteps(steps){await save();
 await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({segment:G('seg').value,steps:steps})});poll();}
async function runTool(tool,name){await save();
 await fetch('/api/run_tool',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tool,name})});poll();}

function pickPoint(i){openPicker(CFG.input.source,'point',(x,y)=>{CFG.persons[i].seed_point=[x,y];renderAll();showTab('p2');});}
function openPicker(file,mode,cb){_picker={file,mode,cb,pts:[]};G('pkfile').value=file||'';
 G('pkhint').textContent='单击取一个点';G('picker').style.display='block';loadPickFrame();}
function loadPickFrame(){G('pkimg').src='/api/frame_at?file='+encodeURIComponent(G('pkfile').value)+'&t='+(G('pktime').value||0)+'&_='+Date.now();_picker.pts=[];}
function pkClick(e){let img=e.target,r=img.getBoundingClientRect();
 let x=Math.round((e.clientX-r.left)*img.naturalWidth/r.width),y=Math.round((e.clientY-r.top)*img.naturalHeight/r.height);
 _picker.cb(x,y);G('picker').style.display='none';}

async function refresh(){let j=await (await fetch('/api/status')).json();
 let h='<tr><th>步骤</th><th>provider</th><th>产物</th></tr>';
 for(const s of j.steps){let cls=s.provider.startsWith('product')?'product':'local';
  h+=`<tr><td>${s.idx}.${s.step}</td><td><span class="tag ${cls}">${s.provider}</span></td><td>${s.status}</td></tr>`;}
 G('steps').innerHTML=h;
 let o='';
 if(j.clean&&Object.keys(j.clean).length){o+='<h3>去水印产物</h3>';
  for(const n in j.clean)o+=`<div class="${j.clean[n]?'ok':'no'}">${j.clean[n]?'✓':'—'} ${n}.mp4</div>`;}
 if(j.clips&&Object.keys(j.clips).length){o+='<h3>片段产物</h3>';
  for(const c in j.clips)o+=`<div class="${j.clips[c]?'ok':'no'}">${j.clips[c]?'✓':'—'} ${c}.mp4</div>`;}
 if(G('outs'))G('outs').innerHTML=o;}
async function poll(){let j=await (await fetch('/api/run/log')).json();let l=G('log');
 l.textContent=j.log.join('\n');l.scrollTop=l.scrollHeight;
 G('runflag').textContent=j.running?'(运行中…)':'(空闲)';if(j.running)setTimeout(poll,1000);else refresh();}
async function loadRaw(){G('rawcfg').value=await (await fetch('/api/config')).text();}
async function saveRaw(){let r=await fetch('/api/config',{method:'POST',body:G('rawcfg').value});
 let j=await r.json();if(j.ok){msg('✓ YAML 已保存');boot();}else msg('✗ '+j.error);}
boot();
</script></html>"""


def main():
    cfg = load_config(CONFIG_PATH) if os.path.isfile(CONFIG_PATH) else {}
    host = get(cfg, "webui.host", "127.0.0.1")
    port = int(get(cfg, "webui.port", 8800))
    print(f"配置前端: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
