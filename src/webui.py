"""Web 配置前端(FastAPI):全部配置与操作在网页完成,config.yaml 仅作后台存储。

表单化管理:常规/Provider、输入、人物(可视化取点)、服装、背景(子片段+框选去元素)、
分段 EDL(下拉选背景/地面/按人换装)、运行/日志;另有「高级」原始 YAML 兜底。

启动:.venv/Scripts/python src/webui.py   →  http://127.0.0.1:8800
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import contract, pipeline  # noqa: E402
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
    frames = list(video.read_frames(path, start=idx, count=1))
    if not frames:
        frames = list(video.read_frames(path, start=0, count=1))
    ok, buf = cv2.imencode(".jpg", frames[0])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


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


# ---------------- 状态 / 运行 ----------------

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
    return {"steps": rows}


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


@app.get("/api/run/log")
def run_log():
    return {"running": RUN["running"], "log": RUN["log"][-500:]}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html><html lang=zh><meta charset=utf-8>
<title>转场视频生成器 · 配置</title>
<style>
body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{padding:10px 18px;background:#171a21;border-bottom:1px solid #2a2f3a;display:flex;align-items:center;gap:12px}
header b{font-size:16px}nav{display:flex;gap:4px;padding:8px 14px;background:#13161c;border-bottom:1px solid #2a2f3a;flex-wrap:wrap}
nav button{background:#222733;color:#ccd;border:1px solid #2a2f3a;border-radius:6px;padding:6px 12px;cursor:pointer}
nav button.on{background:#2d6cdf;color:#fff}
main{padding:16px;max-width:1000px}
.tab{display:none}.tab.on{display:block}
.row{display:flex;align-items:center;gap:8px;margin:6px 0;flex-wrap:wrap}
.row label{min-width:90px;color:#9aa}
.sub{border:1px solid #2a2f3a;border-radius:8px;padding:10px;margin:10px 0;background:#13161c}
button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:6px 11px;cursor:pointer;margin:2px}
button.alt{background:#3a4150}
input,select{background:#0b0d12;color:#cfe;border:1px solid #2a2f3a;border-radius:5px;padding:5px}
table{border-collapse:collapse;margin:6px 0}td,th{border-bottom:1px solid #2a2f3a;padding:4px 8px;text-align:left}
.thumb{height:46px;border-radius:4px;margin-left:6px;vertical-align:middle}
.tag{padding:1px 7px;border-radius:10px;font-size:11px}.local{background:#244a2a}.product{background:#5a3a1a}
#log{height:240px;overflow:auto;background:#0b0d12;border:1px solid #2a2f3a;border-radius:6px;padding:8px;
font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap}
textarea{width:100%;height:430px;background:#0b0d12;color:#cfe;border:1px solid #2a2f3a;border-radius:6px;
font-family:Consolas,monospace;font-size:12px;padding:8px}
#picker{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9}
#pkbox{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:14px;max-width:90vw;max-height:90vh;
overflow:auto;margin:3vh auto}
#pkimg{max-width:100%;cursor:crosshair;border:1px solid #2a2f3a}
.hint{color:#8a93a6;font-size:12px}
</style>
<header><b>🎬 转场视频生成器 · 配置</b>
 <button onclick=save()>💾 保存全部</button><span id=msg class=hint></span></header>
<nav id=nav></nav>
<main>
 <div class=tab id=t_general></div>
 <div class=tab id=t_persons></div>
 <div class=tab id=t_garments></div>
 <div class=tab id=t_backgrounds></div>
 <div class=tab id=t_segments></div>
 <div class=tab id=t_run>
   <div class=row>段(留空=全部): <input id=seg size=4 placeholder=all>
    步骤(如 3,5 留空=全部): <input id=stp size=8 placeholder=all>
    <button onclick=run()>▶ 运行</button><button class=alt onclick=refresh()>刷新</button></div>
   <p class=hint>步骤编号:1 beats · 2 camera · 3 matte · 4 plates · 5 garment · 6 relight · 7 composite · 8 assemble</p>
   <table id=steps></table>
   <h3>日志 <span id=runflag class=hint></span></h3><div id=log></div>
 </div>
 <div class=tab id=t_adv>
   <p class=hint>高级:直接编辑 config.yaml(结构化表单的兜底)。注意表单「保存全部」会重写本文件并丢失注释。</p>
   <textarea id=rawcfg></textarea>
   <div><button onclick=saveRaw()>保存 YAML</button><button class=alt onclick=loadRaw()>重载</button></div>
 </div>
</main>
<div id=picker><div id=pkbox>
 <div class=row>文件:<input id=pkfile size=30> 时间(秒):<input id=pktime type=number value=2 size=5>
  <button class=alt onclick=loadPickFrame()>载入帧</button>
  <button class=alt onclick="document.getElementById('picker').style.display='none'">关闭</button></div>
 <p class=hint id=pkhint></p>
 <img id=pkimg onclick=pkClick(event)>
</div></div>
<script>
const G=id=>document.getElementById(id);
let CFG={},ASSETS={images:[],videos:[]},_picker=null;
const TABS=[['general','常规/Provider'],['persons','人物'],['garments','服装'],
 ['backgrounds','背景'],['segments','分段'],['run','运行'],['adv','高级']];

function buildNav(){G('nav').innerHTML=TABS.map(([k,n])=>`<button id=nav_${k} onclick=showTab('${k}')>${n}</button>`).join('');}
function showTab(k){TABS.forEach(([t])=>{G('t_'+t).classList.toggle('on',t==k);G('nav_'+t).classList.toggle('on',t==k);});
 if(k=='run')refresh(); if(k=='adv')loadRaw();}
function msg(s){G('msg').textContent=s;setTimeout(()=>G('msg').textContent='',4000);}
function sel(opts,val,bind){return `<select onchange="${bind}=this.value">`+
 opts.map(o=>`<option ${o==val?'selected':''}>${o}</option>`).join('')+`</select>`;}
function vidOpts(val,bind){let o=['',...ASSETS.videos.map(v=>v.name)];
 return `<select onchange="${bind}=this.value">`+o.map(n=>`<option ${n==val?'selected':''}>${n}</option>`).join('')+`</select>`;}
function imgOpts(val,bind){let o=['',...ASSETS.images];
 return `<select onchange="${bind}=this.value">`+o.map(n=>`<option ${n==val?'selected':''}>${n}</option>`).join('')+`</select>`;}

async function boot(){buildNav();
 CFG=await (await fetch('/api/config.json')).json();
 ASSETS=await (await fetch('/api/assets')).json();
 renderAll();showTab('general');}
function renderAll(){renderGeneral();renderPersons();renderGarments();renderBackgrounds();renderSegments();}
async function save(){let r=await fetch('/api/config.json',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(CFG)});let j=await r.json();
 if(j.ok){msg('✓ 已保存');CFG=await (await fetch('/api/config.json')).json();ASSETS=await (await fetch('/api/assets')).json();renderAll();}
 else msg('✗ '+j.error);}

function renderGeneral(){let p=CFG.project=CFG.project||{},pv=CFG.providers=CFG.providers||{},inp=CFG.input=CFG.input||{};
 let h=`<div class=sub><h3>项目</h3>
  <div class=row><label>fps</label><input type=number value="${p.fps||30}" onchange="CFG.project.fps=+this.value" size=4></div>
  <div class=row><label>分辨率</label><input value="${(p.resolution||[1280,720]).join(',')}" onchange="CFG.project.resolution=this.value.split(',').map(Number)"></div>
  <div class=row><label>转场</label>${sel(['hard_cut','crossfade','mask_wipe'],p.transition||'hard_cut','CFG.project.transition')}</div></div>
  <div class=sub><h3>Provider(每步:本地 or 付费产品)</h3><p class=hint>可填 local 或 product:kling / product:runway 等</p>`;
 for(const s of ['matte','cleanup','ground','garment','relight'])
  h+=`<div class=row><label>${s}</label><input value="${pv[s]||'local'}" onchange="CFG.providers['${s}']=this.value" list=provlist></div>`;
 h+=`<datalist id=provlist><option>local</option><option>product:kling</option><option>product:runway</option></datalist></div>
  <div class=sub><h3>输入</h3>
  <div class=row><label>source</label><input value="${inp.source||''}" onchange="CFG.input.source=this.value" size=40></div>
  <div class=row><label>music</label><input value="${inp.music||''}" onchange="CFG.input.music=this.value" size=40></div>
  <div class=row><label>背景目录</label><input value="${inp.backgrounds_dir||''}" onchange="CFG.input.backgrounds_dir=this.value" size=40></div>
  <div class=row><label>服装目录</label><input value="${inp.garments_dir||''}" onchange="CFG.input.garments_dir=this.value" size=40></div></div>`;
 G('t_general').innerHTML=h;}

function renderPersons(){CFG.persons=CFG.persons||[];
 let h='<div class=sub><h3>人物身份</h3><p class=hint>seed_point 用于抠像识别该人;点「取点」在源片画面上单击自动填坐标。</p><table><tr><th>id</th><th>名称</th><th>seed_point</th><th></th></tr>';
 CFG.persons.forEach((p,i)=>{h+=`<tr>
  <td><input value="${p.id||''}" onchange="CFG.persons[${i}].id=this.value" size=4></td>
  <td><input value="${p.name||''}" onchange="CFG.persons[${i}].name=this.value" size=8></td>
  <td><input id=seed${i} value="${(p.seed_point||[]).join(',')}" onchange="CFG.persons[${i}].seed_point=this.value.split(',').map(Number)" size=10>
   <button class=alt onclick="pickPoint(${i})">取点</button></td>
  <td><button class=alt onclick="CFG.persons.splice(${i},1);renderPersons()">删</button></td></tr>`;});
 h+='</table><button onclick="CFG.persons.push({id:\'p\'+CFG.persons.length,name:\'\',seed_point:[0,0]});renderPersons()">+ 人物</button></div>';
 G('t_persons').innerHTML=h;}

function renderGarments(){CFG.garments=CFG.garments||{};
 let h='<div class=sub><h3>服装库</h3><table><tr><th>id</th><th>图片</th><th></th></tr>';
 for(const id of Object.keys(CFG.garments)){let f=CFG.garments[id];
  h+=`<tr><td>${id}</td><td>${imgOpts(f,`CFG.garments['${id}']`)}`+
   (f?`<img class=thumb src="/api/frame_at?file=${encodeURIComponent(f)}">`:'')+
   `</td><td><button class=alt onclick="delete CFG.garments['${id}'];renderGarments()">删</button></td></tr>`;}
 h+=`</table><div class=row>新增 id:<input id=ngid size=6> 图片:${imgOpts('','window._ng')}
  <button onclick="addGarment()">+ 服装</button></div></div>`;
 G('t_garments').innerHTML=h;}
function addGarment(){let id=G('ngid').value.trim();if(!id||!window._ng){msg('填 id 并选图片');return;}
 CFG.garments[id]=window._ng;window._ng='';renderGarments();}

function renderBackgrounds(){CFG.backgrounds=CFG.backgrounds||{};let h='';
 for(const name of Object.keys(CFG.backgrounds)){let b=CFG.backgrounds[name];b.clips=b.clips||[];b.cleanup=b.cleanup||{};
  let dur=(ASSETS.videos.find(v=>v.name==b.file)||{}).duration;
  h+=`<div class=sub><div class=row><b>${name}</b> 文件:${vidOpts(b.file,`CFG.backgrounds['${name}'].file`)}`+
   (dur?`<span class=hint>时长 ${dur}s</span>`:'')+
   ` <button class=alt onclick="delete CFG.backgrounds['${name}'];renderBackgrounds()">删背景</button></div>
   <div>子片段<table><tr><th>id</th><th>起(s)</th><th>止(s)</th><th></th></tr>`;
  b.clips.forEach((c,j)=>{let r=c.range||[0,0];h+=`<tr>
   <td><input value="${c.id||''}" onchange="CFG.backgrounds['${name}'].clips[${j}].id=this.value" size=10></td>
   <td><input type=number value="${r[0]}" onchange="setClipR('${name}',${j},0,this.value)" size=5></td>
   <td><input type=number value="${r[1]}" onchange="setClipR('${name}',${j},1,this.value)" size=5></td>
   <td><button class=alt onclick="CFG.backgrounds['${name}'].clips.splice(${j},1);renderBackgrounds()">删</button></td></tr>`;});
  h+=`</table><button class=alt onclick="CFG.backgrounds['${name}'].clips.push({id:'${name}_'+CFG.backgrounds['${name}'].clips.length,range:[0,5]});renderBackgrounds()">+ 子片段</button></div>`;
  h+=cleanupUI(name,'watermarks','水印')+cleanupUI(name,'subtitles','字幕')+`</div>`;}
 h+=`<div class=sub><div class=row>新增背景名:<input id=nbgname size=8> 文件:${vidOpts('','window._nbg')}
  <button onclick="addBg()">+ 背景</button></div></div>`;
 G('t_backgrounds').innerHTML=h;}
function setClipR(name,j,k,v){let c=CFG.backgrounds[name].clips[j];c.range=c.range||[0,0];c.range[k]=+v;}
function cleanupUI(name,kind,label){let arr=(CFG.backgrounds[name].cleanup[kind]=CFG.backgrounds[name].cleanup[kind]||[]);
 let h=`<div style=margin-top:6px>去${label}(矩形 x0,y0,x1,y1)<table>`;
 arr.forEach((r,k)=>{h+=`<tr><td><input value="${r.join(',')}" onchange="CFG.backgrounds['${name}'].cleanup['${kind}'][${k}]=this.value.split(',').map(Number)" size=18></td>
  <td><button class=alt onclick="CFG.backgrounds['${name}'].cleanup['${kind}'].splice(${k},1);renderBackgrounds()">删</button></td></tr>`;});
 h+=`</table><button class=alt onclick="pickRect('${name}','${kind}')">框选添加(点两角)</button></div>`;return h;}
function addBg(){let n=G('nbgname').value.trim();if(!n||!window._nbg){msg('填名并选文件');return;}
 CFG.backgrounds[n]={file:window._nbg,clips:[],cleanup:{watermarks:[],subtitles:[],movers:[]}};window._nbg='';renderBackgrounds();}

function allClipIds(){let ids=[];for(const n in CFG.backgrounds)for(const c of (CFG.backgrounds[n].clips||[]))if(c.id)ids.push(c.id);return ids;}
function renderSegments(){CFG.segments=CFG.segments||[];let clips=['',...allClipIds()];
 let persons=(CFG.persons||[]).map(p=>p.id),gars=['(不换)',...Object.keys(CFG.garments||{})];let h='';
 CFG.segments.forEach((s,i)=>{s.garments=s.garments||{};
  h+=`<div class=sub><div class=row><b>段 ${s.id}</b>
   背景:${sel(clips,s.background_clip||'',`CFG.segments[${i}].background_clip`)}
   地面:${sel(['as_is','generate','virtual_plane'],s.ground||'as_is',`CFG.segments[${i}].ground`)}
   <button class=alt onclick="CFG.segments.splice(${i},1);renderSegments()">删段</button></div>
   <div class=row>换装:`;
  persons.forEach(pid=>{let cur=s.garments[pid]||'(不换)';
   h+=`${pid}:<select onchange="setSegG(${i},'${pid}',this.value)">`+
    gars.map(g=>`<option ${g==cur?'selected':''}>${g}</option>`).join('')+`</select> `;});
  if(!persons.length)h+='<span class=hint>先在「人物」里加人物</span>';
  h+=`</div></div>`;});
 h+=`<button onclick="addSeg()">+ 段</button>`;
 G('t_segments').innerHTML=h;}
function setSegG(i,pid,val){if(val=='(不换)')delete CFG.segments[i].garments[pid];else CFG.segments[i].garments[pid]=val;}
function addSeg(){let clips=allClipIds();
 CFG.segments.push({id:CFG.segments.length,background_clip:clips[0]||'',ground:'as_is',garments:{},
  light:{direction:'auto',color_temp:'auto',intensity:'auto'}});renderSegments();}

// 可视化取点/框选
function pickPoint(i){openPicker(CFG.input.source,'point',(x,y)=>{CFG.persons[i].seed_point=[x,y];renderPersons();});}
function pickRect(name,kind){openPicker(CFG.backgrounds[name].file,'rect',(rect)=>{
 CFG.backgrounds[name].cleanup[kind]=CFG.backgrounds[name].cleanup[kind]||[];
 CFG.backgrounds[name].cleanup[kind].push(rect);renderBackgrounds();});}
function openPicker(file,mode,cb){_picker={file,mode,cb,pts:[]};G('pkfile').value=file||'';
 G('pkhint').textContent=mode=='point'?'单击画面取一个点':'依次点矩形两个对角';
 G('picker').style.display='block';loadPickFrame();}
function loadPickFrame(){let f=G('pkfile').value,t=G('pktime').value||0;
 G('pkimg').src='/api/frame_at?file='+encodeURIComponent(f)+'&t='+t+'&_='+Date.now();_picker.pts=[];}
function pkClick(e){let img=e.target,r=img.getBoundingClientRect();
 let x=Math.round((e.clientX-r.left)*img.naturalWidth/r.width);
 let y=Math.round((e.clientY-r.top)*img.naturalHeight/r.height);
 if(_picker.mode=='point'){_picker.cb(x,y);G('picker').style.display='none';return;}
 _picker.pts.push([x,y]);G('pkhint').textContent='已取 '+_picker.pts.length+'/2';
 if(_picker.pts.length==2){let[a,b]=_picker.pts;
  _picker.cb([Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]);
  G('picker').style.display='none';}}

// 运行/日志
async function refresh(){let j=await (await fetch('/api/status')).json();
 let h='<tr><th>步骤</th><th>provider</th><th>各段产物</th></tr>';
 for(const s of j.steps){let cls=s.provider.startsWith('product')?'product':'local';
  h+=`<tr><td>${s.idx}. ${s.step}</td><td><span class="tag ${cls}">${s.provider}</span></td><td>${s.status}</td></tr>`;}
 G('steps').innerHTML=h;}
async function run(){await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({segment:G('seg').value,steps:G('stp').value})});poll();}
async function poll(){let j=await (await fetch('/api/run/log')).json();
 let l=G('log');l.textContent=j.log.join('\n');l.scrollTop=l.scrollHeight;
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
