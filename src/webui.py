"""Web 配置前端(FastAPI):编辑配置、为每步选 provider、按段配置、触发运行、浏览产物。

启动:
    .venv/Scripts/python src/webui.py        # 默认 http://127.0.0.1:8800
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import contract, pipeline  # noqa: E402
from src.utils.config import get, load_config, project_root  # noqa: E402

try:
    import yaml
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
    import uvicorn
except ImportError as e:  # pragma: no cover
    raise SystemExit("缺少 web 依赖,请安装:pip install fastapi uvicorn") from e

ROOT = project_root()
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
app = FastAPI(title="转场视频生成器 · 配置前端")

RUN = {"running": False, "log": [], "step": ""}


def _log(msg: str) -> None:
    RUN["log"].append(str(msg))


def _segment_ids(cfg: dict) -> list[int]:
    return [s.get("id") for s in get(cfg, "segments", []) or []]


STEPS = pipeline.STEP_ORDER

PAGE = """<!doctype html><html lang=zh><meta charset=utf-8>
<title>转场视频生成器 · 配置</title>
<style>
body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
header{padding:12px 20px;background:#171a21;border-bottom:1px solid #2a2f3a;font-weight:600}
.wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
.card{background:#171a21;border:1px solid #2a2f3a;border-radius:8px;padding:14px;flex:1;min-width:340px}
h3{margin:0 0 10px}textarea{width:100%;height:420px;background:#0b0d12;color:#cfe;border:1px solid #2a2f3a;
border-radius:6px;font-family:Consolas,monospace;font-size:12px;padding:8px}
button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:7px 12px;cursor:pointer;margin:2px}
button.alt{background:#3a4150}table{width:100%;border-collapse:collapse;font-size:13px}
td,th{border-bottom:1px solid #2a2f3a;padding:6px;text-align:left}
#log{height:240px;overflow:auto;background:#0b0d12;border:1px solid #2a2f3a;border-radius:6px;
padding:8px;font-family:Consolas,monospace;font-size:12px;white-space:pre-wrap}
.tag{padding:1px 7px;border-radius:10px;font-size:11px}.local{background:#244a2a}.product{background:#5a3a1a}
input{background:#0b0d12;color:#cfe;border:1px solid #2a2f3a;border-radius:5px;padding:5px}
</style>
<header>🎬 转场视频生成器 · 配置前端</header>
<div class=wrap>
 <div class=card style=flex:1.3>
  <h3>config.yaml</h3>
  <textarea id=cfg></textarea>
  <div><button onclick=saveCfg()>保存配置</button><button class=alt onclick=loadCfg()>重新载入</button>
  <span id=cfgmsg></span></div>
 </div>
 <div class=card>
  <h3>步骤 / Provider / 产物</h3>
  <table id=steps></table>
  <h3 style=margin-top:14px>运行</h3>
  <div>段(留空=全部): <input id=seg size=4 placeholder=all>
   步骤(如 3,5 留空=全部): <input id=stp size=8 placeholder=all></div>
  <div style=margin-top:8px><button onclick=run()>▶ 运行</button>
   <button class=alt onclick=refresh()>刷新状态</button></div>
  <h3 style=margin-top:14px>日志 <span id=runflag></span></h3>
  <div id=log></div>
 </div>
</div>
<script>
async function loadCfg(){let r=await fetch('/api/config');document.getElementById('cfg').value=await r.text();}
async function saveCfg(){let r=await fetch('/api/config',{method:'POST',body:document.getElementById('cfg').value});
 let j=await r.json();document.getElementById('cfgmsg').textContent=j.ok?'✓ 已保存':'✗ '+j.error;refresh();}
async function refresh(){let r=await fetch('/api/status');let j=await r.json();
 let h='<tr><th>步骤</th><th>provider</th><th>各段产物</th></tr>';
 for(const s of j.steps){let cls=s.provider.startsWith('product')?'product':'local';
  h+=`<tr><td>${s.idx}. ${s.step}</td><td><span class="tag ${cls}">${s.provider}</span></td><td>${s.status}</td></tr>`;}
 document.getElementById('steps').innerHTML=h;}
async function run(){let seg=document.getElementById('seg').value,stp=document.getElementById('stp').value;
 await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({segment:seg,steps:stp})});poll();}
async function poll(){let r=await fetch('/api/run/log');let j=await r.json();
 let l=document.getElementById('log');l.textContent=j.log.join('\\n');l.scrollTop=l.scrollHeight;
 document.getElementById('runflag').textContent=j.running?'(运行中…)':'(空闲)';
 if(j.running)setTimeout(poll,1000);else refresh();}
loadCfg();refresh();setInterval(()=>{if(document.getElementById('runflag').textContent.includes('运行'))poll();},2000);
</script></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/config", response_class=PlainTextResponse)
def get_config():
    if not os.path.isfile(CONFIG_PATH):
        return ""
    return open(CONFIG_PATH, encoding="utf-8").read()


@app.post("/api/config")
async def save_config(req: Request):
    text = (await req.body()).decode("utf-8")
    try:
        yaml.safe_load(text)  # 校验
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"YAML 解析失败: {e}"})
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return {"ok": True}


@app.get("/api/status")
def status():
    cfg = load_config(CONFIG_PATH)
    segs = _segment_ids(cfg)
    art = contract.artifact_status(ROOT, segs)
    out_map = {"beats": "beats", "camera": "locked", "matte": "alpha", "plates": "plates",
               "garment": "garment", "relight": "relit", "composite": "comp", "assemble": "final"}
    rows = []
    for i, step in enumerate(STEPS, 1):
        prov = get(cfg, f"providers.{step}", "local") if step in (
            "matte", "cleanup", "ground", "garment", "relight") else "local"
        a = out_map[step]
        if a in ("beats", "final"):
            st = "✓" if art.get(a) else "—"
        else:
            counts = art.get(a, {})
            st = " ".join(f"s{k}:{v}" for k, v in counts.items()) or "—"
        rows.append({"idx": i, "step": step, "provider": prov, "status": st})
    return {"steps": rows, "segments": segs}


@app.post("/api/run")
async def run_pipeline(req: Request):
    if RUN["running"]:
        return {"ok": False, "error": "已有任务在运行"}
    body = await req.json()
    seg = body.get("segment", "").strip()
    stp = body.get("steps", "").strip()
    segment = int(seg) if seg else None
    steps = pipeline.parse_steps(stp or None)

    def worker():
        RUN.update(running=True, log=[])
        try:
            pipeline.run(CONFIG_PATH, segment, steps, log=_log)
        except Exception:
            _log("ERROR:\n" + traceback.format_exc())
        finally:
            RUN["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


@app.get("/api/run/log")
def run_log():
    return {"running": RUN["running"], "log": RUN["log"][-400:]}


@app.get("/api/frame")
def frame(art: str, seg: int, i: int = 0):
    d = contract.seg_dir(ROOT, art, seg)
    p = os.path.join(d, f"f{i:05d}.png")
    if os.path.isfile(p):
        return FileResponse(p)
    return JSONResponse({"error": "not found"}, status_code=404)


def main():
    cfg = load_config(CONFIG_PATH) if os.path.isfile(CONFIG_PATH) else {}
    host = get(cfg, "webui.host", "127.0.0.1")
    port = int(get(cfg, "webui.port", 8800))
    print(f"配置前端: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
