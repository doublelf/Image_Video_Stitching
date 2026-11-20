"""HTTP service that stitches uploaded videos.

    uvicorn hstitch.server.app:app --host 0.0.0.0 --port 8000

Endpoints:

``POST /api/v1/stitch/videos``
    multipart upload of two or more videos, returns a job id
``GET  /api/v1/stitch/videos/{job_id}``
    job status
``GET  /api/v1/stitch/videos/{job_id}/result``
    the stitched mp4
``GET  /``
    a small upload page
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .jobs import JobStore

app = FastAPI(title="hstitch video service", version="1.0.0")

_store = JobStore(
    os.environ.get(
        "HSTITCH_JOB_ROOT", os.path.join(tempfile.gettempdir(), "hstitch-jobs")
    )
)


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/stitch/videos")
async def create_job(
    files: List[UploadFile] = File(..., description="two or more videos"),
    mode: str = Form("blend"),
    blend: str = Form("multiband"),
    sync: str = Form("timestamp"),
    height: Optional[int] = Form(None),
    fps: Optional[float] = Form(None),
    overlap_px: Optional[str] = Form(None),
    max_frames: Optional[int] = Form(None),
) -> dict:
    """Accept the videos, queue the stitching and return the job id."""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="at least two videos are needed")
    options = {"mode": mode, "blend": blend, "sync": sync}
    if height:
        options["height"] = int(height)
    if fps:
        options["fps"] = float(fps)
    if max_frames:
        options["max_frames"] = int(max_frames)
    if overlap_px:
        try:
            values = [int(item) for item in overlap_px.replace(",", " ").split()]
        except ValueError:
            raise HTTPException(
                status_code=400, detail="overlap_px must be a list of integers"
            )
        options["overlap_px"] = values
        options["auto_overlap"] = False

    uploads = []
    for item in files:
        payload = await item.read()
        if not payload:
            raise HTTPException(status_code=400, detail=f"{item.filename} is empty")
        uploads.append((item.filename or "video.mp4", payload))

    job = _store.create(uploads, options)
    return job.as_dict()


@app.get("/api/v1/stitch/videos/{job_id}")
def job_status(job_id: str) -> dict:
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.as_dict()


@app.get("/api/v1/stitch/videos/{job_id}/result")
def job_result(job_id: str) -> FileResponse:
    job = _store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "the job failed")
    if job.status != "done" or not job.output:
        raise HTTPException(status_code=409, detail="the job is not finished yet")
    return FileResponse(job.output, media_type="video/mp4", filename=f"{job_id}.mp4")


INDEX_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>hstitch 视频拼接</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 46rem; }
  fieldset { border: 1px solid #ccc; border-radius: 6px; margin-bottom: 1rem; }
  label { display: inline-block; min-width: 6rem; }
  button { padding: .5rem 1rem; }
  #log { background: #f5f5f5; padding: .8rem; border-radius: 6px; white-space: pre-wrap; }
</style>
</head>
<body>
<h1>hstitch 视频横向拼接</h1>
<form id="form">
  <fieldset>
    <legend>视频（按从左到右的顺序选择）</legend>
    <input type="file" id="files" accept="video/*" multiple required>
  </fieldset>
  <fieldset>
    <legend>参数</legend>
    <div><label>模式</label>
      <select id="mode"><option value="blend">blend 重叠融合</option><option value="concat">concat 并排</option></select>
    </div>
    <div><label>融合器</label>
      <select id="blend"><option value="multiband">multiband</option><option value="feather">feather</option></select>
    </div>
    <div><label>帧同步</label>
      <select id="sync"><option value="timestamp">timestamp 时间戳</option><option value="index">index 帧序号</option></select>
    </div>
    <div><label>重叠像素</label><input id="overlap_px" size="10" placeholder="留空自动估计"></div>
  </fieldset>
  <button type="submit">上传并拼接</button>
</form>
<h2>状态</h2>
<div id="log">等待上传</div>
<script>
const log = (text) => { document.getElementById('log').textContent = text; };
document.getElementById('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = document.getElementById('files').files;
  if (files.length < 2) { log('至少需要两个视频'); return; }
  const body = new FormData();
  for (const file of files) body.append('files', file, file.name);
  body.append('mode', document.getElementById('mode').value);
  body.append('blend', document.getElementById('blend').value);
  body.append('sync', document.getElementById('sync').value);
  const overlap = document.getElementById('overlap_px').value.trim();
  if (overlap) body.append('overlap_px', overlap);
  log('上传中 ...');
  const response = await fetch('/api/v1/stitch/videos', { method: 'POST', body });
  const job = await response.json();
  if (!response.ok) { log('失败: ' + JSON.stringify(job)); return; }
  log('任务 ' + job.job_id + ' 已提交，状态 ' + job.status);
  const timer = setInterval(async () => {
    const state = await (await fetch('/api/v1/stitch/videos/' + job.job_id)).json();
    log('状态 ' + state.status + '  帧数 ' + state.progress + (state.error ? '\\n' + state.error : ''));
    if (state.status === 'done') {
      clearInterval(timer);
      const link = document.createElement('a');
      link.href = '/api/v1/stitch/videos/' + job.job_id + '/result';
      link.textContent = '下载拼接结果 (mp4)';
      link.style.display = 'block';
      link.style.marginTop = '1rem';
      document.getElementById('log').after(link);
    }
    if (state.status === 'failed') clearInterval(timer);
  }, 1000);
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_PAGE)
