"""Background jobs for the upload service."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..video.pipeline import VideoStitchOptions, stitch_videos

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass
class Job:
    """One upload and its stitching run."""

    id: str
    status: str = PENDING
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    error: Optional[str] = None
    progress: int = 0
    total: int = 0
    report: Optional[Dict] = None
    sources: List[str] = field(default_factory=list)
    output: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "created": self.created,
            "finished": self.finished,
            "error": self.error,
            "progress": self.progress,
            "total": self.total,
            "sources": self.sources,
            "output": self.output,
            "report": self.report,
            "ready": self.status == DONE and bool(self.output),
        }


class JobStore:
    """Keeps the uploads on disk and runs the stitching in the background."""

    def __init__(self, root, workers: int = 2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers))

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def create(self, uploads, options: Optional[Dict] = None) -> Job:
        """Store the uploaded files and queue the stitching."""
        options = dict(options or {})
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, sources=[name for name, _ in uploads])
        with self._lock:
            self._jobs[job_id] = job

        folder = self.root / job_id
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, payload in uploads:
            suffix = Path(name).suffix or ".mp4"
            target = folder / f"source{len(paths)}{suffix}"
            target.write_bytes(payload)
            paths.append(str(target))

        self._pool.submit(self._run, job, paths, options, folder)
        return job

    def _run(self, job: Job, paths, options: Dict, folder: Path) -> None:
        job.status = RUNNING
        output = folder / "stitched.mp4"

        def progress(stage: str, done: int, total: int) -> None:
            job.progress = done
            job.total = total

        try:
            stitch_options = VideoStitchOptions(**options)
            report = stitch_videos(paths, str(output), stitch_options, progress)
            job.report = report
            job.output = str(output)
            job.status = DONE
        except Exception as error:  # pragma: no cover - defensive
            job.status = FAILED
            job.error = f"{error}\n{traceback.format_exc(limit=3)}"
        finally:
            job.finished = time.time()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
