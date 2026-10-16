"""AEGIS Background Job Manager.

Handles asynchronous execution of threat analysis pipeline jobs,
progress tracking, status polling, and job-isolated artifacts.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.pipeline import run_complete_pipeline

logger = logging.getLogger("aegis.job_manager")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = PROJECT_ROOT / "output" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: int  # 0 to 100
    stage: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    location: str = "Connaught Place, New Delhi"
    video_path: Optional[str] = None
    venue_capacity: int = 15
    frame_skip: int = 1
    output_dir: str = ""
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Omit full results from basic status check to keep polling lightweight
        if data["result"]:
            data["has_results"] = True
            del data["result"]
        else:
            data["has_results"] = False
        return data


class JobManager:
    """Thread-safe background job scheduler and tracker."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        location: str,
        video_path: Optional[str | Path] = None,
        venue_capacity: int = 15,
        frame_skip: int = 1,
    ) -> Job:
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        job_output_dir = JOBS_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            status="queued",
            progress=0,
            stage="Job queued for processing",
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            location=location,
            video_path=str(video_path) if video_path else None,
            venue_capacity=venue_capacity,
            frame_skip=frame_skip,
            output_dir=str(job_output_dir),
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job_worker,
            args=(job,),
            daemon=True,
            name=f"AEGIS-Worker-{job_id}",
        )
        thread.start()

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def _run_job_worker(self, job: Job) -> None:
        logger.info(f"Starting job {job.job_id} for location: {job.location}")
        job.status = "processing"
        job.stage = "Starting intelligence pipeline..."
        job.progress = 5

        def _progress_cb(stage: str, percent: int) -> None:
            with self._lock:
                job.stage = stage
                job.progress = percent

        try:
            result = run_complete_pipeline(
                location=job.location,
                video_path=job.video_path,
                venue_capacity=job.venue_capacity,
                zone="zone-alpha-1",
                output_dir=Path(job.output_dir),
                frame_skip=job.frame_skip,
                progress_callback=_progress_cb,
            )

            with self._lock:
                job.status = "completed"
                job.progress = 100
                job.stage = "Analysis complete."
                job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                job.result = result

            logger.info(f"Job {job.job_id} completed successfully.")
        except Exception as exc:
            logger.error(f"Job {job.job_id} failed: {exc}", exc_info=True)
            with self._lock:
                job.status = "failed"
                job.stage = f"Error: {exc}"
                job.error = str(exc)
                job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Singleton instance
job_manager = JobManager()
