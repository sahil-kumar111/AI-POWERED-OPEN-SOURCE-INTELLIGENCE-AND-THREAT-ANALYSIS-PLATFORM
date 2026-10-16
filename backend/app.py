"""AEGIS Threat Intelligence & Analysis Platform - FastAPI Backend Server.

Provides RESTful API endpoints for:
- Video and location submission & validation
- Asynchronous analysis job tracking & progress
- Result retrieval in standardized schema
- Artifact streaming (analyzed video, EigenCAM visual heatmap, reports, GeoJSON)
- Model health checks and sample scenario analysis
"""

from __future__ import annotations

import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.job_manager import job_manager
from backend.pipeline import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    FIRE_SMOKE_MODEL_PATH,
    PERSON_MODEL_PATH,
    PROJECT_ROOT,
    VIDEO_DETECTOR_AVAILABLE,
    WEAPON_MODEL_PATH,
    XAI_AVAILABLE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aegis.api")

UPLOADS_DIR = DEFAULT_OUTPUT_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB

app = FastAPI(
    title="AEGIS Threat Intelligence & Analysis Platform API",
    description="Backend REST API providing real-time multimodal threat detection, ByteTrack tracking, NLP intelligence, OSINT fusion, and explainable AI.",
    version="2.0.0",
)

# Enable CORS for React frontend (Vite default: http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="System Health & Model Status")
def get_health() -> Dict[str, Any]:
    models = {
        "person_detector": {
            "path": str(PERSON_MODEL_PATH),
            "exists": PERSON_MODEL_PATH.is_file(),
        },
        "weapon_detector": {
            "path": str(WEAPON_MODEL_PATH),
            "exists": WEAPON_MODEL_PATH.is_file(),
        },
        "fire_smoke_detector": {
            "path": str(FIRE_SMOKE_MODEL_PATH),
            "exists": FIRE_SMOKE_MODEL_PATH.is_file(),
        },
    }
    all_models_present = all(m["exists"] for m in models.values())

    return {
        "status": "healthy" if all_models_present and VIDEO_DETECTOR_AVAILABLE else "degraded",
        "service": "AEGIS Threat Intelligence Platform",
        "version": "2.0.0",
        "detectors_available": VIDEO_DETECTOR_AVAILABLE,
        "xai_visual_available": XAI_AVAILABLE,
        "models": models,
        "timestamp": os.getenv("API_STARTED_AT", ""),
    }


@app.get("/api/models", summary="List Installed AI Model Detectors")
def list_models() -> Dict[str, Any]:
    return {
        "video_detectors": {
            "weapon": {
                "name": "YOLOv11 Weapon Detector",
                "classes": ["pistol", "knife", "rifle", "shotgun", "smg"],
                "path": str(WEAPON_MODEL_PATH),
                "loaded": WEAPON_MODEL_PATH.is_file(),
            },
            "fire_smoke": {
                "name": "YOLOv11 Fire & Smoke Detector",
                "classes": ["Fire", "Smoke"],
                "path": str(FIRE_SMOKE_MODEL_PATH),
                "loaded": FIRE_SMOKE_MODEL_PATH.is_file(),
            },
            "person": {
                "name": "YOLOv11 Person Detector + ByteTrack",
                "classes": ["person"],
                "path": str(PERSON_MODEL_PATH),
                "loaded": PERSON_MODEL_PATH.is_file(),
            },
        },
        "representation_learning": {
            "embedding_dimension": 384,
            "archetypes": [
                "Armed Assault Hazard",
                "Active Fire & Smoke Emergency",
                "Mass Gathering & Civil Disorder",
                "Low Risk / Monitored Baseline",
            ],
        },
    }


@app.post("/api/analyze", status_code=status.HTTP_202_ACCEPTED, summary="Submit Video & Location for Analysis")
async def analyze_video(
    location: str = Form(..., description="Target event address or location query"),
    video: Optional[UploadFile] = File(None, description="Video file to analyze (.mp4, .avi, .mov)"),
    venue_capacity: int = Form(15, description="Expected venue occupancy / crowd capacity limit"),
    frame_skip: int = Form(1, description="Frame skip factor for faster processing (1 = every frame)"),
) -> Dict[str, Any]:
    loc_clean = location.strip()
    if not loc_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Location cannot be empty. Please specify a monitoring address or city.",
        )

    saved_video_path: Optional[Path] = None

    if video and video.filename:
        suffix = Path(video.filename).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported video format: '{suffix}'. Supported formats are: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
            )

        # Save uploaded file
        dest_filename = f"upload_{int(Path(video.filename).stem != '')}_{Path(video.filename).name}"
        dest_path = UPLOADS_DIR / dest_filename

        try:
            with dest_path.open("wb") as buffer:
                bytes_read = 0
                while chunk := await video.read(1024 * 1024):  # 1MB chunks
                    bytes_read += len(chunk)
                    if bytes_read > MAX_VIDEO_SIZE_BYTES:
                        dest_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Uploaded video exceeds maximum allowed size of {MAX_VIDEO_SIZE_BYTES // (1024 * 1024)} MB.",
                        )
                    buffer.write(chunk)
            saved_video_path = dest_path
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Failed to save uploaded video: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not save uploaded video: {exc}",
            )

    job = job_manager.create_job(
        location=loc_clean,
        video_path=saved_video_path,
        venue_capacity=max(venue_capacity, 1),
        frame_skip=max(frame_skip, 1),
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": "Analysis started in background.",
        "location": loc_clean,
        "has_video": saved_video_path is not None,
    }


@app.post("/api/sample-analyze", status_code=status.HTTP_202_ACCEPTED, summary="Analyze Bundled Scenario Video")
def analyze_sample(
    location: str = Form("Connaught Place, New Delhi", description="Location query"),
    venue_capacity: int = Form(15, description="Venue capacity limit"),
    frame_skip: int = Form(1, description="Frame skip interval"),
) -> Dict[str, Any]:
    sample_file = DEFAULT_INPUT_DIR / "YOUTUBE_YouTubeCCTV196_gun_2.mp4"
    if not sample_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bundled sample video not found at: {sample_file}",
        )

    job = job_manager.create_job(
        location=location.strip() or "Connaught Place, New Delhi",
        video_path=sample_file,
        venue_capacity=max(venue_capacity, 1),
        frame_skip=max(frame_skip, 1),
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": f"Sample analysis initiated on {sample_file.name}.",
        "location": location,
        "sample_video": sample_file.name,
    }


@app.get("/api/jobs/{job_id}", summary="Check Job Analysis Status & Progress")
def get_job_status(job_id: str) -> Dict[str, Any]:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job.to_dict()


@app.get("/api/jobs/{job_id}/results", summary="Get Complete Threat Analysis Results")
def get_job_results(job_id: str) -> Dict[str, Any]:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    if job.status == "processing" or job.status == "queued":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail={
                "message": "Job is still processing.",
                "progress": job.progress,
                "stage": job.stage,
            },
        )
    if job.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Threat analysis failed.",
                "error": job.error,
            },
        )
    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No result recorded for completed job.",
        )

    return job.result


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}", summary="Download or Stream Generated Artifact")
def get_artifact(job_id: str, artifact_name: str):
    job = job_manager.get_job(job_id)
    if not job:
        # Check if requested from default output dir
        candidate = DEFAULT_OUTPUT_DIR / artifact_name
        if candidate.is_file():
            mime_type, _ = mimetypes.guess_type(str(candidate))
            return FileResponse(candidate, media_type=mime_type or "application/octet-stream")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    output_dir = Path(job.output_dir)
    target = output_dir / artifact_name
    if not target.is_file():
        # Fallback check in default output dir
        target = DEFAULT_OUTPUT_DIR / artifact_name

    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact_name}' not found for job '{job_id}'.",
        )

    mime_type, _ = mimetypes.guess_type(str(target))
    if target.suffix == ".geojson":
        mime_type = "application/geo+json"
    elif target.suffix == ".md":
        mime_type = "text/markdown; charset=utf-8"

    return FileResponse(target, media_type=mime_type or "application/octet-stream", filename=target.name)


@app.get("/api/jobs", summary="List All Recent Jobs")
def list_jobs() -> List[Dict[str, Any]]:
    return job_manager.list_jobs()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
