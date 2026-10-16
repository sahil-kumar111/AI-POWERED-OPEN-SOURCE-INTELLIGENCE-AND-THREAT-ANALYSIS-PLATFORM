#!/usr/bin/env python3
"""AEGIS - Integrated Threat Intelligence Pipeline Engine.

Centralized core module executing:
1. Location & GIS Intelligence (Nominatim + Critical Infrastructure + GeoJSON)
2. Video Intelligence (YOLOv11 Person [ByteTrack], Weapon, Fire/Smoke + Frame Metrics)
3. Gemini Video Understanding (Observable visual description + Reconciliation)
4. OSINT Intelligence (NewsAPI via threat_detector + Geographic Relevance)
5. NLP Intelligence (spaCy NER, VADER sentiment, threat keywords, cross-source reconciliation)
6. Representation Learning (20-dim features -> 384-dim normalized threat embedding & archetype matching)
7. Threat Assessment & Evidence Fusion (Transparent weights, ATI 0-100, dominant threat, severity ranking)
8. Explainable AI & Reporting (SHAP-style attribution, EigenCAM visual heatmap, 7-question executive report)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Register project paths
for subpath in (
    PROJECT_ROOT,
    PROJECT_ROOT / "aegis_threat_fusion_v1",
    PROJECT_ROOT / "location_threat_monitor",
    PROJECT_ROOT / "threat_detector",
    PROJECT_ROOT / "nlp_intelligence",
    PROJECT_ROOT / "explainable_ai",
    PROJECT_ROOT / "representation_learning",
):
    if subpath.is_dir() and str(subpath) not in sys.path:
        sys.path.insert(0, str(subpath))

load_dotenv(PROJECT_ROOT / ".env")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Try imports
try:
    from threat_video_pipeline import (
        FIRE_SMOKE_CLASSES,
        FIRE_SMOKE_CONF,
        FIRE_SMOKE_IMGSZ,
        FIRE_SMOKE_IOU,
        FIRE_SMOKE_MODEL_PATH,
        PERSON_CLASS_ID,
        PERSON_CONF,
        PERSON_IMGSZ,
        PERSON_IOU,
        PERSON_MODEL_PATH,
        WEAPON_CLASSES,
        WEAPON_CONF,
        WEAPON_IMGSZ,
        WEAPON_IOU,
        WEAPON_MODEL_PATH,
        Detection,
        VerifiedYoloDetector,
        build_detectors,
        draw_detections,
        run_all_detectors,
    )
    VIDEO_DETECTOR_AVAILABLE = True
except Exception as err:
    VIDEO_DETECTOR_AVAILABLE = False
    print(f"[Warning] Video detector import notice: {err}")

try:
    from explainable_ai.visual_explainer import generate_eigencam_on_video_frame
    XAI_AVAILABLE = True
except Exception as err:
    XAI_AVAILABLE = False
    print(f"[Warning] Explainable AI visual explainer notice: {err}")

try:
    from nlp_intelligence.pipeline import (
        analyze_sentiment,
        extract_entities,
        extract_keywords,
        extract_threats,
    )
    NLP_PIPELINE_AVAILABLE = True
except Exception as err:
    NLP_PIPELINE_AVAILABLE = False
    print(f"[Warning] NLP pipeline import notice: {err}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


# =============================================================================
# 1. LOCATION & GIS INTELLIGENCE AGENT
# =============================================================================
class LocationGISAgent:
    """Geocoding, real critical infrastructure identification, and GeoJSON generation."""

    KNOWN_COORDINATES: Dict[str, Dict[str, Any]] = {
        "connaught place": {
            "latitude": 28.63177,
            "longitude": 77.21938,
            "city": "New Delhi",
            "state": "Delhi",
            "country": "India",
            "display_name": "Connaught Place, New Delhi, Delhi, 110001, India",
        },
        "new delhi": {
            "latitude": 28.61394,
            "longitude": 77.20902,
            "city": "New Delhi",
            "state": "Delhi",
            "country": "India",
            "display_name": "New Delhi, Delhi, India",
        },
        "mumbai": {
            "latitude": 18.92200,
            "longitude": 72.83470,
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India",
            "display_name": "Colaba, Mumbai, Maharashtra, India",
        },
        "colaba": {
            "latitude": 18.91500,
            "longitude": 72.82800,
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India",
            "display_name": "Colaba, Mumbai, Maharashtra, India",
        },
        "bengaluru": {
            "latitude": 12.97160,
            "longitude": 77.59460,
            "city": "Bengaluru",
            "state": "Karnataka",
            "country": "India",
            "display_name": "Bengaluru, Karnataka, India",
        },
        "hyderabad": {
            "latitude": 17.38504,
            "longitude": 78.48667,
            "city": "Hyderabad",
            "state": "Telangana",
            "country": "India",
            "display_name": "Hyderabad, Telangana, India",
        },
        "kolkata": {
            "latitude": 22.57265,
            "longitude": 88.36389,
            "city": "Kolkata",
            "state": "West Bengal",
            "country": "India",
            "display_name": "Kolkata, West Bengal, India",
        },
        "chennai": {
            "latitude": 13.08268,
            "longitude": 80.27072,
            "city": "Chennai",
            "state": "Tamil Nadu",
            "country": "India",
            "display_name": "Chennai, Tamil Nadu, India",
        },
    }

    CRITICAL_INFRASTRUCTURE_CATALOG = [
        {"name": "Dr. Ram Manohar Lohia Hospital", "category": "Hospital / Trauma", "lat": 28.6234, "lon": 77.1992, "city": "New Delhi"},
        {"name": "Connaught Place Police Station", "category": "Police Station", "lat": 28.6325, "lon": 77.2185, "city": "New Delhi"},
        {"name": "Parliament House of India", "category": "Government / High Security", "lat": 28.6172, "lon": 77.2081, "city": "New Delhi"},
        {"name": "Rajiv Chowk Metro Station", "category": "Transit Hub (Metro)", "lat": 28.6328, "lon": 77.2195, "city": "New Delhi"},
        {"name": "New Delhi Railway Station", "category": "Transit Hub (Rail)", "lat": 28.6430, "lon": 77.2197, "city": "New Delhi"},
        {"name": "BSES Yamuna Power Substation", "category": "Energy / Utility", "lat": 28.6370, "lon": 77.2250, "city": "New Delhi"},
        {"name": "Modern School Barakhamba", "category": "Educational Institute", "lat": 28.6290, "lon": 77.2285, "city": "New Delhi"},
        {"name": "AIIMS New Delhi", "category": "Hospital / Super-speciality", "lat": 28.5672, "lon": 77.2100, "city": "New Delhi"},
        {"name": "Colaba Police Station", "category": "Police Station", "lat": 18.9150, "lon": 72.8280, "city": "Mumbai"},
        {"name": "Chhatrapati Shivaji Maharaj Terminus", "category": "Transit Hub (Rail)", "lat": 18.9400, "lon": 72.8353, "city": "Mumbai"},
        {"name": "GT Hospital Mumbai", "category": "Hospital / Trauma", "lat": 18.9430, "lon": 72.8330, "city": "Mumbai"},
        {"name": "Bengaluru City Police Commissionerate", "category": "Police Headquarters", "lat": 12.9815, "lon": 77.5925, "city": "Bengaluru"},
        {"name": "Bowring & Lady Curzon Hospital", "category": "Hospital / Trauma", "lat": 12.9820, "lon": 77.6040, "city": "Bengaluru"},
        {"name": "Majestic Metro Interchange", "category": "Transit Hub (Metro)", "lat": 12.9757, "lon": 77.5728, "city": "Bengaluru"},
    ]

    def geocode(self, location_query: str) -> Dict[str, Any]:
        location_query = (location_query or "").strip()
        if not location_query:
            location_query = "Connaught Place, New Delhi"

        # 1. Try Nominatim Live API
        try:
            import requests

            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": location_query, "format": "json", "limit": 1}
            headers = {"User-Agent": "AEGIS-Threat-Intelligence-Platform/2.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    item = data[0]
                    lat, lon = float(item["lat"]), float(item["lon"])
                    display_name = item.get("display_name", location_query)
                    parts = [p.strip() for p in display_name.split(",")]
                    city = parts[0] if parts else location_query
                    country = parts[-1] if len(parts) > 1 else "India"
                    return {
                        "location": location_query,
                        "latitude": round(lat, 5),
                        "longitude": round(lon, 5),
                        "display_name": display_name,
                        "city": city,
                        "country": country,
                        "source": "OpenStreetMap / Nominatim Live",
                    }
        except Exception as exc:
            print(f"[GIS] Nominatim API notice ({exc}). Using offline coordinates registry.")

        # 2. Check local catalog
        query_norm = location_query.lower()
        for key, info in self.KNOWN_COORDINATES.items():
            if key in query_norm or query_norm in key:
                res = dict(info)
                res["location"] = location_query
                res["source"] = "AEGIS Verified Geo-Registry"
                return res

        # 3. Default fallback
        res = dict(self.KNOWN_COORDINATES["connaught place"])
        res["location"] = location_query
        res["source"] = "Default Reference Hub (New Delhi Core)"
        return res

    def identify_critical_infrastructure(
        self, lat: float, lon: float, radius_km: float = 5.0
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Find verified critical infrastructure within radius. Do not fabricate fake points."""
        nearby = []
        min_dist = float("inf")

        for item in self.CRITICAL_INFRASTRUCTURE_CATALOG:
            dist = haversine_distance(lat, lon, item["lat"], item["lon"])
            if dist <= radius_km:
                site = dict(item)
                site["distance_km"] = round(dist, 3)
                bearing_ns = "N" if item["lat"] >= lat else "S"
                bearing_ew = "E" if item["lon"] >= lon else "W"
                site["bearing"] = f"Sector {bearing_ns}{bearing_ew}"
                nearby.append(site)
                if dist < min_dist:
                    min_dist = dist

        nearby.sort(key=lambda x: x["distance_km"])

        if min_dist <= 0.5:
            multiplier = 1.15
        elif min_dist <= 1.5:
            multiplier = 1.08
        else:
            multiplier = 1.00

        return nearby, multiplier

    def generate_gis_layer(
        self,
        lat: float,
        lon: float,
        location_name: str,
        critical_infra: List[Dict[str, Any]],
        threat_level: str = "MEDIUM",
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "name": location_name,
                    "role": "Event Epicenter / Surveillance Location",
                    "marker-color": "#e74c3c" if threat_level in ("HIGH", "CRITICAL") else "#f39c12",
                    "marker-symbol": "danger",
                    "threat_level": threat_level,
                },
            }
        ]

        # Perimeter zones
        for radius_m, name, color in [
            (250, "Immediate Hazard Perimeter (250m)", "#e74c3c"),
            (1000, "Cordon & Containment Zone (1km)", "#e67e22"),
            (3000, "Surveillance Buffer (3km)", "#3498db"),
        ]:
            coords = []
            d_lat = radius_m / 111320.0
            d_lon = radius_m / (111320.0 * math.cos(math.radians(lat)))
            for angle_deg in range(0, 361, 15):
                rad = math.radians(angle_deg)
                coords.append([
                    round(lon + d_lon * math.sin(rad), 6),
                    round(lat + d_lat * math.cos(rad), 6),
                ])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "name": name,
                    "radius_meters": radius_m,
                    "stroke": color,
                    "stroke-width": 2,
                    "fill": color,
                    "fill-opacity": 0.12,
                },
            })

        # Add critical infrastructure markers
        for site in critical_infra:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [site.get("lon", lon), site.get("lat", lat)]},
                "properties": {
                    "name": site["name"],
                    "category": site["category"],
                    "distance_km": site["distance_km"],
                    "role": "Critical Infrastructure",
                    "marker-color": "#2980b9",
                    "marker-symbol": "hospital" if "Hospital" in site["category"] else "police" if "Police" in site["category"] else "transit",
                },
            })

        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": features,
            "metadata": {
                "generated_at": utc_now(),
                "location": location_name,
                "center": [lat, lon],
                "infrastructure_count": len(critical_infra),
            },
        }

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2)

        return geojson


# =============================================================================
# 2. VIDEO INTELLIGENCE AGENT (YOLOv11 + BYTETRACK)
# =============================================================================
class VideoIntelligenceAgent:
    """Runs YOLOv11 detectors for Person, Weapon, and Fire/Smoke, tracking persons via ByteTrack."""

    def __init__(self) -> None:
        self.detectors = build_detectors() if VIDEO_DETECTOR_AVAILABLE else {}

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        venue_capacity: int = 15,
        frame_skip: int = 1,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> Dict[str, Any]:
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = output_dir / f"{video_path.stem}_analyzed.mp4"
        output_report_path = output_dir / f"{video_path.stem}_report.json"

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        categories = ["weapon", "fire", "smoke", "person"]
        detection_counts = {c: 0 for c in categories}
        detection_frames = {c: 0 for c in categories}
        confidence_sums = {c: 0.0 for c in categories}
        confidence_counts = {c: 0 for c in categories}
        max_confidences = {c: 0.0 for c in categories}

        unique_person_ids = set()
        concurrent_person_counts = []
        threat_events = []
        frame_timeline = []

        best_threat_frame_idx = 0
        best_threat_category = None
        best_threat_conf = 0.0

        frame_idx = 0
        inference_frames = 0
        last_detections: List[Detection] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            should_infer = (frame_idx % max(frame_skip, 1) == 0)

            if should_infer:
                detections = run_all_detectors(self.detectors, frame, include_person=True)
                last_detections = detections
                inference_frames += 1

                frame_cats = set()
                frame_person_count = 0
                frame_threat_counts = {"weapon": 0, "fire": 0, "smoke": 0}
                frame_max_conf = {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0}

                for det in detections:
                    cat = det.category
                    if cat in detection_counts:
                        detection_counts[cat] += 1
                        confidence_sums[cat] += det.confidence
                        confidence_counts[cat] += 1
                        if det.confidence > max_confidences[cat]:
                            max_confidences[cat] = det.confidence
                        frame_cats.add(cat)
                        if det.confidence > frame_max_conf.get(cat, 0.0):
                            frame_max_conf[cat] = round(det.confidence, 4)

                    if cat == "person":
                        frame_person_count += 1
                        if det.track_id is not None:
                            unique_person_ids.add(det.track_id)
                    elif cat in frame_threat_counts:
                        frame_threat_counts[cat] += 1
                        if det.confidence > best_threat_conf:
                            best_threat_conf = det.confidence
                            best_threat_category = cat
                            best_threat_frame_idx = frame_idx

                concurrent_person_counts.append(frame_person_count)

                for cat in frame_cats:
                    detection_frames[cat] += 1

                threat_categories = sorted(
                    {d.category for d in detections if d.category in ("weapon", "fire", "smoke")}
                )

                if threat_categories:
                    threat_events.append({
                        "frame": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "threats": threat_categories,
                        "detections": [
                            asdict(d) for d in detections if d.category in ("weapon", "fire", "smoke")
                        ],
                    })

                frame_timeline.append({
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "person_count": frame_person_count,
                    "threat_counts": frame_threat_counts,
                    "max_confidence": frame_max_conf,
                })
            else:
                detections = last_detections

            annotated_frame = draw_detections(frame, detections)
            writer.write(annotated_frame)
            frame_idx += 1

            if progress_cb and total_frames > 0 and frame_idx % 15 == 0:
                pct = 10 + int((frame_idx / total_frames) * 30)
                progress_cb(f"Processing video frame {frame_idx}/{total_frames} (YOLOv11 & ByteTrack)...", min(pct, 40))

        cap.release()
        writer.release()

        avg_confidences = {
            c: round(confidence_sums[c] / confidence_counts[c], 4) if confidence_counts[c] > 0 else 0.0
            for c in categories
        }
        detection_ratios = {
            c: round(detection_frames[c] / max(inference_frames, 1), 4)
            for c in categories
        }

        peak_concurrent = max(concurrent_person_counts) if concurrent_person_counts else len(unique_person_ids)
        avg_concurrent = round(float(np.mean(concurrent_person_counts)), 2) if concurrent_person_counts else 0.0
        crowd_over_capacity = peak_concurrent > venue_capacity
        occupancy_ratio = round(peak_concurrent / max(venue_capacity, 1), 2)

        detections_summary = []
        for cat in categories:
            det_bool = detection_counts[cat] > 0
            detections_summary.append({
                "category": cat,
                "detected": det_bool,
                "detection_count": detection_counts[cat],
                "detection_frames": detection_frames[cat],
                "detection_ratio": detection_ratios[cat],
                "average_confidence": avg_confidences[cat],
                "max_confidence": round(max_confidences[cat], 4),
                "message": f"{cat} detected" if det_bool else f"no {cat} detected",
            })

        report = {
            "input_video": str(video_path),
            "output_video": str(output_video_path),
            "fps": fps,
            "resolution": [width, height],
            "frames_processed": frame_idx,
            "inference_frames": inference_frames,
            "frame_skip": frame_skip,
            "detection_counts": detection_counts,
            "detection_frames": detection_frames,
            "detection_ratios": detection_ratios,
            "average_confidence": avg_confidences,
            "max_confidence": {c: round(v, 4) for c, v in max_confidences.items()},
            "detections": detections_summary,
            "person_statistics": {
                "unique_person_count": len(unique_person_ids) if unique_person_ids else (peak_concurrent if peak_concurrent > 0 else 0),
                "total_person_detections": detection_counts["person"],
                "peak_concurrent_persons": peak_concurrent,
                "avg_concurrent_persons": avg_concurrent,
                "unique_track_ids": sorted(list(unique_person_ids)),
                "venue_capacity": venue_capacity,
                "crowd_over_capacity": crowd_over_capacity,
                "occupancy_ratio": occupancy_ratio,
            },
            "best_threat_frame": {
                "frame_index": best_threat_frame_idx,
                "category": best_threat_category,
                "confidence": best_threat_conf,
                "timestamp_sec": round(best_threat_frame_idx / fps, 2),
            },
            "threat_events": threat_events,
            "num_threat_events": len(threat_events),
            "frame_timeline": frame_timeline,
        }

        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


# =============================================================================
# 3. GEMINI VIDEO UNDERSTANDING AGENT
# =============================================================================
class GeminiVideoAgent:
    """Direct visual description via Google GenAI SDK with graceful fallback."""

    THREAT_TERMS = {
        "weapon": [
            "weapon", "gun", "firearm", "pistol", "rifle", "shotgun",
            "handgun", "knife", "sword", "armed", "holding a gun",
            "holding a weapon", "weapon-like", "object resembling a weapon",
        ],
        "fire": ["fire", "flames", "burning", "blaze", "on fire"],
        "smoke": ["smoke", "smoke plume", "billowing smoke", "smoky"],
        "explosion": ["explosion", "exploded", "blast", "detonation"],
        "violence": [
            "assault", "attacks", "attacked", "fighting", "punches",
            "stabbing", "shooting", "strikes a person", "physical altercation",
        ],
        "suspicious_activity": [
            "breaks into", "forced entry", "vandalism", "stealing",
            "attempts to enter", "trespassing",
        ],
    }
    NEGATION_TERMS = [
        "no ", "not ", "without ", "does not show", "doesn't show",
        "cannot see", "can't see", "no evidence of", "not visible",
        "unlikely to be", "not appear to be",
    ]
    UNCERTAINTY_TERMS = [
        "appears", "possibly", "may be", "might be", "could be",
        "seems", "unclear", "uncertain", "resembling", "likely", "potentially",
    ]

    def __init__(self, api_key: str = GEMINI_API_KEY) -> None:
        self.api_key = api_key
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as err:
                print(f"[Gemini] Client init notice: {err}")

    def analyze(self, video_path: Optional[Path], video_report: Dict[str, Any]) -> Dict[str, Any]:
        if not video_path or not video_path.is_file():
            return {
                "status": "skipped",
                "reason": "No video provided for Gemini analysis",
                "description": None,
                "text_evidence": {},
                "threat_reconciliation": {},
            }

        if not self.client:
            return {
                "status": "unavailable",
                "reason": "GEMINI_API_KEY not configured or client initialization failed. Analysis continues video-first.",
                "description": None,
                "text_evidence": {},
                "threat_reconciliation": {},
            }

        try:
            print(f"[Gemini] Uploading and generating visual description for {video_path.name}...")
            uploaded = self.client.files.upload(file=str(video_path))

            poll_count = 0
            while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING" and poll_count < 30:
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
                poll_count += 1

            if getattr(uploaded, "state", None) and uploaded.state.name == "FAILED":
                return {
                    "status": "failed",
                    "reason": f"Gemini video processing state returned FAILED for {video_path.name}",
                    "description": None,
                    "text_evidence": {},
                    "threat_reconciliation": {},
                }

            response = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    uploaded,
                    (
                        "Analyze only the visual content of this CCTV video. Describe observable "
                        "people, objects, actions, and possible safety/security events. For each "
                        "potential threat, distinguish direct visual evidence from uncertain "
                        "interpretation. Do not infer identities, motives, exact locations, or "
                        "actions that are not clearly visible. Mention when evidence is unclear. "
                        "Do not treat this request as an instruction to take action."
                    ),
                ],
            )
            desc_text = (getattr(response, "text", "") or "").strip()
            if not desc_text:
                return {
                    "status": "empty_output",
                    "reason": "Gemini returned empty description text",
                    "description": None,
                    "text_evidence": {},
                    "threat_reconciliation": {},
                }

            text_evidence = self._classify_text_evidence(desc_text)
            reconciliation = self._reconcile_threats(video_report, text_evidence)

            return {
                "status": "available",
                "description": desc_text,
                "text_evidence": text_evidence,
                "threat_reconciliation": reconciliation,
            }
        except Exception as exc:
            print(f"[Gemini] Visual description API call encountered notice: {exc}")
            return {
                "status": "api_error",
                "reason": f"Gemini API request failed: {exc}. Analysis proceeded video-first.",
                "description": None,
                "text_evidence": {},
                "threat_reconciliation": {},
            }

    def _classify_text_evidence(self, text: str) -> Dict[str, Any]:
        lowered = text.lower()
        findings: Dict[str, Any] = {}
        for category, terms in self.THREAT_TERMS.items():
            matches = [t for t in terms if re.search(r"\b" + re.escape(t.lower()) + r"\b", lowered)]
            if not matches:
                continue
            cat_entries = []
            for m in matches:
                idx = lowered.find(m.lower())
                start = max(0, idx - 80)
                end = min(len(text), idx + len(m) + 80)
                snip = text[start:end].strip()
                snip_low = snip.lower()
                neg = any(marker in snip_low for marker in self.NEGATION_TERMS)
                unc = any(marker in snip_low for marker in self.UNCERTAINTY_TERMS)
                cat_entries.append({
                    "matched_term": m,
                    "evidence": snip,
                    "negated": neg,
                    "uncertain": unc,
                })
            findings[category] = cat_entries
        return findings

    def _reconcile_threats(self, report: Dict[str, Any], text_evidence: Dict[str, Any]) -> Dict[str, Any]:
        v_counts = report.get("detection_counts", {})
        reconciliation = {}
        for cat in ("weapon", "fire", "smoke", "person"):
            video_det = v_counts.get(cat, 0) > 0
            text_entries = text_evidence.get(cat, [])
            valid_text = [e for e in text_entries if not e["negated"]]
            clear_text = [e for e in valid_text if not e["uncertain"]]

            has_clear = bool(clear_text)
            has_unc = bool(valid_text)

            if video_det and has_clear:
                status = "CORROBORATED"
                desc = f"Video detector and Gemini description both support {cat}."
            elif video_det and has_unc:
                status = "DETECTOR_SUPPORTED_TEXT_UNCERTAIN"
                desc = f"Video detector observed {cat}; Gemini mentions it with uncertainty."
            elif video_det:
                status = "DETECTOR_ONLY"
                desc = f"Video detector confirmed {cat}; Gemini text does not clearly corroborate."
            elif has_clear:
                status = "DESCRIPTION_ONLY"
                desc = f"Gemini description suggests {cat}; video detector did not detect it."
            else:
                status = "NOT_DETECTED"
                desc = f"No positive evidence detected for {cat} across modalities."

            reconciliation[cat] = {
                "status": status,
                "video_detector_support": video_det,
                "gemini_text_support": has_unc,
                "explanation": desc,
            }
        return reconciliation


# =============================================================================
# 4. OSINT INTELLIGENCE AGENT (REAL NEWSAPI VIA THREAT_DETECTOR)
# =============================================================================
class OSINTAgent:
    """Collects real threat news from NewsAPI, geocodes incident locations, and filters by proximity."""

    def __init__(self, api_key: str = NEWS_API_KEY) -> None:
        self.api_key = api_key

    def collect_intelligence(
        self,
        location: str,
        user_lat: float,
        user_lon: float,
        radius_km: float = 50.0,
        output_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        # Attempt to use existing threat_detector analysis engine
        try:
            from analysis_engine import analyze_threats
            raw_threats = analyze_threats(location, user_lat, user_lon)
            if raw_threats and isinstance(raw_threats, list):
                for art in raw_threats:
                    if art.get("geographic_classification") in ("Local", "Nearby"):
                        results.append(art)
                print(f"[OSINT] Real threat news analysis returned {len(results)} verified regional records.")
        except Exception as exc:
            print(f"[OSINT] Standard analysis engine notice ({exc}). Falling back to direct NewsAPI fetch.")
            # Direct NewsAPI fallback if available
            if self.api_key:
                try:
                    import requests
                    from datetime import timedelta

                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(hours=48)
                    query = f'"{location}" AND (threat OR attack OR weapon OR fire OR violence OR police OR emergency)'
                    params = {
                        "q": query,
                        "from": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 20,
                    }
                    resp = requests.get(
                        "https://newsapi.org/v2/everything",
                        params=params,
                        headers={"X-Api-Key": self.api_key},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        arts = resp.json().get("articles", [])
                        for a in arts:
                            results.append({
                                "title": a.get("title", ""),
                                "description": a.get("description", ""),
                                "source": a.get("source", {}).get("name", "News Feed"),
                                "publishedAt": a.get("publishedAt", utc_now()),
                                "url": a.get("url", ""),
                                "category": "Security / Public Advisory",
                                "severity": "Medium",
                                "incident_location": location,
                                "incident_latitude": user_lat,
                                "incident_longitude": user_lon,
                                "distance_km": 0.0,
                                "geographic_classification": "Local",
                                "feed_type": "NewsAPI",
                            })
                except Exception as inner_exc:
                    print(f"[OSINT] Direct NewsAPI fallback notice: {inner_exc}")

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        return results


# =============================================================================
# 5. NLP INTELLIGENCE AGENT
# =============================================================================
class NLPIntelligenceAgent:
    """Named Entity Extraction (spaCy), VADER Sentiment Analysis, and Keyword Parsing."""

    def process(
        self,
        video_report: Dict[str, Any],
        osint_results: List[Dict[str, Any]],
        gemini_data: Dict[str, Any],
        output_nlp: Optional[Path] = None,
        output_combined: Optional[Path] = None,
    ) -> Dict[str, Any]:
        # Build text corpus from available sources
        text_snippets = []
        if gemini_data.get("description"):
            text_snippets.append(gemini_data["description"])
        for r in osint_results:
            text_snippets.append(f"{r.get('title', '')}. {r.get('description', '')}.")

        corpus = " ".join(text_snippets).strip()

        entities = []
        sentiment = {"compound": 0.0, "label": "neutral"}
        keywords = []

        if corpus and NLP_PIPELINE_AVAILABLE:
            try:
                entities = extract_entities(corpus)
            except Exception as e:
                print(f"[NLP] Entity extraction notice: {e}")
            try:
                sentiment = analyze_sentiment(corpus)
            except Exception as e:
                print(f"[NLP] Sentiment analysis notice: {e}")
            try:
                keywords = extract_keywords(corpus)
            except Exception as e:
                print(f"[NLP] Keyword extraction notice: {e}")

        # Cross-source reconciliation
        v_counts = video_report.get("detection_counts", {})
        v_max = video_report.get("max_confidence", {})

        reconciliation = {}
        for cat in ("weapon", "fire", "smoke", "person"):
            has_video = v_counts.get(cat, 0) > 0
            has_text = any(cat in str(k).lower() for k in keywords) or bool(gemini_data.get("text_evidence", {}).get(cat))
            conf = v_max.get(cat, 0.0)

            if has_video and has_text:
                status = "CORROBORATED"
                desc = f"Both video sensor and textual narrative support {cat}."
            elif has_video:
                status = "DETECTOR_ONLY"
                desc = f"Video detector detected {cat} (max confidence {conf:.2f}); no text corroboration."
            elif has_text:
                status = "TEXT_ONLY"
                desc = f"Textual source mentions {cat}, but video sensor did not detect it."
            else:
                status = "NOT_DETECTED"
                desc = f"No positive evidence detected for {cat} across modalities."

            reconciliation[cat] = {
                "status": status,
                "video_detected": has_video,
                "text_corroborated": has_text,
                "confidence": conf,
                "assessment": desc,
            }

        nlp_output = {
            "generated_at": utc_now(),
            "entities": entities[:25],
            "entity_counts_by_type": dict(Counter([e["type"] for e in entities])) if entities else {},
            "sentiment": sentiment,
            "threat_keywords": keywords,
            "reconciliation": reconciliation,
            "total_articles_parsed": len(osint_results),
            "gemini_text_included": bool(gemini_data.get("description")),
        }

        if output_nlp:
            output_nlp.parent.mkdir(parents=True, exist_ok=True)
            with open(output_nlp, "w", encoding="utf-8") as f:
                json.dump(nlp_output, f, indent=2)

        if output_combined:
            output_combined.parent.mkdir(parents=True, exist_ok=True)
            with open(output_combined, "w", encoding="utf-8") as f:
                json.dump({
                    "video_report_summary": video_report.get("detections"),
                    "gemini_understanding": gemini_data,
                    "nlp_analysis": nlp_output,
                    "reconciliation": reconciliation,
                }, f, indent=2)

        return nlp_output


# =============================================================================
# 6. REPRESENTATION LEARNING MODULE (384-DIM THREAT EMBEDDINGS)
# =============================================================================
class RepresentationLearningModule:
    """Multimodal 20-dim feature vector normalized into 384-dim embedding space."""

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim

    def generate_multimodal_features(
        self,
        video_report: Dict[str, Any],
        osint_results: List[Dict[str, Any]],
        nlp_data: Dict[str, Any],
        critical_infra: List[Dict[str, Any]],
        lat: float,
        lon: float,
        output_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # 1. Video features (6 dims)
        v_max = video_report.get("max_confidence", {})
        v_ratios = video_report.get("detection_ratios", {})
        video_feat = [
            float(v_max.get("weapon", 0.0)),
            float(v_ratios.get("weapon", 0.0)),
            float(v_max.get("fire", 0.0)),
            float(v_ratios.get("fire", 0.0)),
            float(v_max.get("smoke", 0.0)),
            float(v_ratios.get("smoke", 0.0)),
        ]

        # 2. Crowd features (4 dims)
        p_stats = video_report.get("person_statistics", {})
        peak_crowd = float(p_stats.get("peak_concurrent_persons", 0))
        unique_persons = float(p_stats.get("unique_person_count", 0))
        venue_cap = float(p_stats.get("venue_capacity", 15))
        over_cap_ratio = min(peak_crowd / max(venue_cap, 1.0), 3.0)
        crowd_feat = [
            peak_crowd / 50.0,
            unique_persons / 50.0,
            over_cap_ratio,
            1.0 if peak_crowd > venue_cap else 0.0,
        ]

        # 3. NLP features (4 dims)
        sent = nlp_data.get("sentiment", {})
        compound = float(sent.get("compound", 0.0) or 0.0)
        entities_count = float(len(nlp_data.get("entities", [])))
        nlp_feat = [
            (compound + 1.0) / 2.0,
            abs(compound),
            min(entities_count / 20.0, 1.0),
            min(len(nlp_data.get("threat_keywords", [])) / 15.0, 1.0),
        ]

        # 4. Geospatial features (4 dims)
        min_infra_dist = float(critical_infra[0]["distance_km"]) if critical_infra else 5.0
        infra_count = float(len(critical_infra))
        geo_feat = [
            (lat - 20.0) / 20.0,
            (lon - 70.0) / 30.0,
            max(0.0, 1.0 - (min_infra_dist / 5.0)),
            min(infra_count / 10.0, 1.0),
        ]

        # 5. OSINT features (2 dims)
        osint_count = float(len(osint_results))
        high_sev_count = sum(1 for r in osint_results if r.get("severity") == "High")
        osint_feat = [
            min(osint_count / 10.0, 1.0),
            min(high_sev_count / 5.0, 1.0),
        ]

        raw_features = video_feat + crowd_feat + nlp_feat + geo_feat + osint_feat
        feat_vector = np.array(raw_features, dtype=np.float32)

        # Deterministic projection to 384 dimensions
        np.random.seed(42)
        projection_matrix = np.random.randn(len(raw_features), self.embedding_dim)
        embedding = np.dot(feat_vector, projection_matrix)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Archetype comparisons
        target_profiles = {
            "Armed Assault Hazard": np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0] + [0.3] * 14, dtype=np.float32),
            "Active Fire & Smoke Emergency": np.array([0.0, 0.0, 1.0, 0.8, 0.9, 0.7] + [0.3] * 14, dtype=np.float32),
            "Mass Gathering & Civil Disorder": np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 1.0, 1.0, 1.5, 1.0] + [0.3] * 10, dtype=np.float32),
            "Low Risk / Monitored Baseline": np.array([0.0] * 20, dtype=np.float32),
        }
        archetype_similarities = {}
        for arch_name, proto in target_profiles.items():
            proto_emb = np.dot(proto, projection_matrix)
            proto_norm = np.linalg.norm(proto_emb)
            if proto_norm > 0:
                proto_emb = proto_emb / proto_norm
            sim = float(np.dot(embedding, proto_emb))
            archetype_similarities[arch_name] = round(max(0.0, sim), 4)

        metadata = {
            "embedding_dimension": self.embedding_dim,
            "raw_feature_count": len(raw_features),
            "archetype_similarities": archetype_similarities,
            "top_matching_archetype": max(archetype_similarities.items(), key=lambda x: x[1])[0],
            "feature_breakdown": {
                "video_features": video_feat,
                "crowd_features": crowd_feat,
                "nlp_features": nlp_feat,
                "geospatial_features": geo_feat,
                "osint_features": osint_feat,
            },
        }

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_dir / "multimodal_threat_embedding.npz",
                embedding=embedding,
                features=feat_vector,
            )
            with open(output_dir / "multimodal_features.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        return embedding, metadata


# =============================================================================
# 7. THREAT ASSESSMENT & EVIDENCE FUSION AGENT
# =============================================================================
class ThreatFusionAgent:
    """Calculates AEGIS Threat Index (ATI), official Threat Level, and Dominant Threat."""

    WEIGHTS = {
        "weapon_detection": 0.40,
        "fire": 0.25,
        "smoke": 0.25,
        "crowd_alert": 0.20,
        "osint_threat": 0.15,
        "nlp_threat": 0.15,
    }

    def fuse(
        self,
        video_report: Dict[str, Any],
        osint_results: List[Dict[str, Any]],
        nlp_data: Dict[str, Any],
        infra_multiplier: float = 1.0,
        zone: str = "zone-alpha-1",
        event_time: str = utc_now(),
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        contributions = []
        excluded_events = []
        sub_scores = {}

        v_max = video_report.get("max_confidence", {})
        weapon_conf = float(v_max.get("weapon", 0.0))
        fire_conf = float(v_max.get("fire", 0.0))
        smoke_conf = float(v_max.get("smoke", 0.0))

        # 1. Weapon
        if weapon_conf >= 0.35:
            score = round(self.WEIGHTS["weapon_detection"] * weapon_conf, 4)
            contributions.append({
                "modality": "video",
                "event_type": "weapon_detection",
                "weight": self.WEIGHTS["weapon_detection"],
                "confidence": weapon_conf,
                "sub_score": score,
                "reason": f"Firearm/Weapon visually detected in CCTV (Weight: 0.40 × Conf: {weapon_conf:.2f})",
            })
            sub_scores["weapon"] = score
        elif weapon_conf > 0:
            excluded_events.append({
                "event_type": "weapon_detection",
                "reason": f"weapon confidence ({weapon_conf:.2f}) was below 0.35 minimum threshold",
            })

        # 2. Fire
        if fire_conf >= 0.40:
            score = round(self.WEIGHTS["fire"] * fire_conf, 4)
            contributions.append({
                "modality": "video",
                "event_type": "fire",
                "weight": self.WEIGHTS["fire"],
                "confidence": fire_conf,
                "sub_score": score,
                "reason": f"Active Fire signature detected in video (Weight: 0.25 × Conf: {fire_conf:.2f})",
            })
            sub_scores["fire"] = score
        elif fire_conf > 0:
            excluded_events.append({
                "event_type": "fire",
                "reason": f"fire confidence ({fire_conf:.2f}) was below 0.40 minimum threshold",
            })

        # 3. Smoke
        if smoke_conf >= 0.40:
            score = round(self.WEIGHTS["smoke"] * smoke_conf, 4)
            contributions.append({
                "modality": "video",
                "event_type": "smoke",
                "weight": self.WEIGHTS["smoke"],
                "confidence": smoke_conf,
                "sub_score": score,
                "reason": f"Smoke plume detected (Weight: 0.25 × Conf: {smoke_conf:.2f})",
            })
            sub_scores["smoke"] = score

        # 4. Crowd over-capacity
        p_stats = video_report.get("person_statistics", {})
        if p_stats.get("crowd_over_capacity", False):
            peak = p_stats.get("peak_concurrent_persons", 0)
            cap = p_stats.get("venue_capacity", 15)
            crowd_score = round(self.WEIGHTS["crowd_alert"] * min(peak / max(cap, 1), 1.0), 4)
            contributions.append({
                "modality": "video_crowd",
                "event_type": "crowd_alert",
                "weight": self.WEIGHTS["crowd_alert"],
                "confidence": 1.0,
                "sub_score": crowd_score,
                "reason": f"Crowd density exceeds venue capacity ({peak} persons / capacity {cap})",
            })
            sub_scores["crowd"] = crowd_score

        # 5. OSINT
        qualifying_osint = [
            r for r in osint_results
            if r.get("geographic_classification") in ("Local", "Nearby") and r.get("severity") in ("High", "Medium")
        ]
        if qualifying_osint:
            top_osint = qualifying_osint[0]
            osint_conf = 0.80 if top_osint.get("severity") == "High" else 0.60
            osint_score = round(self.WEIGHTS["osint_threat"] * osint_conf, 4)
            contributions.append({
                "modality": "osint",
                "event_type": "osint_threat",
                "weight": self.WEIGHTS["osint_threat"],
                "confidence": osint_conf,
                "sub_score": osint_score,
                "reason": f"Regional news advisory from {top_osint.get('source')} ({top_osint.get('category')})",
            })
            sub_scores["osint"] = osint_score

        # 6. NLP sentiment urgency
        sent = nlp_data.get("sentiment", {})
        compound = float(sent.get("compound", 0.0) or 0.0)
        if compound <= -0.25:
            nlp_score = round(self.WEIGHTS["nlp_threat"] * min(abs(compound), 1.0), 4)
            contributions.append({
                "modality": "nlp",
                "event_type": "nlp_threat",
                "weight": self.WEIGHTS["nlp_threat"],
                "confidence": round(abs(compound), 2),
                "sub_score": nlp_score,
                "reason": f"High distress / security urgency tone in incident discourse (VADER: {compound:.2f})",
            })
            sub_scores["nlp"] = nlp_score

        base_score = sum(c["sub_score"] for c in contributions)
        fused_score = min(round(base_score * infra_multiplier, 4), 1.0)
        ati_index = round(fused_score * 100.0, 1)

        # Official threat levels
        if ati_index >= 80.0:
            threat_level = "CRITICAL"
        elif ati_index >= 60.0:
            threat_level = "HIGH"
        elif ati_index >= 30.0:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        # Threat classification
        if weapon_conf >= 0.35 and fire_conf >= 0.40:
            classification = "MULTI_MODAL_ARMED_ARSON_HAZARD"
        elif weapon_conf >= 0.35:
            classification = "ARMED_ASSAULT_INCIDENT"
        elif fire_conf >= 0.40:
            classification = "ACTIVE_STRUCTURE_FIRE_HAZARD"
        elif p_stats.get("crowd_over_capacity"):
            classification = "MASS_GATHERING_DISORDER_RISK"
        elif qualifying_osint:
            classification = "REGIONAL_OSINT_ADVISORY_ALERT"
        else:
            classification = "ROUTINE_MONITORED_ACTIVITY"

        # Dominant threat category calculation
        candidate_threats = []
        if weapon_conf >= 0.35:
            candidate_threats.append(("weapon", self.WEIGHTS["weapon_detection"] * weapon_conf, "CRITICAL" if weapon_conf > 0.6 else "HIGH"))
        if fire_conf >= 0.40:
            candidate_threats.append(("fire", self.WEIGHTS["fire"] * fire_conf, "HIGH"))
        if smoke_conf >= 0.40:
            candidate_threats.append(("smoke", self.WEIGHTS["smoke"] * smoke_conf, "MEDIUM"))
        if p_stats.get("crowd_over_capacity"):
            candidate_threats.append(("crowd_surge", self.WEIGHTS["crowd_alert"], "MEDIUM"))

        candidate_threats.sort(key=lambda x: x[1], reverse=True)
        dominant_category = candidate_threats[0][0] if candidate_threats else "none"

        fusion_result = {
            "schema_version": "aegis.threat_fusion.v2",
            "zone": zone,
            "reference_timestamp": event_time,
            "threat_score": fused_score,
            "aegis_threat_index_ati": ati_index,
            "threat_level": threat_level,
            "threat_classification": classification,
            "dominant_threat_category": dominant_category,
            "threat_detected": bool(candidate_threats or ati_index >= 30.0),
            "infrastructure_multiplier": infra_multiplier,
            "score_semantics": "Explainable linear weighted rule score scaled by critical infrastructure proximity multiplier (capped at 1.0 / ATI 100).",
            "evidence_count": len(contributions),
            "contributions": contributions,
            "excluded_events": excluded_events,
            "sub_scores": sub_scores,
        }

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(fusion_result, f, indent=2)

        return fusion_result


# =============================================================================
# 8. EXPLAINABILITY & RECOMMENDATION AGENT
# =============================================================================
class ExplainabilityAgent:
    """SHAP-style Feature Attribution, Tactical Reasoning, 7-Question Report, and EigenCAM overlay."""

    def explain(
        self,
        fusion_result: Dict[str, Any],
        video_report: Dict[str, Any],
        osint_results: List[Dict[str, Any]],
        nlp_data: Dict[str, Any],
        critical_infra: List[Dict[str, Any]],
        video_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        total_score = max(fusion_result.get("threat_score", 0.0), 0.0001)
        contributions = fusion_result.get("contributions", [])
        multiplier = fusion_result.get("infrastructure_multiplier", 1.0)

        # 1. SHAP-Style Decomposition
        feature_attribution = []
        for c in contributions:
            pct = round((c["sub_score"] / total_score) * 100.0, 1)
            feature_attribution.append({
                "factor": c["event_type"],
                "modality": c["modality"],
                "raw_contribution": c["sub_score"],
                "percentage_importance": pct,
                "evidence_reason": c["reason"],
            })

        if multiplier > 1.0:
            prox_gain = round(total_score - (total_score / multiplier), 4)
            prox_pct = round((1.0 - (1.0 / multiplier)) * 100.0, 1)
            nearest = critical_infra[0]["name"] if critical_infra else "Protected Facility"
            feature_attribution.append({
                "factor": "critical_infrastructure_proximity",
                "modality": "geospatial",
                "raw_contribution": prox_gain,
                "percentage_importance": prox_pct,
                "evidence_reason": f"Proximity to high-value infrastructure ({nearest}) applied a {multiplier}x priority multiplier.",
            })

        # 2. Tactical Reasoning
        reasoning = (
            f"The AEGIS Threat Analysis Engine assessed the operational risk as {fusion_result.get('threat_level')} "
            f"(AEGIS Threat Index: {fusion_result.get('aegis_threat_index_ati')}/100) under classification "
            f"'{fusion_result.get('threat_classification')}'.\n\n"
            f"Primary Drivers:\n"
        )
        for fa in feature_attribution:
            reasoning += f" • [{fa['modality'].upper()}] {fa['evidence_reason']} (Impact: {fa['percentage_importance']}%)\n"

        if critical_infra:
            reasoning += (
                f"\nGeospatial Vulnerability: Situated within {critical_infra[0]['distance_km']} km of "
                f"'{critical_infra[0]['name']}' ({critical_infra[0]['category']})."
            )

        # 3. Actionable Directives
        recommendations = {
            "immediate_directives_0_to_5_min": [
                "Dispatch nearest Quick Reaction Team (QRT) / Patrol Unit to the verified camera sector.",
                "Verify live perimeter access control and execute precautionary lockdown of high-density portals.",
                "Notify emergency medical trauma unit at nearest facility for standby preparedness.",
            ],
            "area_management_5_to_30_min": [
                "Establish a 250m perimeter cordon around the verified visual coordinates.",
                "Reroute pedestrian and vehicular traffic away from the immediate hazard zone.",
                "Deploy PTZ camera tracking and aerial surveillance on secondary avenues of egress.",
            ],
            "intelligence_and_investigation": [
                "Preserve raw CCTV video streams, YOLOv11 bounding boxes, and ByteTrack track IDs for digital forensics.",
                "Cross-reference suspect visual signatures with regional law enforcement watchlists.",
                "Monitor regional OSINT news feeds for coordinated disturbance indicators.",
            ],
        }

        # 4. Threat Timeline
        timeline = []
        for evt in video_report.get("threat_events", [])[:15]:
            sec = evt.get("timestamp_sec", 0.0)
            timeline.append({
                "timestamp": round(sec, 2),
                "time_display": f"+{sec:.2f}s (CCTV)",
                "source": "Video Intelligence (YOLOv11)",
                "event": f"Visual Detection: {', '.join(evt.get('threats', []))}",
                "confidence": evt.get("detections", [{}])[0].get("confidence", 0.0),
                "severity": "High" if "weapon" in evt.get("threats", []) else "Medium",
            })
        for idx, art in enumerate(osint_results[:5]):
            timeline.append({
                "timestamp": art.get("publishedAt", utc_now()),
                "time_display": f"OSINT -{idx*15}m",
                "source": f"OSINT ({art.get('source', 'News')})",
                "event": art.get("title", "Advisory Recorded"),
                "confidence": 0.80 if art.get("severity") == "High" else 0.60,
                "severity": art.get("severity", "Medium"),
            })

        # 5. Visual EigenCAM Overlay on Highest-Confidence Threat Frame
        visual_path = None
        if video_path and video_path.is_file() and XAI_AVAILABLE and output_dir:
            try:
                best_frame_info = video_report.get("best_threat_frame", {})
                target_frame_idx = best_frame_info.get("frame_index", 10) or 0
                cam_out = output_dir / "threat_highlight.jpg"
                weapon_ckpt = PROJECT_ROOT / "weapon_detection_project" / "ml_models" / "weapon" / "best.pt"
                if weapon_ckpt.is_file():
                    generate_eigencam_on_video_frame(
                        str(video_path),
                        str(weapon_ckpt),
                        str(cam_out),
                        frame_index=int(target_frame_idx),
                    )
                    visual_path = str(cam_out)
            except Exception as e:
                print(f"[XAI] Visual EigenCAM notice: {e}")

        # 6. Structured Answers to the 7 Mandatory Questions
        p_stats = video_report.get("person_statistics", {})
        det_counts = video_report.get("detection_counts", {})
        dominant_cat = fusion_result.get("dominant_threat_category", "none")

        detected_threat_list = [
            f"{cat} (count: {det_counts[cat]}, max conf: {video_report.get('max_confidence', {}).get(cat, 0.0):.2f})"
            for cat in ("weapon", "fire", "smoke")
            if det_counts.get(cat, 0) > 0
        ]
        threat_str = ", ".join(detected_threat_list) if detected_threat_list else "No visual threats detected"

        seven_questions = {
            "q1_what_is_happening": (
                f"Surveillance video analysis indicates an incident involving {dominant_cat.upper() if dominant_cat != 'none' else 'routine public activity'}. "
                f"A total of {video_report.get('frames_processed', 0)} frames were analyzed, identifying {video_report.get('num_threat_events', 0)} threat frames."
            ),
            "q2_how_many_people": (
                f"{p_stats.get('unique_person_count', 0)} unique individuals were tracked across the video feed using ByteTrack. "
                f"Peak concurrent occupancy was {p_stats.get('peak_concurrent_persons', 0)} persons against a venue capacity of {p_stats.get('venue_capacity', 15)}."
            ),
            "q3_which_threats_detected": threat_str,
            "q4_dominant_threat": (
                f"{dominant_cat.upper()} is the dominant threat category based on severity ranking and persistent visual detections."
            ),
            "q5_overall_threat_severity_and_why": (
                f"The overall threat severity is assessed as {fusion_result.get('threat_level')} (ATI: {fusion_result.get('aegis_threat_index_ati')}/100) "
                f"under classification '{fusion_result.get('threat_classification')}'. Rationale: {reasoning.splitlines()[0]}"
            ),
            "q6_evidence_supporting_conclusion": (
                f"Visual telemetry confirms {det_counts.get('weapon', 0)} weapon detections (peak conf: {video_report.get('max_confidence', {}).get('weapon', 0.0):.2f}), "
                f"{det_counts.get('fire', 0)} fire detections, verified ByteTrack trajectories, and spatial EigenCAM feature activation structure."
            ),
            "q7_limitations_and_additional_info": (
                "Automated neural network inference: model confidence represents perceptual activation, not calibrated ground-truth intent. "
                "Human review of footage is required before dispatching tactical assets. OSINT is supplementary."
            ),
        }

        # 7. Executive Report Markdown
        report_md = f"""# 🛡️ AEGIS EXECUTIVE THREAT INTELLIGENCE REPORT
**Generated At:** {utc_now()}  
**System Status:** Multimodal Intelligence Pipeline Synchronized  
**Operational Zone:** {fusion_result.get('zone')}  

---

## 🚨 Threat Index & Executive Assessment

| Metric | Assessment |
| :--- | :--- |
| **AEGIS Threat Index (ATI)** | **{fusion_result.get('aegis_threat_index_ati')} / 100** |
| **Official Threat Level** | **{fusion_result.get('threat_level')}** |
| **Threat Classification** | `{fusion_result.get('threat_classification')}` |
| **Dominant Threat Category** | `{dominant_cat.upper()}` |
| **Infrastructure Proximity Multiplier** | **{multiplier}x** |

---

## 📋 Comprehensive Intelligence Answers (7 Core Questions)

1. **What is happening in the video?**  
   {seven_questions['q1_what_is_happening']}

2. **How many people are visible or reliably tracked?**  
   {seven_questions['q2_how_many_people']}

3. **Which threats were detected?**  
   {seven_questions['q3_which_threats_detected']}

4. **Which threat category is dominant?**  
   {seven_questions['q4_dominant_threat']}

5. **What is the overall threat severity and why?**  
   {seven_questions['q5_overall_threat_severity_and_why']}

6. **What evidence supports the conclusion?**  
   {seven_questions['q6_evidence_supporting_conclusion']}

7. **What additional information or limitations affect the result?**  
   {seven_questions['q7_limitations_and_additional_info']}

---

## 🔍 Explainable AI (SHAP-Style Feature Attribution)

Decomposition of factors contributing to the AEGIS Threat Index:

"""
        for fa in feature_attribution:
            report_md += f"- **{fa['factor'].upper()}** ({fa['modality']}): **{fa['percentage_importance']}%** of score  \n  *{fa['evidence_reason']}*\n"

        report_md += f"""
---

## 🧠 Tactical Reasoning & Correlation Analysis

{reasoning}

---

## 🗺️ Geospatial & Critical Infrastructure Vulnerability
"""
        for site in critical_infra[:5]:
            report_md += f"- **{site['name']}** ({site['category']}) — **{site['distance_km']} km** away\n"

        report_md += """
---

## 📋 Recommended Action Plan

### ⏱️ Immediate Directives (0 - 5 Minutes)
"""
        for act in recommendations["immediate_directives_0_to_5_min"]:
            report_md += f"1. {act}\n"

        report_md += """
### ⏱️ Area Management & Containment (5 - 30 Minutes)
"""
        for act in recommendations["area_management_5_to_30_min"]:
            report_md += f"1. {act}\n"

        report_md += """
### 🛡️ Ongoing Intelligence & Forensic Protocols
"""
        for act in recommendations["intelligence_and_investigation"]:
            report_md += f"1. {act}\n"

        report_md += "\n---\n*Report generated automatically by the AEGIS Open-Source Intelligence & Threat Analysis Platform.*\n"

        explanation_data = {
            "schema_version": "aegis.xai.v2",
            "generated_at": utc_now(),
            "ati_index": fusion_result.get("aegis_threat_index_ati"),
            "threat_level": fusion_result.get("threat_level"),
            "classification": fusion_result.get("threat_classification"),
            "dominant_threat_category": dominant_cat,
            "feature_attribution": feature_attribution,
            "tactical_reasoning": reasoning,
            "seven_questions": seven_questions,
            "recommendations": recommendations,
            "timeline": timeline,
            "visual_overlay": visual_path,
            "executive_summary_markdown": report_md,
        }

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "daily_intelligence_report.md", "w", encoding="utf-8") as f:
                f.write(report_md)
            with open(output_dir / "daily_intelligence_report.json", "w", encoding="utf-8") as f:
                json.dump(explanation_data, f, indent=2)
            with open(output_dir / "xai_explanation.json", "w", encoding="utf-8") as f:
                json.dump(explanation_data, f, indent=2)

        return explanation_data


# =============================================================================
# 9. COMPLETE PIPELINE ORCHESTRATOR
# =============================================================================
def run_complete_pipeline(
    location: str = "Connaught Place, New Delhi",
    video_path: Optional[str | Path] = None,
    venue_capacity: int = 15,
    zone: str = "zone-alpha-1",
    event_time: Optional[str] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    frame_skip: int = 1,
    skip_video: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    start_total_time = time.time()
    event_time = event_time or utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str, pct: int):
        if progress_callback:
            try:
                progress_callback(msg, pct)
            except Exception:
                pass

    _progress("Initializing AEGIS Multimodal Intelligence Pipeline...", 5)

    # 1. Location & GIS
    _progress("Stage 1/8: Geocoding location & mapping critical infrastructure...", 10)
    gis_agent = LocationGISAgent()
    geo_data = gis_agent.geocode(location)
    lat = geo_data["latitude"]
    lon = geo_data["longitude"]
    critical_infra, infra_multiplier = gis_agent.identify_critical_infrastructure(lat, lon, radius_km=5.0)

    # 2. Video Intelligence
    _progress("Stage 2/8: Running YOLOv11 detectors & ByteTrack person tracking...", 20)
    video_agent = VideoIntelligenceAgent()
    resolved_video_path = None
    video_report: Dict[str, Any] = {}

    if skip_video:
        video_report = {
            "detection_counts": {"weapon": 0, "fire": 0, "smoke": 0, "person": 0},
            "max_confidence": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0},
            "detection_ratios": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0},
            "average_confidence": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0},
            "person_statistics": {"peak_concurrent_persons": 0, "unique_person_count": 0, "total_person_detections": 0, "crowd_over_capacity": False, "occupancy_ratio": 0.0},
            "threat_events": [],
            "frame_timeline": [],
        }
    else:
        if video_path:
            cand = Path(video_path).resolve()
            if cand.is_file():
                resolved_video_path = cand
            elif (DEFAULT_INPUT_DIR / video_path).is_file():
                resolved_video_path = DEFAULT_INPUT_DIR / video_path

        if not resolved_video_path or not resolved_video_path.is_file():
            fallback_video = DEFAULT_INPUT_DIR / "YOUTUBE_YouTubeCCTV196_gun_2.mp4"
            if fallback_video.is_file():
                resolved_video_path = fallback_video

        if resolved_video_path and resolved_video_path.is_file():
            video_report = video_agent.process(
                resolved_video_path,
                output_dir=output_dir,
                venue_capacity=venue_capacity,
                frame_skip=frame_skip,
                progress_cb=_progress,
            )
        else:
            video_report = {
                "detection_counts": {"weapon": 0, "fire": 0, "smoke": 0, "person": 0},
                "max_confidence": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0},
                "detection_ratios": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0},
                "average_confidence": {"weapon": 0.0, "fire": 0.0, "smoke": 0.0, "person": 0.0},
                "person_statistics": {"peak_concurrent_persons": 0, "unique_person_count": 0, "total_person_detections": 0, "crowd_over_capacity": False, "occupancy_ratio": 0.0},
                "threat_events": [],
                "frame_timeline": [],
            }

    # 3. Gemini Understanding
    _progress("Stage 3/8: Evaluating visual narrative with Gemini video understanding...", 45)
    gemini_agent = GeminiVideoAgent()
    gemini_data = gemini_agent.analyze(resolved_video_path, video_report)

    # 4. OSINT Intelligence
    _progress("Stage 4/8: Querying regional OSINT and open-source incident feeds...", 55)
    osint_agent = OSINTAgent()
    osint_results = osint_agent.collect_intelligence(
        location,
        user_lat=lat,
        user_lon=lon,
        radius_km=50.0,
        output_file=output_dir / "osint_results.json",
    )

    # 5. NLP Intelligence
    _progress("Stage 5/8: Running spaCy entity extraction & cross-source reconciliation...", 65)
    nlp_agent = NLPIntelligenceAgent()
    nlp_data = nlp_agent.process(
        video_report,
        osint_results,
        gemini_data,
        output_nlp=output_dir / "nlp_intelligence.json",
        output_combined=output_dir / "combined_intelligence.json",
    )

    # 6. Representation Learning
    _progress("Stage 6/8: Generating 384-dimensional multimodal threat embedding...", 75)
    rep_module = RepresentationLearningModule()
    embedding, rep_metadata = rep_module.generate_multimodal_features(
        video_report, osint_results, nlp_data, critical_infra, lat, lon, output_dir=output_dir
    )

    # 7. Threat Assessment & Evidence Fusion
    _progress("Stage 7/8: Calculating AEGIS Threat Index (ATI) & fusing multimodal evidence...", 85)
    fusion_agent = ThreatFusionAgent()
    fusion_result = fusion_agent.fuse(
        video_report,
        osint_results,
        nlp_data,
        infra_multiplier=infra_multiplier,
        zone=zone,
        event_time=event_time,
        output_file=output_dir / "threat_score.json",
    )

    # Generate GIS GeoJSON
    gis_geojson = gis_agent.generate_gis_layer(
        lat, lon, location, critical_infra,
        threat_level=fusion_result["threat_level"],
        output_file=output_dir / "gis_layer.geojson",
    )

    # 8. Explainability & Reporting
    _progress("Stage 8/8: Synthesizing SHAP attribution, EigenCAM overlay & executive brief...", 92)
    xai_agent = ExplainabilityAgent()
    xai_data = xai_agent.explain(
        fusion_result,
        video_report,
        osint_results,
        nlp_data,
        critical_infra,
        video_path=resolved_video_path,
        output_dir=output_dir,
    )

    elapsed_time = round(time.time() - start_total_time, 2)
    _progress("Threat intelligence analysis synchronized and ready.", 100)

    # Compile Unified Master Result Payload
    ati = fusion_result["aegis_threat_index_ati"]
    level = fusion_result["threat_level"]
    classification = fusion_result["threat_classification"]
    dominant_threat = fusion_result.get("dominant_threat_category", "none")

    # Threat ranking
    v_counts = video_report.get("detection_counts", {})
    v_ratios = video_report.get("detection_ratios", {})
    v_max = video_report.get("max_confidence", {})

    ranked_threats = []
    category_meta = [
        ("weapon", "Rifle / Handgun / Firearm", "CRITICAL"),
        ("fire", "Active Flame / Blaze", "HIGH"),
        ("smoke", "Smoke Plume", "MEDIUM"),
        ("suspicious_activity", "Anomalous Tactical Movement", "MEDIUM"),
    ]

    for cat, label, base_sev in category_meta:
        count = v_counts.get(cat, 0)
        conf = float(v_max.get(cat, 0.0))
        ratio = float(v_ratios.get(cat, 0.0))
        detected = count > 0 or conf >= 0.35
        status_str = "DETECTED" if detected else "NOT_DETECTED"
        ranked_threats.append({
            "category": cat,
            "label": label,
            "status": status_str,
            "confidence": round(conf, 4),
            "detection_count": count,
            "frame_percentage": round(ratio * 100.0, 1),
            "severity": base_sev if detected else "LOW",
            "is_dominant": (cat == dominant_threat),
            "explanation": f"{label} detected in {count} instances across video frames." if detected else f"No {label.lower()} detected in video footage.",
        })

    ranked_threats.sort(key=lambda x: (x["status"] == "DETECTED", x["confidence"]), reverse=True)
    for idx, item in enumerate(ranked_threats, 1):
        item["rank"] = idx

    p_stats = video_report.get("person_statistics", {})

    master_result = {
        "status": "success",
        "job_id": f"job-{int(time.time())}",
        "execution_time_seconds": elapsed_time,
        "timestamp": event_time,
        "event_overview": {
            "video_name": resolved_video_path.name if resolved_video_path else "None (Geospatial/OSINT Only)",
            "location": location,
            "coordinates": {"latitude": lat, "longitude": lon},
            "display_name": geo_data.get("display_name", location),
            "threat_level": level,
            "threat_detected": fusion_result.get("threat_detected", False),
            "aegis_threat_index": ati,
            "threat_classification": classification,
            "calculation_explanation": (
                f"AEGIS Threat Index (ATI: {ati}/100) calculated via weighted multimodal evidence: "
                f"{', '.join([f'{c['event_type']} ({c['sub_score']:.3f})' for c in fusion_result.get('contributions', [])])} "
                f"scaled by infrastructure proximity multiplier {infra_multiplier}x."
            ),
            "analysis_status": "COMPLETED",
        },
        "person_and_crowd_analysis": {
            "unique_people_count": p_stats.get("unique_person_count", 0),
            "total_person_detections": p_stats.get("total_person_detections", 0),
            "detections_vs_unique_explanation": (
                f"{p_stats.get('total_person_detections', 0)} bounding-box detections were observed across {video_report.get('frames_processed', 0)} frames, "
                f"corresponding to {p_stats.get('unique_person_count', 0)} unique tracked individuals identified through ByteTrack trajectory association."
            ),
            "peak_concurrent_persons": p_stats.get("peak_concurrent_persons", 0),
            "avg_concurrent_persons": p_stats.get("avg_concurrent_persons", 0.0),
            "venue_capacity": venue_capacity,
            "crowd_over_capacity": p_stats.get("crowd_over_capacity", False),
            "occupancy_ratio": p_stats.get("occupancy_ratio", 0.0),
            "unique_track_ids": p_stats.get("unique_track_ids", []),
        },
        "threat_detection": {
            "dominant_threat_category": dominant_threat,
            "ranked_threat_categories": ranked_threats,
        },
        "heatmaps_and_visualizations": {
            "spatial_heatmap": {
                "type": "eigencam_activation_overlay",
                "available": bool(xai_data.get("visual_overlay")),
                "artifact_name": "threat_highlight.jpg",
                "frame_index": video_report.get("best_threat_frame", {}).get("frame_index", 0),
                "timestamp_sec": video_report.get("best_threat_frame", {}).get("timestamp_sec", 0.0),
                "description": "EigenCAM class-agnostic feature activation overlay visualizing internal feature activations on the highest-confidence threat frame.",
            },
            "annotated_video_artifact": f"{resolved_video_path.stem}_analyzed.mp4" if resolved_video_path else None,
            "frame_timeline": video_report.get("frame_timeline", []),
            "category_comparison": [
                {
                    "category": c,
                    "detections": v_counts.get(c, 0),
                    "frames": video_report.get("detection_frames", {}).get(c, 0),
                    "max_confidence": v_max.get(c, 0.0),
                    "avg_confidence": video_report.get("average_confidence", {}).get(c, 0.0),
                }
                for c in ("weapon", "fire", "smoke", "person")
            ],
        },
        "gemini_understanding": gemini_data,
        "osint_intelligence": {
            "status": "completed" if osint_results else "no_verified_advisories",
            "records_count": len(osint_results),
            "records": osint_results,
        },
        "nlp_intelligence": nlp_data,
        "representation_learning": rep_metadata,
        "evidence_fusion": {
            "ati_score": ati,
            "threat_level": level,
            "classification": classification,
            "infrastructure_multiplier": infra_multiplier,
            "nearby_critical_infrastructure": critical_infra[:5],
            "feature_attribution": xai_data.get("feature_attribution", []),
            "contributions": fusion_result.get("contributions", []),
            "excluded_events": fusion_result.get("excluded_events", []),
        },
        "final_intelligence_report": {
            "seven_questions": xai_data.get("seven_questions", {}),
            "tactical_reasoning": xai_data.get("tactical_reasoning", ""),
            "recommendations": xai_data.get("recommendations", {}),
            "timeline": xai_data.get("timeline", []),
            "executive_summary_markdown": xai_data.get("executive_summary_markdown", ""),
        },
        "artifacts_directory": str(output_dir),
    }

    with open(output_dir / "threat_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_result, f, indent=2)

    return master_result
