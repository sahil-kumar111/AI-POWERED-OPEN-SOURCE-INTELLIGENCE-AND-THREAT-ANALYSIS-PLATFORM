#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed.")
    print("Run:")
    print("    pip install ultralytics opencv-python")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


WEAPON_MODEL_PATH = (
    PROJECT_ROOT
    / "weapon_detection_project"
    / "ml_models"
    / "weapon"
    / "best.pt"
)

WEAPON_SHA256 = (
    "66fdda89b7b535bd0afe50d61287b793d25fc7411338b6e14d2b8c9d48cd76f9"
)

WEAPON_CLASSES = {
    0: "pistol",
    1: "knife",
    2: "rifle",
    3: "shotgun",
    4: "smg",
}


FIRE_SMOKE_MODEL_PATH = (
    PROJECT_ROOT
    / "fire_smoke_detection"
    / "fire_smoke_detection"
    / "ml_models"
    / "fire_smoke"
    / "best_nano_111.pt"
)

FIRE_SMOKE_SHA256 = (
    "3cbeb9569d8c4057734a81437b90c3e95e55ebdd40a5c5ea7dd7873a247cef13"
)

FIRE_SMOKE_CLASSES = {
    0: "Fire",
    1: "Smoke",
}

FIRE_CLASS_ID = 0
SMOKE_CLASS_ID = 1


PERSON_MODEL_PATH = (
    PROJECT_ROOT
    / "person_detection"
    / "person_detection"
    / "ml_models"
    / "person"
    / "yolo11n.pt"
)

PERSON_CLASS_ID = 0


WEAPON_IMGSZ =  320
WEAPON_CONF = 0.3
WEAPON_IOU = 0.50

FIRE_SMOKE_IMGSZ = 640
FIRE_SMOKE_CONF = 0.40
FIRE_SMOKE_IOU = 0.50

PERSON_IMGSZ = 640
PERSON_CONF = 0.40
PERSON_IOU = 0.50


THREAT_CATEGORIES = {
    "weapon",
    "fire",
    "smoke",
}


COLORS = {
    "weapon": (0, 0, 255),
    "fire": (0, 100, 255),
    "smoke": (180, 180, 180),
    "person": (0, 200, 0),
}


@dataclass(frozen=True)
class Detection:
    category: str
    class_name: str
    confidence: float
    bbox: List[float]
    track_id: Optional[int] = None


def sha256_of_file(path: Path) -> str:
    hash_object = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hash_object.update(chunk)

    return hash_object.hexdigest()


def validate_frame(frame: np.ndarray) -> None:
    if frame is None:
        raise ValueError("Frame is None.")

    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy.ndarray.")

    if frame.size == 0:
        raise ValueError("Frame is empty.")

    if frame.ndim != 3:
        raise ValueError("Frame must be an HxWxC array.")

    if frame.shape[2] != 3:
        raise ValueError("Frame must have exactly 3 channels.")


class VerifiedYoloDetector:
    def __init__(
        self,
        name: str,
        model_path: Path,
        expected_sha256: Optional[str],
        expected_classes: Optional[dict],
        imgsz: int,
        conf: float,
        iou: float,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.expected_sha256 = expected_sha256
        self.expected_classes = expected_classes

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.enabled = False
        self.model = None

        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            print(
                f"WARNING: {self.name} model not found: "
                f"{self.model_path}"
            )
            return

        if not self.model_path.is_file():
            print(
                f"WARNING: {self.name} model path is not a file: "
                f"{self.model_path}"
            )
            return

        if self.expected_sha256 is not None:
            actual_sha256 = sha256_of_file(self.model_path)

            if actual_sha256 != self.expected_sha256:
                print(
                    f"ERROR: {self.name} SHA-256 mismatch."
                )
                print(f"Expected: {self.expected_sha256}")
                print(f"Actual:   {actual_sha256}")
                return

        try:
            model = YOLO(str(self.model_path))
        except Exception as error:
            print(
                f"ERROR: {self.name} failed to load model: "
                f"{error}"
            )
            return

        if self.expected_classes is not None:
            actual_classes = dict(model.names)

            if actual_classes != self.expected_classes:
                print(
                    f"ERROR: {self.name} class mapping mismatch."
                )
                print(f"Expected: {self.expected_classes}")
                print(f"Actual:   {actual_classes}")
                return

        self.model = model
        self.enabled = True

    def raw_predict(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None,
    ):
        if not self.enabled:
            return None

        validate_frame(frame)

        return self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=classes,
            verbose=False,
        )

    def raw_track(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None,
    ):
        if not self.enabled:
            return None

        validate_frame(frame)

        return self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )


def build_detectors():
    detectors = {}

    detectors["weapon"] = VerifiedYoloDetector(
        name="WeaponDetector",
        model_path=WEAPON_MODEL_PATH,
        expected_sha256=WEAPON_SHA256,
        expected_classes=WEAPON_CLASSES,
        imgsz=WEAPON_IMGSZ,
        conf=WEAPON_CONF,
        iou=WEAPON_IOU,
    )

    detectors["fire_smoke"] = VerifiedYoloDetector(
        name="FireSmokeDetector",
        model_path=FIRE_SMOKE_MODEL_PATH,
        expected_sha256=FIRE_SMOKE_SHA256,
        expected_classes=FIRE_SMOKE_CLASSES,
        imgsz=FIRE_SMOKE_IMGSZ,
        conf=FIRE_SMOKE_CONF,
        iou=FIRE_SMOKE_IOU,
    )

    detectors["person"] = VerifiedYoloDetector(
        name="PersonDetector",
        model_path=PERSON_MODEL_PATH,
        expected_sha256=None,
        expected_classes=None,
        imgsz=PERSON_IMGSZ,
        conf=PERSON_CONF,
        iou=PERSON_IOU,
    )

    return detectors


def convert_yolo_results(
    results,
    category: str,
    class_map: dict,
) -> List[Detection]:
    detections = []

    if not results:
        return detections

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()

    for bbox, confidence, class_id in zip(
        xyxy,
        confidences,
        class_ids,
    ):
        class_id = int(class_id)

        class_name = class_map.get(
            class_id,
            "unknown",
        )

        detections.append(
            Detection(
                category=category,
                class_name=class_name,
                confidence=float(confidence),
                bbox=[
                    float(value)
                    for value in bbox
                ],
            )
        )

    return detections


def run_weapon_detector(
    detector: VerifiedYoloDetector,
    frame: np.ndarray,
) -> List[Detection]:
    if not detector.enabled:
        return []

    results = detector.raw_predict(frame)

    return convert_yolo_results(
        results,
        "weapon",
        WEAPON_CLASSES,
    )


def run_fire_smoke_detector(
    detector: VerifiedYoloDetector,
    frame: np.ndarray,
) -> List[Detection]:
    if not detector.enabled:
        return []

    results = detector.raw_predict(frame)

    detections = []

    if not results:
        return detections

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()

    for bbox, confidence, class_id in zip(
        xyxy,
        confidences,
        class_ids,
    ):
        class_id = int(class_id)

        if class_id == FIRE_CLASS_ID:
            category = "fire"
            class_name = "fire"

        elif class_id == SMOKE_CLASS_ID:
            category = "smoke"
            class_name = "smoke"

        else:
            continue

        detections.append(
            Detection(
                category=category,
                class_name=class_name,
                confidence=float(confidence),
                bbox=[
                    float(value)
                    for value in bbox
                ],
            )
        )

    return detections


def run_person_detector(
    detector: VerifiedYoloDetector,
    frame: np.ndarray,
) -> List[Detection]:
    if not detector.enabled:
        return []

    results = detector.raw_track(
        frame,
        classes=[PERSON_CLASS_ID],
    )

    detections = []

    if not results:
        return detections

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()

    track_ids = None
    if boxes.id is not None:
        track_ids = boxes.id.tolist()

    for index, (bbox, confidence, class_id) in enumerate(
        zip(
            xyxy,
            confidences,
            class_ids,
        )
    ):
        if int(class_id) != PERSON_CLASS_ID:
            continue

        track_id = None

        if track_ids is not None and index < len(track_ids):
            track_id = int(track_ids[index])

        detections.append(
            Detection(
                category="person",
                class_name="person",
                confidence=float(confidence),
                bbox=[
                    float(value)
                    for value in bbox
                ],
                track_id=track_id,
            )
        )

    return detections


def run_all_detectors(
    detectors: dict,
    frame: np.ndarray,
    include_person: bool,
) -> List[Detection]:
    validate_frame(frame)

    detections = []

    detections.extend(
        run_weapon_detector(
            detectors["weapon"],
            frame,
        )
    )

    detections.extend(
        run_fire_smoke_detector(
            detectors["fire_smoke"],
            frame,
        )
    )

    if include_person:
        detections.extend(
            run_person_detector(
                detectors["person"],
                frame,
            )
        )

    return detections


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in detection.bbox
        ]

        color = COLORS.get(
            detection.category,
            (255, 255, 255),
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = (
            f"{detection.category.upper()}: "
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        (
            text_width,
            text_height,
        ), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        text_y = max(
            y1,
            text_height + 6,
        )

        cv2.rectangle(
            frame,
            (
                x1,
                text_y - text_height - 6,
            ),
            (
                x1 + text_width + 4,
                text_y,
            ),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (
                x1 + 2,
                text_y - 4,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    threats = sorted(
        {
            detection.category
            for detection in detections
            if detection.category
            in THREAT_CATEGORIES
        }
    )

    if threats:
        banner = (
            "THREAT DETECTED: "
            + ", ".join(
                threat.upper()
                for threat in threats
            )
        )

        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 40),
            (0, 0, 200),
            -1,
        )

        cv2.putText(
            frame,
            banner,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def process_video(
    video_path: Path,
    detectors: dict,
    output_dir: Path,
    include_person: bool = True,
    frame_skip: int = 1,
) -> None:
    video = cv2.VideoCapture(
        str(video_path)
    )

    if not video.isOpened():
        print(
            f"ERROR: Could not open video: "
            f"{video_path}"
        )
        return

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_dir
        / f"{video_path.stem}_analyzed.mp4"
    )

    output_report = (
        output_dir
        / f"{video_path.stem}_report.json"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        video.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_video}"
        )

    categories = [
        "weapon",
        "fire",
        "smoke",
        "person",
    ]

    detection_counts = {
        category: 0
        for category in categories
    }

    detection_frames = {
        category: 0
        for category in categories
    }

    detected_any = {
        category: False
        for category in categories
    }

    confidence_sum = {
        category: 0.0
        for category in categories
    }

    confidence_count = {
        category: 0
        for category in categories
    }

    max_confidence = {
        category: 0.0
        for category in categories
    }

    events = []

    unique_person_ids = set()

    frame_idx = 0
    inference_frames = 0

    last_detections: List[Detection] = []

    while True:
        success, frame = video.read()

        if not success:
            break

        should_infer = (
            frame_idx
            % max(frame_skip, 1)
            == 0
        )

        if should_infer:
            detections = run_all_detectors(
                detectors,
                frame,
                include_person,
            )

            last_detections = detections
            inference_frames += 1

            frame_categories = set()

            for detection in detections:
                category = detection.category

                detection_counts[
                    category
                ] += 1

                confidence_sum[
                    category
                ] += detection.confidence

                confidence_count[
                    category
                ] += 1

                if detection.confidence > max_confidence[
                    category
                ]:
                    max_confidence[
                        category
                    ] = detection.confidence

                frame_categories.add(
                    category
                )

                detected_any[
                    category
                ] = True

                if (
                    detection.category == "person"
                    and detection.track_id is not None
                ):
                    unique_person_ids.add(
                        detection.track_id
                    )

            for category in frame_categories:
                detection_frames[
                    category
                ] += 1

            threat_categories = sorted(
                {
                    detection.category
                    for detection in detections
                    if detection.category
                    in THREAT_CATEGORIES
                }
            )

            if threat_categories:
                events.append(
                    {
                        "frame": frame_idx,
                        "timestamp_sec": round(
                            frame_idx / fps,
                            2,
                        ),
                        "threats": threat_categories,
                        "detections": [
                            asdict(detection)
                            for detection in detections
                            if detection.category
                            in THREAT_CATEGORIES
                        ],
                    }
                )

        else:
            detections = last_detections

        annotated_frame = draw_detections(
            frame,
            detections,
        )

        writer.write(
            annotated_frame
        )

        frame_idx += 1

    video.release()
    writer.release()

    average_confidence = {}

    for category in categories:
        if confidence_count[category] > 0:
            average_confidence[
                category
            ] = round(
                confidence_sum[category]
                / confidence_count[category],
                4,
            )
        else:
            average_confidence[
                category
            ] = 0.0

    if inference_frames > 0:
        detection_ratios = {
            category: round(
                detection_frames[category]
                / inference_frames,
                4,
            )
            for category in categories
        }
    else:
        detection_ratios = {
            category: 0.0
            for category in categories
        }

    total_unique_persons = len(unique_person_ids)

    detections_summary = []

    for category in categories:
        detections_summary.append(
            {
                "category": category,
                "detected": detected_any[category],
                "detection_frames": detection_frames[
                    category
                ],
                "detection_ratio": detection_ratios[
                    category
                ],
                "average_confidence": average_confidence[
                    category
                ],
                "max_confidence": round(
                    max_confidence[category],
                    4,
                ),
                "message": (
                    f"{category} detected"
                    if detected_any[category]
                    else f"no {category} detected"
                ),
            }
        )

    report = {
        "input_video": str(video_path),
        "output_video": str(output_video),
        "fps": fps,
        "resolution": [
            width,
            height,
        ],
        "frames_processed": frame_idx,
        "inference_frames": inference_frames,
        "frame_skip": frame_skip,
        "detection_counts": detection_counts,
        "detection_frames": detection_frames,
        "detection_ratios": detection_ratios,
        "average_confidence": average_confidence,
        "max_confidence": {
            category: round(
                confidence,
                4,
            )
            for category, confidence
            in max_confidence.items()
        },
        "detections": detections_summary,
        "person_statistics": {
            "unique_person_count": total_unique_persons,
            "unique_track_ids": sorted(unique_person_ids),
        },
        "threat_events": events,
        "num_threat_events": len(events),
        "model_settings": {
            "weapon": {
                "imgsz": WEAPON_IMGSZ,
                "conf": WEAPON_CONF,
                "iou": WEAPON_IOU,
            },
            "fire_smoke": {
                "imgsz": FIRE_SMOKE_IMGSZ,
                "conf": FIRE_SMOKE_CONF,
                "iou": FIRE_SMOKE_IOU,
            },
            "person": {
                "imgsz": PERSON_IMGSZ,
                "conf": PERSON_CONF,
                "iou": PERSON_IOU,
            },
        },
    }

    with open(
        output_report,
        "w",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print(f"Video: {video_path.name}")
    print(f"Frames: {frame_idx}")
    print()

    for category in categories:
        result = (
            "YES"
            if detected_any[category]
            else "NO"
        )

        print(
            f"{category:<8}: {result}"
        )

    print()
    print(f"Person count: {total_unique_persons}")

    print()
    print("Average confidence")

    for category in categories:
        print(
            f"{category:<8}: "
            f"{average_confidence[category]:.2f}"
        )

    print()
    print(
        f"Output: {output_video}"
    )
    print(
        f"Report: {output_report}"
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combined weapon, fire, smoke "
            "and person video detection pipeline."
        )
    )

    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help=(
            "Process one video. "
            "If omitted, all videos in ./input "
            "are processed."
        ),
    )

    parser.add_argument(
        "--no-person",
        action="store_true",
        help="Disable person detection.",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help=(
            "Run detection every Nth frame. "
            "Default: 1."
        ),
    )

    args = parser.parse_args()

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detectors = build_detectors()

    enabled_detectors = [
        name
        for name, detector
        in detectors.items()
        if detector.enabled
    ]

    if not enabled_detectors:
        print("ERROR: No detectors were loaded.")
        sys.exit(1)

    if args.video:
        videos = [
            Path(args.video)
        ]
    else:
        videos = sorted(
            path
            for path in INPUT_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        )

    if not videos:
        print(
            f"No videos found in: "
            f"{INPUT_DIR}"
        )
        return

    for video_path in videos:
        process_video(
            video_path=video_path,
            detectors=detectors,
            output_dir=OUTPUT_DIR,
            include_person=not args.no_person,
            frame_skip=max(
                args.frame_skip,
                1,
            ),
        )


if __name__ == "__main__":
    main()