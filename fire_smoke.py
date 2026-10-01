import sys
from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parent

fire_smoke_root = (
    project_root
    / "fire_smoke_detection"
    / "fire_smoke_detection"
)

sys.path.insert(
    0,
    str(fire_smoke_root),
)


from backend.app.modules.video.fire_detector import FireDetector

from backend.app.modules.video.smoke_detector import SmokeDetector


input_video = (
    project_root
    / "input"
    / "fire_m.mp4"
)

output_video = (
    project_root
    / "output"
    / "fire_smoke_test.mp4"
)

model_path = (
    fire_smoke_root
    / "ml_models"
    / "fire_smoke"
    / "best_nano_111.pt"
)


if not model_path.exists():

    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )


print(
    f"Using model: {model_path}"
)


fire_detector = FireDetector(
    model_path=model_path,
    imgsz=640,
    conf=0.40,
    iou=0.50,
)


smoke_detector = SmokeDetector(
    model_path=model_path,
    imgsz=640,
    conf=0.40,
    iou=0.50,
)


video = cv2.VideoCapture(
    str(input_video)
)


if not video.isOpened():

    raise RuntimeError(
        f"Could not open video: {input_video}"
    )


fps = video.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30


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


frame_number = 0


while True:

    success, frame = video.read()

    if not success:
        break


    fire_detections = (
        fire_detector.detect(frame)
    )

    smoke_detections = (
        smoke_detector.detect(frame)
    )

    detections = (
        fire_detections
        + smoke_detections
    )


    if detections:

        detected_labels = []


        for detection in detections:

            detected_labels.append(
                detection.class_name
            )


            x1, y1, x2, y2 = map(
                int,
                detection.bbox,
            )


            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )


        print(
            f"Frame {frame_number} = "
            f"{', '.join(detected_labels).upper()} "
            f"DETECTED"
        )

    else:

        print(
            f"Frame {frame_number} = "
            f"NO FIRE/SMOKE"
        )


    writer.write(frame)

    frame_number += 1


video.release()

writer.release()


print()

print("Finished.")

print(
    f"Frames processed: {frame_number}"
)

print(
    f"Output: {output_video}"
)