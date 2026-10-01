import cv2

from weapon_detection_project.weapon_detector import WeaponDetector


input_video = "input/normal_video.mp4"
output_video = "output/weapon_test.mp4"

detector = WeaponDetector(
    imgsz=320,
    conf=0.3,
    iou=0.50,
)

video = cv2.VideoCapture(input_video)

if not video.isOpened():
    raise RuntimeError(f"Could not open video: {input_video}")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height),
)

frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    detections = detector.detect(frame)

    if detections:
        print(f"Frame {frame_number} = WEAPON DETECTED")

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)

            label = f"{detection.class_name} {detection.confidence:.2f}"

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
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    else:
        print(f"Frame {frame_number} = NO WEAPON")

    writer.write(frame)

    frame_number += 1


video.release()
writer.release()

print()
print("Finished.")
print(f"Output: {output_video}")