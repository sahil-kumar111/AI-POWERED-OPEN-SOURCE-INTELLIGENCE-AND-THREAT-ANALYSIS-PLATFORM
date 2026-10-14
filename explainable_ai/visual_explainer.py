from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def _imread_unicode(path: str | Path, cv2: Any) -> Any:
    """Read an image via Python file I/O so non-ASCII Windows paths work."""
    import numpy as np

    try:
        encoded = np.fromfile(str(Path(path)), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: str | Path, image: Any, cv2: Any) -> bool:
    """Write an image via imencode + Python file I/O for Unicode paths."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(destination))
        return destination.is_file() and destination.stat().st_size > 0
    except OSError:
        return False


def generate_eigencam(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    confidence_threshold: float = 0.25,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate a one-image, class-agnostic EigenCAM-style feature map for a YOLO model.

    Optional dependency: ultralytics. This intentionally runs a single image, never
    rescans/retrains the video. EigenCAM visualizes prominent activation structure;
    it is not a class-specific causal attribution and must be human-reviewed.
    """
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Visual XAI requires ultralytics, torch, opencv-python, and numpy. Install into the project's environment with: pip install ultralytics opencv-python numpy") from exc

    image_path = Path(image_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not image_path.is_file(): raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not checkpoint_path.is_file(): raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    if not 0 <= confidence_threshold <= 1: raise ValueError("confidence_threshold must be in [0, 1]")
    image_bgr = _imread_unicode(image_path, cv2)
    if image_bgr is None: raise ValueError(f"OpenCV could not read image: {image_path}")

    detector = YOLO(str(checkpoint_path))
    core = getattr(detector, "model", None)
    layers = getattr(core, "model", None)
    if layers is None: raise RuntimeError("Loaded YOLO checkpoint does not expose the expected model.model layers")
    candidate = None
    candidate_name = None
    # Select the final tensor-producing feature block before the detection head.
    ignored_names = {"detect", "segment", "pose", "obb", "classify", "worlddetect", "v10detect"}
    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        if layer.__class__.__name__.lower() in ignored_names:
            continue
        if any(True for _ in layer.modules()):
            candidate, candidate_name = layer, f"model.model[{layer_index}]::{layer.__class__.__name__}"
            break
    if candidate is None: raise RuntimeError("Could not identify a feature layer for EigenCAM")

    captured: dict[str, Any] = {}
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        # For modules returning tuples/lists, retain the first 4-D tensor.
        tensors = [output] if torch.is_tensor(output) else list(output) if isinstance(output, (tuple, list)) else []
        feature = next((x for x in tensors if torch.is_tensor(x) and x.ndim == 4), None)
        if feature is not None: captured["feature"] = feature.detach()

    handle = candidate.register_forward_hook(hook)
    try:
        predict_kwargs: dict[str, Any] = {"source": str(image_path), "conf": confidence_threshold, "verbose": False}
        if device: predict_kwargs["device"] = device
        predictions = detector.predict(**predict_kwargs)
    finally:
        handle.remove()
    if "feature" not in captured: raise RuntimeError(f"Hook at {candidate_name} did not produce a 4-D feature map")

    feature = captured["feature"][0].float().cpu().numpy()  # C x H x W
    channels, height, width = feature.shape
    matrix = feature.reshape(channels, -1).astype(np.float64)
    matrix -= matrix.mean(axis=1, keepdims=True)
    # First principal component across channel responses at spatial positions.
    try:
        u, _singular, _vt = np.linalg.svd(matrix, full_matrices=False)
        component = u[:, 0]
        flat_cam = component @ matrix
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("SVD failed while calculating EigenCAM") from exc
    # Resolve eigenvector sign consistently with overall activation direction.
    mean_activation = matrix.mean(axis=0)
    if np.std(flat_cam) > 0 and np.std(mean_activation) > 0 and np.corrcoef(flat_cam, mean_activation)[0, 1] < 0:
        flat_cam *= -1
    cam = np.maximum(flat_cam.reshape(height, width), 0)
    cam -= cam.min()
    if cam.max() > 0: cam /= cam.max()
    heat = cv2.resize((cam * 255).astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0]))
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.58, heat_color, 0.42, 0)

    detection_rows = []
    if predictions:
        boxes = getattr(predictions[0], "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
            conf = boxes.conf.detach().cpu().tolist() if getattr(boxes, "conf", None) is not None else []
            cls = boxes.cls.detach().cpu().tolist() if getattr(boxes, "cls", None) is not None else []
            names = getattr(predictions[0], "names", {})
            for box, score, cls_id in zip(xyxy, conf, cls):
                idx = int(cls_id)
                detection_rows.append({"class_id": idx, "class_name": names.get(idx, str(idx)) if isinstance(names, dict) else str(idx), "confidence": round(float(score), 6), "bbox_xyxy": [round(float(v), 2) for v in box]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not _imwrite_unicode(output_path, overlay, cv2): raise RuntimeError(f"Could not save CAM overlay to {output_path}")
    metadata = {
        "method": "EigenCAM-style first-principal-component activation map",
        "method_scope": "class_agnostic_feature_activation_visualization",
        "source_image": str(image_path), "checkpoint": str(checkpoint_path), "overlay_path": str(output_path),
        "target_layer": candidate_name, "feature_map_shape_chw": [int(channels), int(height), int(width)],
        "confidence_threshold": confidence_threshold, "detections": detection_rows,
        "limitations": ["EigenCAM is class-agnostic; it does not explain one specific class score.", "A heatmap is not causal proof and may highlight context or artifacts.", "Human review is required; do not treat the overlay alone as evidence an event is real."],
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def generate_eigencam_on_video_frame(
    video_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    frame_index: int = 0,
    confidence_threshold: float = 0.25,
    device: str | None = None,
) -> dict[str, Any]:
    """Extract exactly one frame and explain it; never runs inference across the full video."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video-frame extraction requires opencv-python: pip install opencv-python") from exc
    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames and frame_index >= total_frames:
            raise ValueError(f"frame_index {frame_index} is outside video frame range 0..{total_frames - 1}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not decode video frame {frame_index}: {video_path}")
    finally:
        capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_frame = output_path.with_name(output_path.stem + f"._frame_{frame_index}.png")
    if not _imwrite_unicode(temp_frame, frame, cv2):
        raise RuntimeError(f"Could not write temporary frame image: {temp_frame}")
    try:
        metadata = generate_eigencam(temp_frame, checkpoint_path, output_path, confidence_threshold=confidence_threshold, device=device)
    finally:
        try:
            temp_frame.unlink(missing_ok=True)
        except OSError:
            pass
    metadata["source_video"] = str(video_path)
    metadata["frame_index"] = frame_index
    metadata["frame_timestamp_seconds"] = round(frame_index / fps, 6) if fps > 0 else None
    metadata["video_fps"] = fps if fps > 0 else None
    metadata["video_total_frames"] = total_frames or None
    metadata["source_image"] = f"{video_path}#frame={frame_index}"
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata
