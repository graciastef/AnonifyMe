import os
import pickle
import subprocess
from pathlib import Path
import threading

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
from clearml import Task
from clearml.automation.controller import PipelineController
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

VIDEOS_DIR = "./videos"
VIDEO_IN_DIR = f"{VIDEOS_DIR}/in"
VIDEO_OUT_DIR = f"{VIDEOS_DIR}/out"
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# Shared progress state keyed by file_key, updated between pipeline steps
# and read by VideoProcessingTask to drive SSE streaming.
_progress_state: dict[str, dict] = {}
_progress_events: dict[str, threading.Event] = {}

# Upload results written by the upload step, read by the task wrapper.
_upload_results: dict[str, dict] = {}


def _update_progress(file_key: str, percentage: int, eta: str = "00:00:00") -> None:
    if file_key in _progress_state:
        _progress_state[file_key]["percentage"] = percentage
        _progress_state[file_key]["eta"] = eta
    if file_key in _progress_events:
        _progress_events[file_key].set()


def _progress_callback(file_key: str, percentage: int):
    """Returns a post_execute_callback that fires after a pipeline step completes."""
    def _cb(pipeline, node):
        _update_progress(file_key, percentage)
    return _cb


def _get_providers(device_id: int = 0):
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": device_id})], device_id
    if "CoreMLExecutionProvider" in available:
        return [("CoreMLExecutionProvider", {"ml_program_compile_on_demand": True})], 0
    return ["CPUExecutionProvider"], -1


def get_face_det_app(device_id: int = 0) -> FaceAnalysis:
    providers, ctx_id = _get_providers(device_id)
    app = FaceAnalysis(
        name="buffalo_l",
        root=".",
        allowed_modules=["detection", "landmark_2d_106"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id)
    return app


def apply_blur(frame, x1, y1, x2, y2):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (99, 99), 30)
    mask_roi = mask[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = np.where(mask_roi[:, :, np.newaxis] == 255, blurred, roi)
    return frame


# ── Pipeline step functions ───────────────────────────────────────────────────

def fetch_video(file_key: str, filename: str, in_dir: str):
    import os
    import cv2
    from clearml import Task
    from face_blur.services.video_storage import download_blob_by_name, download_blobs

    video_path = download_blob_by_name(file_key, filename, in_dir)
    whitelist_dir = os.path.join(in_dir, "whitelist")
    download_blobs(file_key, whitelist_dir)

    cap = cv2.VideoCapture(video_path)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS) or 0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    whitelist_count = len([
        f for f in os.listdir(whitelist_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]) if os.path.exists(whitelist_dir) else 0

    logger = Task.current_task().get_logger()
    logger.report_single_value("fps", fps)
    logger.report_single_value("resolution_width", frame_width)
    logger.report_single_value("resolution_height", frame_height)
    logger.report_single_value("duration_seconds", total_frames / fps)
    logger.report_single_value("whitelist_images", whitelist_count)

    return video_path, whitelist_dir, fps, total_frames, frame_width, frame_height


def extract_frames(video_path: str, frames_dir: str):
    import os
    import cv2
    from pathlib import Path
    from clearml import Task

    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()

    Task.current_task().get_logger().report_single_value("frames_extracted", idx)
    return frames_dir


def detect_faces(frames_dir: str, total_frames: int, device_id: int = 0):
    import os
    import pickle
    import numpy as np
    import cv2
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from clearml import Task

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers, ctx_id = [("CUDAExecutionProvider", {"device_id": device_id})], device_id
    elif "CoreMLExecutionProvider" in available:
        providers, ctx_id = [("CoreMLExecutionProvider", {"ml_program_compile_on_demand": True})], 0
    else:
        providers, ctx_id = ["CPUExecutionProvider"], -1

    face_app = FaceAnalysis(name="buffalo_l", root=".", allowed_modules=["detection", "landmark_2d_106"], providers=providers)
    face_app.prepare(ctx_id=ctx_id)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_"))
    raw_detections = []
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        faces = face_app.get(frame)
        frame_dets = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            frame_dets.append({"xywh": [x1, y1, x2 - x1, y2 - y1], "conf": float(face.det_score), "kps": face.kps})
        raw_detections.append(frame_dets)

    all_confs = [d["conf"] for frame in raw_detections for d in frame]
    logger = Task.current_task().get_logger()
    logger.report_single_value("avg_detection_confidence", float(np.mean(all_confs)) if all_confs else 0.0)
    logger.report_single_value("total_faces_detected", len(all_confs))

    out_path = os.path.join(frames_dir, "raw_detections.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(raw_detections, f)
    return out_path


def track_faces(frames_dir: str, raw_detections_path: str, fps: int):
    import os
    import pickle
    import cv2
    from clearml import Task
    from deep_sort_realtime.deepsort_tracker import DeepSort

    with open(raw_detections_path, "rb") as f:
        raw_detections = pickle.load(f)

    tracker = DeepSort(max_age=fps // 2, n_init=fps // 2, max_iou_distance=0.9, max_cosine_distance=0.1, embedder="mobilenet", embedder_gpu=True)
    min_conf = 0.3
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_"))
    tracked_detections = []

    for frame_file, frame_raw in zip(frame_files, raw_detections):
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        deepsort_input = [(d["xywh"], d["conf"], "face", d["kps"]) for d in frame_raw]
        others = [[[d["xywh"][0], d["xywh"][1], d["xywh"][0] + d["xywh"][2], d["xywh"][1] + d["xywh"][3]], d["conf"], d["kps"]] for d in frame_raw]
        tracks = tracker.update_tracks(deepsort_input, frame=frame, others=others)

        frame_tracked = []
        for track in tracks:
            supp = track.get_det_supplementary()
            if track.is_confirmed() and track.time_since_update == 0:
                x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
            else:
                if supp is None or supp[1] < min_conf:
                    continue
                x1, y1, x2, y2 = [int(v) for v in supp[0]]
            frame_tracked.append({"bbox": [x1, y1, x2, y2], "kps": supp[2] if supp else None})
        tracked_detections.append(frame_tracked)

    total_tracked = sum(len(f) for f in tracked_detections)
    logger = Task.current_task().get_logger()
    logger.report_single_value("total_tracked_detections", total_tracked)
    logger.report_single_value("avg_faces_per_frame", total_tracked / len(tracked_detections) if tracked_detections else 0.0)
    logger.report_single_value("frames_with_faces", sum(1 for f in tracked_detections if f))

    out_path = os.path.join(frames_dir, "tracked_detections.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(tracked_detections, f)
    return out_path


def align_faces(frames_dir: str, tracked_detections_path: str):
    import os
    import pickle
    import numpy as np
    import cv2
    from pathlib import Path
    from clearml import Task
    from insightface.utils.face_align import norm_crop

    with open(tracked_detections_path, "rb") as f:
        tracked_detections = pickle.load(f)

    crops_dir = os.path.join(frames_dir, "crops")
    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_"))
    alignment: dict = {}

    for frame_idx, (frame_file, frame_dets) in enumerate(zip(frame_files, tracked_detections)):
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        for det_idx, det in enumerate(frame_dets):
            kps = det.get("kps")
            if kps is None:
                continue
            crop = norm_crop(frame, np.asarray(kps))
            crop_path = os.path.join(crops_dir, f"crop_{frame_idx:06d}_{det_idx:04d}.jpg")
            cv2.imwrite(crop_path, crop)
            alignment[(frame_idx, det_idx)] = crop_path

    crops_produced = len(alignment)
    total_detections = sum(len(f) for f in tracked_detections)
    logger = Task.current_task().get_logger()
    logger.report_single_value("crops_produced", crops_produced)
    logger.report_single_value("landmark_skipped", total_detections - crops_produced)

    alignment_path = os.path.join(frames_dir, "alignment.pkl")
    with open(alignment_path, "wb") as f:
        pickle.dump(alignment, f)
    return crops_dir, alignment_path


def blur_faces(
    frames_dir: str,
    tracked_detections_path: str,
    alignment_path: str,
    whitelist_dir: str,
    video_path: str,
    fps: int,
    frame_width: int,
    frame_height: int,
    out_dir: str,
    stem: str,
    device_id: int = 0,
):
    import os
    import pickle
    import subprocess
    import numpy as np
    import cv2
    import onnxruntime as ort
    import imageio_ffmpeg
    from insightface.app import FaceAnalysis
    from clearml import Task
    from face_blur.services.face_recognition import FaceRecognizer

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers, ctx_id = [("CUDAExecutionProvider", {"device_id": device_id})], device_id
    elif "CoreMLExecutionProvider" in available:
        providers, ctx_id = [("CoreMLExecutionProvider", {"ml_program_compile_on_demand": True})], 0
    else:
        providers, ctx_id = ["CPUExecutionProvider"], -1

    face_app = FaceAnalysis(name="buffalo_l", root=".", allowed_modules=["detection", "landmark_2d_106"], providers=providers)
    face_app.prepare(ctx_id=ctx_id)
    face_recognizer = FaceRecognizer(face_app, whitelist_dir)

    with open(tracked_detections_path, "rb") as f:
        tracked_detections = pickle.load(f)
    with open(alignment_path, "rb") as f:
        alignment = pickle.load(f)

    def _apply_blur(frame, x1, y1, x2, y2):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2, np.newaxis] == 255, cv2.GaussianBlur(roi, (99, 99), 30), roi)
        return frame

    blurred_path = os.path.join(out_dir, f"blurred_{stem}.mp4")
    final_path = os.path.join(out_dir, f"processed_{stem}.mp4")
    writer = cv2.VideoWriter(blurred_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_"))

    faces_blurred = faces_whitelisted = 0
    for frame_idx, (frame_file, frame_dets) in enumerate(zip(frame_files, tracked_detections)):
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        for det_idx, det in enumerate(frame_dets):
            crop_path = alignment.get((frame_idx, det_idx))
            if crop_path is None:
                continue
            crop = cv2.imread(crop_path)
            if crop is None:
                continue
            if face_recognizer.is_whitelisted(crop):
                faces_whitelisted += 1
            else:
                x1, y1, x2, y2 = det["bbox"]
                frame = _apply_blur(frame, x1, y1, x2, y2)
                faces_blurred += 1
        writer.write(frame)
    writer.release()

    total_faces = faces_blurred + faces_whitelisted
    logger = Task.current_task().get_logger()
    logger.report_single_value("faces_blurred", faces_blurred)
    logger.report_single_value("faces_whitelisted", faces_whitelisted)
    logger.report_single_value("whitelist_rate", faces_whitelisted / total_faces if total_faces else 0.0)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-y", "-i", blurred_path, "-i", video_path, "-map", "0:v", "-map", "1:a?", "-c", "copy", "-shortest", final_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    os.remove(blurred_path)
    return final_path


def upload_video(local_file_path: str, file_key: str, result_path: str):
    """Upload processed video; write result to result_path so the parent process can read it."""
    import json
    from face_blur.services.video_storage import upload_local_file_to_blob

    output_url, output_path = upload_local_file_to_blob(local_file_path=local_file_path, file_key=file_key)
    with open(result_path, "w") as f:
        json.dump({"output_url": output_url, "output_path": output_path}, f)
    return output_url, output_path


# ── Task wrapper ──────────────────────────────────────────────────────────────

class VideoProcessingTask:
    def __init__(
        self,
        file_key: str,
        in_dir: str = VIDEO_IN_DIR,
        out_dir: str = VIDEO_OUT_DIR,
        device_id: int = 0,
    ):
        self.file_key = str(file_key)
        self.in_dir = f"{in_dir}/{self.file_key}"
        self.out_dir = f"{out_dir}/{self.file_key}"
        self.device_id = device_id

        self.record = None
        self.clearml_task = None
        self.logger = None

        self.progress = {"percentage": 0, "eta": "00:00:00"}
        self.progress_event = threading.Event()

        _progress_state[self.file_key] = self.progress
        _progress_events[self.file_key] = self.progress_event

        Path(self.in_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def _build_pipeline(self, filename: str, stem: str) -> PipelineController:
        from face_blur.services.video_storage import upload_local_file_to_blob

        frames_dir = os.path.join(self.out_dir, "frames")
        file_key = self.file_key
        device_id = self.device_id
        cb = _progress_callback  # shorthand

        pipe = PipelineController(
            name=f"process-{filename}",
            project="face-blur-pipeline",
            version="1.0",
            add_pipeline_tags=False,
        )

        pipe.add_function_step(
            name="fetch_video",
            function=fetch_video,
            function_kwargs={"file_key": file_key, "filename": filename, "in_dir": self.in_dir},
            function_return=["video_path", "whitelist_dir", "fps", "total_frames", "frame_width", "frame_height"],
            post_execute_callback=cb(file_key, 10),
        )
        pipe.add_function_step(
            name="extract_frames",
            function=extract_frames,
            function_kwargs={"video_path": "${fetch_video.video_path}", "frames_dir": frames_dir},
            function_return=["frames_dir"],
            parents=["fetch_video"],
            post_execute_callback=cb(file_key, 20),
        )
        pipe.add_function_step(
            name="detect_faces",
            function=detect_faces,
            function_kwargs={
                "frames_dir": "${extract_frames.frames_dir}",
                "total_frames": "${fetch_video.total_frames}",
                "device_id": device_id,
            },
            function_return=["raw_detections_path"],
            parents=["extract_frames"],
            post_execute_callback=cb(file_key, 50),
        )
        pipe.add_function_step(
            name="track_faces",
            function=track_faces,
            function_kwargs={
                "frames_dir": "${extract_frames.frames_dir}",
                "raw_detections_path": "${detect_faces.raw_detections_path}",
                "fps": "${fetch_video.fps}",
            },
            function_return=["tracked_detections_path"],
            parents=["detect_faces"],
            post_execute_callback=cb(file_key, 65),
        )
        pipe.add_function_step(
            name="align_faces",
            function=align_faces,
            function_kwargs={
                "frames_dir": "${extract_frames.frames_dir}",
                "tracked_detections_path": "${track_faces.tracked_detections_path}",
            },
            function_return=["crops_dir", "alignment_path"],
            parents=["track_faces"],
            post_execute_callback=cb(file_key, 75),
        )
        pipe.add_function_step(
            name="blur_faces",
            function=blur_faces,
            function_kwargs={
                "frames_dir": "${extract_frames.frames_dir}",
                "tracked_detections_path": "${track_faces.tracked_detections_path}",
                "alignment_path": "${align_faces.alignment_path}",
                "whitelist_dir": "${fetch_video.whitelist_dir}",
                "video_path": "${fetch_video.video_path}",
                "fps": "${fetch_video.fps}",
                "frame_width": "${fetch_video.frame_width}",
                "frame_height": "${fetch_video.frame_height}",
                "out_dir": self.out_dir,
                "stem": stem,
                "device_id": device_id,
            },
            function_return=["final_path"],
            parents=["align_faces"],
            post_execute_callback=cb(file_key, 90),
        )
        result_path = os.path.join(self.out_dir, "upload_result.json")
        pipe.add_function_step(
            name="upload_video",
            function=upload_video,
            function_kwargs={
                "local_file_path": "${blur_faces.final_path}",
                "file_key": file_key,
                "result_path": result_path,
            },
            function_return=["output_url", "output_path"],
            parents=["blur_faces"],
            post_execute_callback=cb(file_key, 100),
        )
        pipe._result_path = result_path

        return pipe

    def process(self):
        from django.utils import timezone
        from face_blur.models import FileMetadata
        from face_blur.services.video_storage import generate_download_sas_url

        self.record = FileMetadata.objects.get(file_key=self.file_key)
        filename = self.record.original_file_name
        stem = Path(filename).stem

        self.record.status = FileMetadata.Status.PROCESSING
        self.record.save(update_fields=["status"])

        try:
            # PipelineController creates its own controller-type task, which is
            # what registers it on the ClearML Pipelines page. Don't call
            # Task.init() before this — it would conflict with pipeline task creation.
            pipe = self._build_pipeline(filename, stem)
            self.clearml_task = pipe._task
            self.logger = self.clearml_task.get_logger()
            self.clearml_task.connect({"file_key": self.file_key, "filename": filename})

            self.record.task_id = self.clearml_task.id
            self.record.save(update_fields=["task_id"])

            pipe.start_locally(run_pipeline_steps_locally=True)

            import json
            with open(pipe._result_path) as f:
                result = json.load(f)
            output_url, output_path = result["output_url"], result["output_path"]

            self.record.status = FileMetadata.Status.COMPLETED
            self.record.download_blob_url = generate_download_sas_url(output_path)
            self.record.date_processing_finished = timezone.localtime()
            self.record.save(update_fields=["status", "download_blob_url", "date_processing_finished"])

        except Exception as e:
            if self.record is not None:
                self.record.status = FileMetadata.Status.FAILED
                self.record.save(update_fields=["status"])
            if self.logger is not None:
                self.logger.report_text(f"Processing failed: {e}")
            raise

        finally:
            self._cleanup()

    def _cleanup(self):
        import shutil
        for directory in (self.in_dir, self.out_dir):
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
        _progress_state.pop(self.file_key, None)
        _progress_events.pop(self.file_key, None)
        _upload_results.pop(self.file_key, None)
        if self.clearml_task is not None:
            self.clearml_task.close()


def trigger_video_processing(file_key: str) -> VideoProcessingTask:
    task = VideoProcessingTask(file_key=file_key)
    worker = threading.Thread(target=task.process, daemon=False)
    worker.start()
    return task
