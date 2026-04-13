import os
import time
import subprocess
from pathlib import Path
import threading
import onnxruntime as ort

import cv2
import imageio_ffmpeg
import numpy as np
from clearml import Task
from django.utils import timezone
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

from face_blur.models import FileMetadata
from face_blur.services.face_recognition import FaceRecognizer
from face_blur.services.video_storage import (
    download_blob_by_name,
    download_blobs,
    upload_local_file_to_blob, generate_download_sas_url
)
from .face_detection import Detection

VIDEOS_DIR = "./videos"
VIDEO_IN_DIR = f"{VIDEOS_DIR}/in"
VIDEO_OUT_DIR = f"{VIDEOS_DIR}/out"
BLURRED_SUFFIX = "blurred_"
PROCESSED_SUFFIX = "processed_"
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


def get_face_det_app(device_id: int = 0) -> FaceAnalysis:
    # 1. Detect what this specific machine supports
    available_providers = ort.get_available_providers()

    providers = []
    ctx_id = 0  # Default context ID for GPU

    if "CUDAExecutionProvider" in available_providers:
        # Best for NVIDIA GPUs
        providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        ctx_id = device_id
        print("Running on NVIDIA GPU (CUDA)")

    elif "CoreMLExecutionProvider" in available_providers:
        # Best for Mac M1/M2/M3/M4
        providers.append(("CoreMLExecutionProvider", {
            "ml_program_compile_on_demand": True,
        }))
        ctx_id = 0
        print("Running on Apple Silicon (CoreML)")

    else:
        # Fallback for machines without a supported GPU
        providers.append("CPUExecutionProvider")
        ctx_id = -1
        print("Running on CPU")

    # 2. Initialize the app with the detected providers
    _face_det_app = FaceAnalysis(
        name="buffalo_l",
        root=".",
        allowed_modules=["detection", "landmark_2d_106"],
        providers=providers,
    )

    # 3. Prepare the model
    # Note: InsightFace uses ctx_id < 0 to force CPU mode
    _face_det_app.prepare(ctx_id=ctx_id)

    return _face_det_app


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
        self.filename = None
        self.video_path = None
        self.whitelist_dir = None
        self.blurred_filepath = None
        self.final_output_path = None

        self.task = None
        self.logger = None
        self.cap = None
        self.out = None

        self.face_det_app = None
        self.detector = None
        self.face_recognizer = None

        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.total_frames = None

        self.start_time = None
        self.detected_faces = 0
        self.whitelisted_faces = 0
        self.blurred_faces = 0

        self.progress = {"percentage": 0, "eta": 0}
        self.progress_event = threading.Event()

        Path(self.in_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)


    def setup(self):
        self.record = FileMetadata.objects.get(file_key=self.file_key)
        self.filename = self.record.original_file_name

        stem = Path(self.filename).stem
        self.blurred_filepath = os.path.join(self.out_dir, f"{BLURRED_SUFFIX}{stem}.mp4")
        self.final_output_path = os.path.join(self.out_dir, f"{PROCESSED_SUFFIX}{stem}.mp4")

        self.task = Task.init(
            project_name="face-blur-inference",
            task_name=f"process-{self.filename}",
            reuse_last_task_id=False,
        )
        self.logger = self.task.get_logger()

        self.record.status = FileMetadata.Status.PROCESSING
        self.record.task_id = self.task.id
        self.record.save(update_fields=["status", "task_id"])

        self.video_path = download_blob_by_name(self.file_key, self.filename, self.in_dir)
        self.whitelist_dir = os.path.join(self.in_dir, "whitelist")
        download_blobs(self.file_key, self.whitelist_dir)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS) or 0))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.blurred_filepath,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        self.face_det_app = get_face_det_app(self.device_id)
        self.detector = Detection(self.face_det_app, self.fps)
        self.face_recognizer = FaceRecognizer(self.face_det_app, self.whitelist_dir)

        self.task.connect(
            {
                "file_key": self.file_key,
                "filename": self.filename,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
            }
        )

    def run_detection(self, frame):
        start = time.perf_counter()
        detections = self.detector.detect(frame)
        elapsed = time.perf_counter() - start

        self.detected_faces += len(detections)
        return detections, elapsed

    def run_recognition(self, crop):
        start = time.perf_counter()
        recognized = self.face_recognizer.is_whitelisted(crop)
        elapsed = time.perf_counter() - start

        if recognized:
            self.whitelisted_faces += 1
        return recognized, elapsed

    def process(self):
        self.setup()
        self.start_time = time.perf_counter()

        frame_count = 0
        detection_time_total = 0.0
        recognition_time_total = 0.0

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if frame_count >= self.total_frames:
                        break
                    raise RuntimeError(
                        f"Failed to read frame {frame_count} from video {self.filename}"
                    )

                frame_count += 1
                detections, det_time = self.run_detection(frame)
                detection_time_total += det_time

                for detection in detections:
                    kps = detection.get("kps")
                    if kps is None:
                        continue

                    crop = norm_crop(frame, np.asarray(kps))
                    recognized, rec_time = self.run_recognition(crop)
                    recognition_time_total += rec_time

                    if not recognized:
                        x1, y1, x2, y2 = detection["bbox"]
                        frame = apply_blur(frame, x1, y1, x2, y2)
                        self.blurred_faces += 1

                self.out.write(frame)

                if frame_count % 50 == 0:
                    self.progress = self.get_progress(frame_count)
                    self.progress_event.set()

                    self.logger.report_text(
                        f"frame={frame_count}/{self.total_frames} eta={self.progress['eta']}"
                    )

            self.cap.release()
            self.out.release()

            # mark progress as complete
            self.progress["percentage"] = 100
            self.progress_event.set()

            self.add_audio()
            output_url, output_path = upload_local_file_to_blob(
                local_file_path=self.final_output_path,
                file_key=self.file_key, )

            total_time = time.perf_counter() - self.start_time

            self.logger.report_single_value("total_inference_time_seconds", total_time)
            self.logger.report_single_value("detection_time_seconds", detection_time_total)
            self.logger.report_single_value("recognition_time_seconds", recognition_time_total)
            self.logger.report_single_value("frames_processed", frame_count)
            self.logger.report_single_value("faces_detected", self.detected_faces)
            self.logger.report_single_value("faces_whitelisted", self.whitelisted_faces)
            self.logger.report_single_value("faces_blurred", self.blurred_faces)

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
            self.cleanup()

    def get_progress(self, frame_count: int) -> dict:
        elapsed = time.perf_counter() - self.start_time
        frames_remaining = max(0, self.total_frames - frame_count)
        average_processing_fps = frame_count / elapsed if elapsed > 0 else 0.0
        eta = frames_remaining / average_processing_fps if average_processing_fps > 0 else 0.0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

        return {
            "percentage": int((100 * frame_count) / self.total_frames) if self.total_frames else 0,
            "eta": eta_str,
        }

    def add_audio(self):
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", self.blurred_filepath,
            "-i", self.video_path,
            "-map", "0:v",
            "-map", "1:a?",
            "-c", "copy",
            "-shortest",
            self.final_output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Processing failed: {result.stderr}")

    def cleanup(self):
        ## TODO: Delete local files and files from storage
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()

        if os.path.exists(self.blurred_filepath or ""):
            os.remove(self.blurred_filepath)

        if self.task is not None:
            self.task.close()


def trigger_video_processing(file_key: str) -> VideoProcessingTask:
    task = VideoProcessingTask(file_key=file_key)
    worker = threading.Thread(
        target=task.process,
        daemon=False,  # do not terminate if task is running
    )
    worker.start()
    return task
