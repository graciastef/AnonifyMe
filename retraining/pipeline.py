"""
SCRFD retraining pipeline.

Strategy: warm-start from the published SCRFD-10GF checkpoint each run.
Dataset: pre-populated WiderFace (retraining/dataset/) + accumulated failure cases from Azure.
No image downloading — WiderFace images and SCRFD keypoint labels are already local.

Pipeline steps (all run locally on a single GPU worker):
  1. fetch_training_data   — pull failure cases + pretrained checkpoint from Azure
  2. prepare_dataset  ─┐
     hpo_search       ─┘  run in parallel after fetch_training_data
  3. train_scrfd           — warm-start with best HPs on combined dataset
  4. evaluate_and_tag      — export ONNX, compare mAP@0.5 vs production, tag 'candidate'
  5. upload_model          — upload ONNX to Azure models/scrfd/

IMPORTANT — ClearML pipeline step scope:
  Each step function is serialized to a temp script and run in a subprocess.
  Module-level names (constants, helpers) are NOT available inside step functions.
  All constants must be defined as locals; all helpers must be nested defs.
  Path constants that depend on __file__ must be computed in _build() and passed
  as explicit parameters, since __file__ in the temp script points to /tmp/....

One-time setup:
  1. Upload the published SCRFD-10GF checkpoint to Azure at
     models/scrfd/pretrained/scrfd_10g.pth
  2. Ensure WiderFace images and SCRFD labelv2 files are under retraining/dataset/.

Failure case format in Azure at training/failure_cases/:
  images/<file_key>_<ts>_<n>.jpg
  annotations.jsonl  — written by inference pipeline; humans delete non-face lines.
    {"image": "<stem>.jpg", "bboxes": [{"bbox": [x1, y1, x2, y2], "conf": <float>}, ...]}
"""

import os
from pathlib import Path

from clearml.automation.controller import PipelineController

# ── Module-level reference values (NOT used inside step functions) ─────────────
# Step functions must redefine what they need locally.

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET_DIR       = os.path.join(_HERE, "dataset")
SCRFD_LABEL_TRAIN  = os.path.join(_DATASET_DIR, "scrfd_label", "train", "labelv2.txt")
SCRFD_LABEL_VAL    = os.path.join(_DATASET_DIR, "scrfd_label", "val",   "labelv2.txt")
WIDER_TRAIN_DIR    = os.path.join(_DATASET_DIR, "WIDER_train", "images")
WIDER_VAL_DIR      = os.path.join(_DATASET_DIR, "WIDER_val",   "images")
INSIGHTFACE_DIR    = "./insightface_repo"
PUBLISHED_LR           = 0.01
PUBLISHED_WEIGHT_DECAY = 0.0005
PUBLISHED_WARMUP_ITERS = 1500
HPO_MAX_JOBS       = 5
HPO_TRIAL_EPOCHS   = 3
INSIGHTFACE_REF    = "f8613d444c6c266e8ff2fb29676a0a1cba6ee7a1"


# ── Step 1: fetch_training_data ────────────────────────────────────────────────

def fetch_training_data(output_dir: str) -> tuple[str, int, str]:
    """Downloads failure cases and pretrained checkpoint from Azure."""
    import os
    from pathlib import Path
    from clearml import Task
    from dotenv import load_dotenv

    failure_cases_blob_prefix  = "training/failure_cases"
    pretrained_checkpoint_blob = "models/scrfd/pretrained/scrfd_10g.pth"

    load_dotenv()
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    svc       = BlobServiceClient(account_url=os.environ["AZURE_BLOB_URL"], credential=DefaultAzureCredential())
    container = svc.get_container_client(os.environ["AZURE_CONTAINER_NAME"])

    def _download(blob_name, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(container.get_blob_client(blob_name).download_blob().readall())
        return local_path

    failure_cases_dir = os.path.join(output_dir, "failure_cases")
    Path(os.path.join(failure_cases_dir, "images")).mkdir(parents=True, exist_ok=True)

    n_images = 0
    for blob in container.list_blobs(name_starts_with=failure_cases_blob_prefix):
        rel   = blob.name[len(failure_cases_blob_prefix):].lstrip("/")
        local = os.path.join(failure_cases_dir, rel)
        _download(blob.name, local)
        if rel.startswith("images/") and rel.lower().endswith((".jpg", ".jpeg", ".png")):
            n_images += 1

    pretrained_path = os.path.join(output_dir, "pretrained_scrfd_10g.pth")
    _download(pretrained_checkpoint_blob, pretrained_path)

    Task.current_task().get_logger().report_single_value("failure_cases_fetched", n_images)
    return failure_cases_dir, n_images, pretrained_path


# ── Step 2a: prepare_dataset ───────────────────────────────────────────────────

def prepare_dataset(
    failure_cases_dir: str,
    output_dir: str,
    scrfd_label_train: str,
    scrfd_label_val: str,
) -> tuple[str, str, str, str, dict]:
    """
    Appends failure cases to the existing SCRFD labelv2 train file.
    Failure case split is by source video (file_key) to prevent person leakage.

    Returns:
        train_label_path        — WiderFace train + failure case train
        combined_val_label_path — WiderFace val + failure case val
        fc_val_label_path       — failure case val only (recovery rate metric)
        wf_val_label_path       — WiderFace val only (unchanged from scrfd_label_val)
        stats
    """
    import os
    import json
    import shutil
    import cv2
    from pathlib import Path
    from collections import defaultdict
    from clearml import Task

    proc_dir = os.path.join(output_dir, "processed")
    Path(proc_dir).mkdir(parents=True, exist_ok=True)

    train_label_path  = os.path.join(proc_dir, "train_label.txt")
    fc_val_label_path = os.path.join(proc_dir, "fc_val_label.txt")
    combined_val_label_path = os.path.join(proc_dir, "combined_val_label.txt")
    split_manifest_path = os.path.join(proc_dir, "failure_case_split.json")

    shutil.copy2(scrfd_label_train, train_label_path)
    Path(fc_val_label_path).write_text("")
    shutil.copy2(scrfd_label_val, combined_val_label_path)

    def _split_by_video(records, train_frac=0.9):
        groups = defaultdict(list)
        for rec in records:
            stem      = rec["image"].rsplit(".", 1)[0]
            video_key = stem.rsplit("_", 2)[0]   # strip _timestamp_index; file_key has hyphens only
            groups[video_key].append(rec)
        video_keys = sorted(groups)
        split_idx  = max(1, int(len(video_keys) * train_frac))
        train_keys = set(video_keys[:split_idx])
        train_recs = [r for k, recs in groups.items() for r in recs if k in train_keys]
        val_recs   = [r for k, recs in groups.items() for r in recs if k not in train_keys]
        return train_recs, val_recs

    def _append_labels(records, label_path):
        no_kps = "-1 -1 -1  -1 -1 -1  -1 -1 -1  -1 -1 -1  -1 -1 -1"
        total  = 0
        with open(label_path, "a") as fh:
            for rec in records:
                img_path = os.path.abspath(os.path.join(failure_cases_dir, "images", rec["image"]))
                if not Path(img_path).exists():
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                bboxes = rec.get("bboxes", [])
                if not bboxes:
                    continue
                lines = []
                for entry in bboxes:
                    x1, y1, x2, y2 = entry["bbox"] if isinstance(entry, dict) else entry
                    x1 = max(0.0, min(float(x1), float(img_w - 1)))
                    y1 = max(0.0, min(float(y1), float(img_h - 1)))
                    x2 = max(0.0, min(float(x2), float(img_w - 1)))
                    y2 = max(0.0, min(float(y2), float(img_h - 1)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    lines.append(f"{x1} {y1} {x2} {y2}  {no_kps}\n")
                if lines:
                    fh.write(f"# {img_path} {img_w} {img_h}\n")
                    fh.writelines(lines)
                    total += len(lines)
        return total

    def _count_label_file(label_path):
        images = bboxes = 0
        if not Path(label_path).exists():
            return images, bboxes
        with open(label_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    images += 1
                else:
                    bboxes += 1
        return images, bboxes

    fc_train_n = fc_val_n = 0
    split_manifest = {
        "split_strategy": "by_video_key",
        "train_fraction": 0.9,
        "train_images": [],
        "val_images": [],
    }
    annotations_path = os.path.join(failure_cases_dir, "annotations.jsonl")
    if Path(annotations_path).exists():
        with open(annotations_path) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        fc_train_recs, fc_val_recs = _split_by_video(records)
        fc_train_n = _append_labels(fc_train_recs, train_label_path)
        fc_val_n   = _append_labels(fc_val_recs,   fc_val_label_path)
        _append_labels(fc_val_recs, combined_val_label_path)
        split_manifest["train_images"] = sorted({rec["image"] for rec in fc_train_recs})
        split_manifest["val_images"] = sorted({rec["image"] for rec in fc_val_recs})

    with open(split_manifest_path, "w") as fh:
        json.dump(split_manifest, fh, indent=2, sort_keys=True)

    wider_train_images, wider_train_bboxes = _count_label_file(scrfd_label_train)
    wider_val_images, wider_val_bboxes = _count_label_file(scrfd_label_val)

    stats = {
        "wider_train_images": wider_train_images,
        "wider_train_bboxes": wider_train_bboxes,
        "wider_val_images": wider_val_images,
        "wider_val_bboxes": wider_val_bboxes,
        "failure_cases_train": fc_train_n,
        "failure_cases_val": fc_val_n,
        "failure_case_train_images": len(split_manifest["train_images"]),
        "failure_case_val_images": len(split_manifest["val_images"]),
        "combined_train_images": wider_train_images + len(split_manifest["train_images"]),
        "combined_train_bboxes": wider_train_bboxes + fc_train_n,
    }
    logger = Task.current_task().get_logger()
    for k, v in stats.items():
        logger.report_single_value(k, v)

    return train_label_path, combined_val_label_path, fc_val_label_path, scrfd_label_val, stats


# ── Step 2c: upload_dataset_version ────────────────────────────────────────────

def upload_dataset_version(
    train_label_path: str,
    combined_val_label_path: str,
    fc_val_label_path: str,
    dataset_stats: dict,
    version: str,
    insightface_ref: str,
) -> str:
    """Uploads generated labels and split metadata for this training version."""
    import os
    import json
    from pathlib import Path
    from clearml import Task
    from dotenv import load_dotenv

    dataset_blob_dir = f"training/datasets/{version}"

    load_dotenv()
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings

    svc       = BlobServiceClient(account_url=os.environ["AZURE_BLOB_URL"], credential=DefaultAzureCredential())
    container = svc.get_container_client(os.environ["AZURE_CONTAINER_NAME"])

    def _upload_file(local_path, blob_name, content_type="text/plain"):
        if not local_path or not Path(local_path).exists():
            return None
        with open(local_path, "rb") as fh:
            container.get_blob_client(blob_name).upload_blob(
                fh, overwrite=False,
                content_settings=ContentSettings(content_type=content_type),
            )
        return blob_name

    processed_dir = Path(train_label_path).parent
    split_manifest_path = processed_dir / "failure_case_split.json"
    manifest_path = processed_dir / "dataset_manifest.json"
    manifest = {
        "version": version,
        "insightface_ref": insightface_ref,
        "stats": dataset_stats,
        "files": {
            "train_label": "train_label.txt",
            "combined_val_label": "combined_val_label.txt",
            "fc_val_label": "fc_val_label.txt",
            "failure_case_split": "failure_case_split.json",
        },
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    uploaded = []
    for local_path, filename, content_type in [
        (train_label_path, "train_label.txt", "text/plain"),
        (combined_val_label_path, "combined_val_label.txt", "text/plain"),
        (fc_val_label_path, "fc_val_label.txt", "text/plain"),
        (str(split_manifest_path), "failure_case_split.json", "application/json"),
        (str(manifest_path), "dataset_manifest.json", "application/json"),
    ]:
        blob_name = _upload_file(local_path, f"{dataset_blob_dir}/{filename}", content_type)
        if blob_name:
            uploaded.append(blob_name)

    logger = Task.current_task().get_logger()
    logger.report_text("Uploaded dataset version:\n" + "\n".join(uploaded))
    logger.report_single_value("dataset_version_files_uploaded", len(uploaded))
    return dataset_blob_dir


# ── Step 2b: hpo_search ───────────────────────────────────────────────────────

def hpo_search(
    pretrained_path: str,
    train_label_path: str,
    val_label_path: str,
    output_dir: str,
    wider_train_dir: str,
    wider_val_dir: str,
    insightface_dir: str,
    insightface_ref: str,
    hpo_max_jobs: int = 5,
    hpo_trial_epochs: int = 3,
) -> tuple[float, float, int]:
    """
    Tunes hyperparameters on the combined train labels generated by prepare_dataset.
    Validation also uses the combined validation labels.

    Published params are enqueued as trial 0 so TPE starts from a known-good point.
    """
    import os
    import subprocess
    import sys
    import threading
    from pathlib import Path
    import optuna
    from clearml import Task

    published_lr           = 0.01
    published_weight_decay = 0.0005
    published_warmup_iters = 1500
    insightface_repo_url   = "https://github.com/deepinsight/insightface.git"
    scrfd_subdir           = "detection/scrfd"
    scrfd_config           = f"{scrfd_subdir}/configs/scrfd/scrfd_10g.py"
    scrfd_train_script     = f"{scrfd_subdir}/tools/train.py"
    hpo_dir                = os.path.abspath(os.path.join(output_dir, "hpo"))
    Path(hpo_dir).mkdir(parents=True, exist_ok=True)

    logger = Task.current_task().get_logger()
    logger.report_single_value("hpo_max_jobs", hpo_max_jobs)
    logger.report_single_value("hpo_trial_epochs", hpo_trial_epochs)

    def _scrfd_env():
        env = os.environ.copy()
        scrfd_root = os.path.join(insightface_dir, scrfd_subdir)
        env["PYTHONPATH"] = scrfd_root + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _run_streamed(command, cwd, env):
        stdout_chunks = []
        stderr_chunks = []

        def _reader(pipe, sink, target):
            for line in iter(pipe.readline, ""):
                sink.write(line)
                sink.flush()
                target.append(line)
            pipe.close()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )
        stdout_thread = threading.Thread(target=_reader, args=(process.stdout, sys.stdout, stdout_chunks))
        stderr_thread = threading.Thread(target=_reader, args=(process.stderr, sys.stderr, stderr_chunks))
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        return returncode, stdout, stderr

    def _ensure_repo():
        if Path(insightface_dir).exists():
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=insightface_dir,
            )
            target = subprocess.run(
                ["git", "rev-parse", "--verify", f"{insightface_ref}^{{commit}}"],
                capture_output=True, text=True, cwd=insightface_dir,
            )
            if head.returncode == 0 and target.returncode == 0 and head.stdout.strip() == target.stdout.strip():
                scrfd_root = Path(insightface_dir) / scrfd_subdir
                if not scrfd_root.exists():
                    raise FileNotFoundError(f"{insightface_dir} exists but {scrfd_root} is missing")
                return
        else:
            Path(insightface_dir).parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["git", "clone", "--depth=1", insightface_repo_url, insightface_dir], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Could not clone InsightFace into {insightface_dir}") from exc

        try:
            subprocess.run(["git", "fetch", "--depth=1", "origin", insightface_ref], check=True, cwd=insightface_dir)
            subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], check=True, cwd=insightface_dir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Could not checkout InsightFace ref {insightface_ref!r}") from exc

        scrfd_root = Path(insightface_dir) / scrfd_subdir
        if not scrfd_root.exists():
            raise FileNotFoundError(f"{insightface_dir} exists but {scrfd_root} is missing")

    def _run_trial(lr, weight_decay, warmup_iters, trial_num):
        _ensure_repo()
        work_dir = os.path.join(hpo_dir, f"trial_{trial_num}")
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        cfg_options = [
            f"optimizer.lr={lr}",
            f"optimizer.weight_decay={weight_decay}",
            f"lr_config.warmup_iters={warmup_iters}",
            f"total_epochs={hpo_trial_epochs}",
            f"lr_config.step=[{max(1, hpo_trial_epochs)}]",
            "checkpoint_config.interval=1",
            f"load_from='{pretrained_path}'",
            f"data.train.ann_file='{train_label_path}'",
            f"data.train.img_prefix='{wider_train_dir}'",
            f"data.val.ann_file='{val_label_path}'",
            f"data.val.img_prefix='{wider_val_dir}'",
        ]
        returncode, stdout, stderr = _run_streamed(
            [sys.executable, os.path.join(insightface_dir, scrfd_train_script),
             os.path.join(insightface_dir, scrfd_config),
            "--work-dir", work_dir, "--cfg-options", *cfg_options],
            cwd=os.path.join(insightface_dir, scrfd_subdir),
            env=_scrfd_env(),
        )
        if returncode != 0:
            raise RuntimeError(
                f"HPO trial {trial_num} failed with exit code {returncode}.\n"
                f"STDOUT:\n{stdout[-4000:]}\n"
                f"STDERR:\n{stderr[-4000:]}"
            )
        checkpoints = sorted(Path(work_dir).glob("epoch_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint in {work_dir}")
        return str(checkpoints[-1])

    def _load_labelv2(label_path):
        import numpy as np

        images = []
        current = None
        with open(label_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    parts = stripped[1:].strip().split()
                    current = {"image": parts[0], "bboxes": []}
                    images.append(current)
                    continue
                if current is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in stripped.split()[:4]]
                if x2 <= x1 or y2 <= y1:
                    continue
                current["bboxes"].append([x1, y1, x2, y2])
        for item in images:
            item["bboxes"] = np.array(item["bboxes"], dtype=np.float32).reshape((-1, 4))
        return images

    def _resolve_image_path(image_name):
        if os.path.isabs(image_name):
            return image_name
        return os.path.join(wider_val_dir, image_name)

    def _as_detections(result):
        import numpy as np

        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, list):
            if not result:
                return np.zeros((0, 5), dtype=np.float32)
            result = result[0]
        result = np.asarray(result, dtype=np.float32)
        if result.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return result.reshape((-1, result.shape[-1]))[:, :5]

    def _to_metric_inputs(labels, outputs):
        import torch

        preds = []
        targets = []
        for item, result in zip(labels, outputs):
            detections = _as_detections(result)
            if detections.shape[0]:
                boxes = torch.as_tensor(detections[:, :4], dtype=torch.float32)
                scores = torch.as_tensor(detections[:, 4], dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
            gt_boxes = torch.as_tensor(item["bboxes"], dtype=torch.float32)
            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            })
            targets.append({
                "boxes": gt_boxes,
                "labels": torch.zeros((gt_boxes.shape[0],), dtype=torch.int64),
            })
        return preds, targets

    def _compute_recall_50(preds, targets):
        import torch
        from torchvision.ops import box_iou

        total_gt = sum(int(target["boxes"].shape[0]) for target in targets)
        if total_gt == 0:
            return 0.0

        matched_gt = 0
        for pred, target in zip(preds, targets):
            gt_boxes = target["boxes"]
            pred_boxes = pred["boxes"]
            if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
                continue
            order = torch.argsort(pred["scores"], descending=True)
            ious = box_iou(pred_boxes[order], gt_boxes)
            matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
            for pred_idx in range(ious.shape[0]):
                best_iou, best_gt = torch.max(ious[pred_idx], dim=0)
                if best_iou >= 0.5 and not matched[best_gt]:
                    matched[best_gt] = True
            matched_gt += int(matched.sum().item())
        return matched_gt / total_gt

    def _wider_face_ap(preds_list, targets_list, iou_thresh=0.5):
        import numpy as np

        all_entries = []
        for img_idx, pred in enumerate(preds_list):
            boxes  = pred["boxes"].numpy()
            scores = pred["scores"].numpy()
            for box, score in zip(boxes, scores):
                all_entries.append((float(score), img_idx, box))

        total_gt = sum(t["boxes"].shape[0] for t in targets_list)
        if total_gt == 0 or not all_entries:
            return 0.0

        all_entries.sort(key=lambda x: -x[0])
        gt_matched  = [np.zeros(t["boxes"].shape[0], dtype=bool) for t in targets_list]
        gt_boxes_np = [t["boxes"].numpy() for t in targets_list]

        tp_arr = []
        for _, img_idx, box in all_entries:
            gt_boxes = gt_boxes_np[img_idx]
            matched  = gt_matched[img_idx]
            tp = 0
            if len(gt_boxes):
                b     = box[None]
                ix1   = np.maximum(b[:, 0], gt_boxes[:, 0])
                iy1   = np.maximum(b[:, 1], gt_boxes[:, 1])
                ix2   = np.minimum(b[:, 2], gt_boxes[:, 2])
                iy2   = np.minimum(b[:, 3], gt_boxes[:, 3])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                area_b  = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                iou  = inter / np.maximum(area_b + area_gt - inter, 1e-6)
                best = int(np.argmax(iou))
                if iou[best] >= iou_thresh and not matched[best]:
                    matched[best] = True
                    tp = 1
            tp_arr.append(tp)

        tp_arr = np.array(tp_arr, dtype=np.float32)
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)
        prec   = cum_tp / (cum_tp + cum_fp)
        rec    = cum_tp / total_gt
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _compute_detection_metrics(labels, outputs):
        if len(labels) != len(outputs):
            raise ValueError(f"Prediction count {len(outputs)} does not match label image count {len(labels)}")
        if sum(len(item["bboxes"]) for item in labels) == 0:
            return {"ap_50": 0.0, "recall_50": 0.0}

        preds, targets = _to_metric_inputs(labels, outputs)
        return {
            "ap_50": _wider_face_ap(preds, targets),
            "recall_50": float(_compute_recall_50(preds, targets)),
        }

    def _eval_metrics(checkpoint_path, label_path):
        _ensure_repo()
        result_dir = os.path.join(hpo_dir, "eval_results")
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(result_dir, "predictions.pkl")
        returncode, stdout, stderr = _run_streamed(
            [sys.executable, os.path.join(insightface_dir, "detection/scrfd/tools/test.py"),
             os.path.join(insightface_dir, scrfd_config), checkpoint_path,
             "--out", out_path,
             "--cfg-options",
             f"data.test.ann_file='{label_path}'",
             f"data.test.img_prefix='{wider_val_dir}'"],
            cwd=os.path.join(insightface_dir, scrfd_subdir),
            env=_scrfd_env(),
        )
        if returncode != 0:
            raise RuntimeError(
                f"Evaluation failed with exit code {returncode}.\n"
                f"STDOUT:\n{stdout[-4000:]}\n"
                f"STDERR:\n{stderr[-4000:]}"
            )
        import pickle
        with open(out_path, "rb") as fh:
            outputs = pickle.load(fh)
        return _compute_detection_metrics(_load_labelv2(label_path), outputs)

    def objective(trial):
        lr           = trial.suggest_float("lr",           1e-3, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 2e-3, log=True)
        warmup_iters = trial.suggest_int(  "warmup_iters", 750,  2500, step=250)
        ckpt      = _run_trial(lr, weight_decay, warmup_iters, trial.number)
        metrics   = _eval_metrics(ckpt, val_label_path)
        logger.report_scalar("hpo", "ap_50",        metrics["ap_50"],       trial.number)
        logger.report_scalar("hpo", "recall_50",    metrics["recall_50"],   trial.number)
        logger.report_scalar("hpo", "lr",            lr,           trial.number)
        logger.report_scalar("hpo", "weight_decay",  weight_decay, trial.number)
        logger.report_scalar("hpo", "warmup_iters",  warmup_iters, trial.number)
        return metrics["ap_50"]

    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({"lr": published_lr, "weight_decay": published_weight_decay, "warmup_iters": published_warmup_iters})
    study.optimize(objective, n_trials=hpo_max_jobs)

    best   = study.best_params
    lr     = float(best["lr"])
    wd     = float(best["weight_decay"])
    warmup = int(best["warmup_iters"])
    logger.report_single_value("best_lr",           lr)
    logger.report_single_value("best_weight_decay", wd)
    logger.report_single_value("best_warmup_iters", warmup)
    return lr, wd, warmup


# ── Step 3: train_scrfd ────────────────────────────────────────────────────────

def train_scrfd(
    train_label_path: str,
    val_label_path: str,
    pretrained_path: str,
    output_dir: str,
    lr: float,
    weight_decay: float,
    warmup_iters: int,
    wider_train_dir: str,
    wider_val_dir: str,
    insightface_dir: str,
    insightface_ref: str,
    max_epochs: int = 30,
) -> str:
    """
    Warm-starts from the pretrained checkpoint and trains on the combined
    WiderFace + failure case dataset with the best HPs from hpo_search.
    """
    import os
    import subprocess
    import sys
    import threading
    from pathlib import Path
    from clearml import Task

    insightface_repo_url = "https://github.com/deepinsight/insightface.git"
    scrfd_subdir         = "detection/scrfd"
    scrfd_config         = f"{scrfd_subdir}/configs/scrfd/scrfd_10g.py"
    scrfd_train_script   = f"{scrfd_subdir}/tools/train.py"

    def _scrfd_env():
        env = os.environ.copy()
        scrfd_root = os.path.join(insightface_dir, scrfd_subdir)
        env["PYTHONPATH"] = scrfd_root + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _run_streamed(command, cwd, env):
        stdout_chunks = []
        stderr_chunks = []

        def _reader(pipe, sink, target):
            for line in iter(pipe.readline, ""):
                sink.write(line)
                sink.flush()
                target.append(line)
            pipe.close()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )
        stdout_thread = threading.Thread(target=_reader, args=(process.stdout, sys.stdout, stdout_chunks))
        stderr_thread = threading.Thread(target=_reader, args=(process.stderr, sys.stderr, stderr_chunks))
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        return returncode, stdout, stderr

    def _ensure_repo():
        if Path(insightface_dir).exists():
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=insightface_dir,
            )
            target = subprocess.run(
                ["git", "rev-parse", "--verify", f"{insightface_ref}^{{commit}}"],
                capture_output=True, text=True, cwd=insightface_dir,
            )
            if head.returncode == 0 and target.returncode == 0 and head.stdout.strip() == target.stdout.strip():
                scrfd_root = Path(insightface_dir) / scrfd_subdir
                if not scrfd_root.exists():
                    raise FileNotFoundError(f"{insightface_dir} exists but {scrfd_root} is missing")
                return
        else:
            Path(insightface_dir).parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["git", "clone", "--depth=1", insightface_repo_url, insightface_dir], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Could not clone InsightFace into {insightface_dir}") from exc

        try:
            subprocess.run(["git", "fetch", "--depth=1", "origin", insightface_ref], check=True, cwd=insightface_dir)
            subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], check=True, cwd=insightface_dir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Could not checkout InsightFace ref {insightface_ref!r}") from exc

        scrfd_root = Path(insightface_dir) / scrfd_subdir
        if not scrfd_root.exists():
            raise FileNotFoundError(f"{insightface_dir} exists but {scrfd_root} is missing")

    _ensure_repo()

    work_dir = os.path.abspath(os.path.join(output_dir, "scrfd_work_dir"))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    cfg_options = [
        f"optimizer.lr={lr}",
        f"optimizer.weight_decay={weight_decay}",
        f"lr_config.warmup_iters={warmup_iters}",
        f"total_epochs={max_epochs}",
        f"lr_config.step=[{int(max_epochs * 0.69)}, {int(max_epochs * 0.85)}]",
        f"load_from='{pretrained_path}'",
        f"data.train.ann_file='{train_label_path}'",
        f"data.train.img_prefix='{wider_train_dir}'",
        f"data.val.ann_file='{val_label_path}'",
        f"data.val.img_prefix='{wider_val_dir}'",
    ]

    logger = Task.current_task().get_logger()
    returncode, stdout, stderr = _run_streamed(
        [sys.executable, os.path.join(insightface_dir, scrfd_train_script),
         os.path.join(insightface_dir, scrfd_config),
         "--work-dir", work_dir, "--cfg-options", *cfg_options],
        cwd=os.path.join(insightface_dir, scrfd_subdir),
        env=_scrfd_env(),
    )

    if returncode != 0:
        raise RuntimeError(
            f"Training failed with exit code {returncode}.\n"
            f"STDOUT:\n{stdout[-4000:]}\n"
            f"STDERR:\n{stderr[-4000:]}"
        )

    # Parse epoch-averaged losses from training stdout and plot curves.
    import re
    import collections

    _epoch_pat = re.compile(r'Epoch\s+\[(\d+)\]\[\d+/\d+\](.+)')
    _kv_pat    = re.compile(r'(\w+):\s*([\d.eE+\-]+)')
    epoch_losses = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in (stdout + stderr).splitlines():
        m = _epoch_pat.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        for km in _kv_pat.finditer(m.group(2)):
            key = km.group(1)
            if 'loss' in key.lower():
                epoch_losses[epoch][key].append(float(km.group(2)))

    if epoch_losses:
        epochs_sorted = sorted(epoch_losses)
        epoch_means = {
            e: {k: sum(v) / len(v) for k, v in losses.items()}
            for e, losses in epoch_losses.items()
        }
        for epoch in epochs_sorted:
            for key, val in epoch_means[epoch].items():
                logger.report_scalar("Training Loss", key, val, epoch)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            loss_keys = sorted({k for m in epoch_means.values() for k in m})
            fig, ax = plt.subplots(figsize=(10, 5))
            for key in loss_keys:
                ys = [epoch_means[e].get(key, float('nan')) for e in epochs_sorted]
                ax.plot(epochs_sorted, ys, marker='o', markersize=3, label=key)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_path = os.path.join(work_dir, 'loss_curves.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.report_image("Loss Curves", "train", local_path=plot_path)
            logger.report_text(f"Loss curve saved to {plot_path}")
        except Exception as _plot_exc:
            logger.report_text(f"[WARN] Could not generate loss plot: {_plot_exc}")

    checkpoints = sorted(Path(work_dir).glob("best_*.pth")) or sorted(Path(work_dir).glob("epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}")

    best_ckpt = str(checkpoints[-1])
    logger.report_text(f"checkpoint: {best_ckpt}")
    return best_ckpt


# ── Step 4: evaluate_and_tag ───────────────────────────────────────────────────

def evaluate_and_tag(
    checkpoint_path: str,
    wf_val_label_path: str,
    fc_val_label_path: str,
    output_dir: str,
    version: str,
    wider_val_dir: str,
    insightface_dir: str,
    insightface_ref: str,
) -> tuple[float, float, float, bool, str]:
    """
    Exports checkpoint to ONNX, runs four separate evaluations:
      new_wf  — new model on WiderFace val
      new_fc  — new model on failure-case val
      current_wf — production model on WiderFace val
      current_fc — production model on failure-case val
    Promotion decision is based on WiderFace val map@0.5 only.
    """
    import os
    import subprocess
    import sys
    import threading
    from pathlib import Path
    from clearml import Task, OutputModel
    from dotenv import load_dotenv

    current_production_pointer_blob = "models/scrfd/current_production.txt"
    insightface_repo_url = "https://github.com/deepinsight/insightface.git"
    scrfd_subdir         = "detection/scrfd"
    scrfd_config         = f"{scrfd_subdir}/configs/scrfd/scrfd_10g.py"
    scrfd_export_script  = f"{scrfd_subdir}/tools/scrfd2onnx.py"
    eval_dir             = os.path.abspath(os.path.join(output_dir, "eval"))
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    def _scrfd_env():
        env = os.environ.copy()
        scrfd_root = os.path.join(insightface_dir, scrfd_subdir)
        env["PYTHONPATH"] = scrfd_root + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _run_streamed(command, cwd, env):
        stdout_chunks = []
        stderr_chunks = []

        def _reader(pipe, sink, target):
            for line in iter(pipe.readline, ""):
                sink.write(line)
                sink.flush()
                target.append(line)
            pipe.close()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )
        stdout_thread = threading.Thread(target=_reader, args=(process.stdout, sys.stdout, stdout_chunks))
        stderr_thread = threading.Thread(target=_reader, args=(process.stderr, sys.stderr, stderr_chunks))
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        return returncode, stdout, stderr

    def _ensure_repo():
        if Path(insightface_dir).exists():
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=insightface_dir,
            )
            target = subprocess.run(
                ["git", "rev-parse", "--verify", f"{insightface_ref}^{{commit}}"],
                capture_output=True, text=True, cwd=insightface_dir,
            )
            if head.returncode == 0 and target.returncode == 0 and head.stdout.strip() == target.stdout.strip():
                scrfd_root = Path(insightface_dir) / scrfd_subdir
                if not scrfd_root.exists():
                    raise FileNotFoundError(f"{insightface_dir} exists but {scrfd_root} is missing")
                return
        else:
            Path(insightface_dir).parent.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["git", "clone", "--depth=1", insightface_repo_url, insightface_dir], check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Could not clone InsightFace into {insightface_dir}") from exc

        try:
            subprocess.run(["git", "fetch", "--depth=1", "origin", insightface_ref], check=True, cwd=insightface_dir)
            subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], check=True, cwd=insightface_dir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Could not checkout InsightFace ref {insightface_ref!r}") from exc

        scrfd_root = Path(insightface_dir) / scrfd_subdir
        if not scrfd_root.exists():
            raise FileNotFoundError(f"{insightface_dir} exists but {scrfd_root} is missing")

    _ensure_repo()

    def _export_onnx(ckpt, output_name):
        onnx_path = os.path.join(eval_dir, output_name)
        input_img = next(Path(wider_val_dir).rglob("*.jpg"), None)
        if input_img is None:
            raise FileNotFoundError(f"No .jpg files found under {wider_val_dir} for ONNX export input")
        # scrfd2onnx.py defaults to tests/data/t1.jpg when --input-img is not
        # supplied; keep that fixture available even though we pass input_img.
        default_input_img = Path(insightface_dir) / scrfd_subdir / "tests" / "data" / "t1.jpg"
        default_input_img.parent.mkdir(parents=True, exist_ok=True)
        if not default_input_img.exists():
            default_input_img.write_bytes(input_img.read_bytes())
        returncode, stdout, stderr = _run_streamed(
            [sys.executable, os.path.join(insightface_dir, scrfd_export_script),
             os.path.join(insightface_dir, scrfd_config),
            ckpt,
            "--input-img", str(input_img),
            "--output-file", onnx_path],
            cwd=os.path.join(insightface_dir, scrfd_subdir),
            env=_scrfd_env(),
        )
        if returncode != 0:
            raise RuntimeError(f"ONNX export failed:\nSTDOUT:\n{stdout[-2000:]}\nSTDERR:\n{stderr[-2000:]}")
        return onnx_path

    def _load_labelv2(label_path):
        import numpy as np

        images = []
        current = None
        with open(label_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    parts = stripped[1:].strip().split()
                    current = {"image": parts[0], "bboxes": []}
                    images.append(current)
                    continue
                if current is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in stripped.split()[:4]]
                if x2 <= x1 or y2 <= y1:
                    continue
                current["bboxes"].append([x1, y1, x2, y2])
        for item in images:
            item["bboxes"] = np.array(item["bboxes"], dtype=np.float32).reshape((-1, 4))
        return images

    def _resolve_image_path(image_name):
        if os.path.isabs(image_name):
            return image_name
        return os.path.join(wider_val_dir, image_name)

    def _as_detections(result):
        import numpy as np

        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, list):
            if not result:
                return np.zeros((0, 5), dtype=np.float32)
            result = result[0]
        result = np.asarray(result, dtype=np.float32)
        if result.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return result.reshape((-1, result.shape[-1]))[:, :5]

    def _to_metric_inputs(labels, outputs):
        import torch

        preds = []
        targets = []
        for item, result in zip(labels, outputs):
            detections = _as_detections(result)
            if detections.shape[0]:
                boxes = torch.as_tensor(detections[:, :4], dtype=torch.float32)
                scores = torch.as_tensor(detections[:, 4], dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
            gt_boxes = torch.as_tensor(item["bboxes"], dtype=torch.float32)
            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            })
            targets.append({
                "boxes": gt_boxes,
                "labels": torch.zeros((gt_boxes.shape[0],), dtype=torch.int64),
            })
        return preds, targets

    def _compute_recall_50(preds, targets):
        import torch
        from torchvision.ops import box_iou

        total_gt = sum(int(target["boxes"].shape[0]) for target in targets)
        if total_gt == 0:
            return 0.0

        matched_gt = 0
        for pred, target in zip(preds, targets):
            gt_boxes = target["boxes"]
            pred_boxes = pred["boxes"]
            if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
                continue
            order = torch.argsort(pred["scores"], descending=True)
            ious = box_iou(pred_boxes[order], gt_boxes)
            matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
            for pred_idx in range(ious.shape[0]):
                best_iou, best_gt = torch.max(ious[pred_idx], dim=0)
                if best_iou >= 0.5 and not matched[best_gt]:
                    matched[best_gt] = True
            matched_gt += int(matched.sum().item())
        return matched_gt / total_gt

    def _wider_face_ap(preds_list, targets_list, iou_thresh=0.5):
        import numpy as np

        all_entries = []
        for img_idx, pred in enumerate(preds_list):
            boxes  = pred["boxes"].numpy()
            scores = pred["scores"].numpy()
            for box, score in zip(boxes, scores):
                all_entries.append((float(score), img_idx, box))

        total_gt = sum(t["boxes"].shape[0] for t in targets_list)
        if total_gt == 0 or not all_entries:
            return 0.0

        all_entries.sort(key=lambda x: -x[0])
        gt_matched  = [np.zeros(t["boxes"].shape[0], dtype=bool) for t in targets_list]
        gt_boxes_np = [t["boxes"].numpy() for t in targets_list]

        tp_arr = []
        for _, img_idx, box in all_entries:
            gt_boxes = gt_boxes_np[img_idx]
            matched  = gt_matched[img_idx]
            tp = 0
            if len(gt_boxes):
                b     = box[None]
                ix1   = np.maximum(b[:, 0], gt_boxes[:, 0])
                iy1   = np.maximum(b[:, 1], gt_boxes[:, 1])
                ix2   = np.minimum(b[:, 2], gt_boxes[:, 2])
                iy2   = np.minimum(b[:, 3], gt_boxes[:, 3])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                area_b  = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                iou  = inter / np.maximum(area_b + area_gt - inter, 1e-6)
                best = int(np.argmax(iou))
                if iou[best] >= iou_thresh and not matched[best]:
                    matched[best] = True
                    tp = 1
            tp_arr.append(tp)

        tp_arr = np.array(tp_arr, dtype=np.float32)
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)
        prec   = cum_tp / (cum_tp + cum_fp)
        rec    = cum_tp / total_gt
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _compute_detection_metrics(labels, outputs):
        if len(labels) != len(outputs):
            raise ValueError(f"Prediction count {len(outputs)} does not match label image count {len(labels)}")
        if sum(len(item["bboxes"]) for item in labels) == 0:
            return {"ap_50": 0.0, "recall_50": 0.0}

        preds, targets = _to_metric_inputs(labels, outputs)
        return {
            "ap_50": _wider_face_ap(preds, targets),
            "recall_50": float(_compute_recall_50(preds, targets)),
        }

    def _load_scrfd_onnx_class():
        import importlib.util

        module_path = Path(insightface_dir) / scrfd_subdir / "tools" / "scrfd.py"
        spec = importlib.util.spec_from_file_location("scrfd_onnx_eval", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load SCRFD ONNX helper from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SCRFD

    def _eval_onnx(model_path, label_path):
        import cv2

        labels = _load_labelv2(label_path)
        if not labels:
            raise ValueError(f"No images found in label file {label_path}")

        SCRFD = _load_scrfd_onnx_class()
        detector = SCRFD(model_file=model_path)
        detector.prepare(-1, input_size=(640, 640))

        outputs = []
        for item in labels:
            image_path = _resolve_image_path(item["image"])
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read evaluation image {image_path}")
            detections, _ = detector.detect(image, thresh=0.02, input_size=(640, 640))
            outputs.append(detections)
        return _compute_detection_metrics(labels, outputs)

    def _eval_checkpoint(checkpoint_path, label_path, output_name):
        labels = _load_labelv2(label_path)
        if not labels:
            raise ValueError(f"No images found in label file {label_path}")

        result_dir = os.path.join(eval_dir, output_name)
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(result_dir, "predictions.pkl")
        returncode, stdout, stderr = _run_streamed(
            [sys.executable, os.path.join(insightface_dir, "detection/scrfd/tools/test.py"),
             os.path.join(insightface_dir, scrfd_config), checkpoint_path,
             "--out", out_path,
             "--cfg-options",
             f"data.test.ann_file='{label_path}'",
             f"data.test.img_prefix='{wider_val_dir}'"],
            cwd=os.path.join(insightface_dir, scrfd_subdir),
            env=_scrfd_env(),
        )
        if returncode != 0:
            raise RuntimeError(
                f"Evaluation failed with exit code {returncode}.\n"
                f"STDOUT:\n{stdout[-4000:]}\n"
                f"STDERR:\n{stderr[-4000:]}"
            )
        import pickle
        with open(out_path, "rb") as fh:
            outputs = pickle.load(fh)
        return _compute_detection_metrics(labels, outputs)

    def _eval(model_path, label_path, output_name):
        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            return _eval_onnx(model_path, label_path)
        if suffix == ".pth":
            return _eval_checkpoint(model_path, label_path, output_name)
        raise ValueError(f"Unsupported model format for evaluation: {model_path}")

    onnx_path = _export_onnx(checkpoint_path, f"det_10g_{version}.onnx")

    has_fc = Path(fc_val_label_path).exists() and Path(fc_val_label_path).stat().st_size > 0

    new_wf_metrics = _eval(onnx_path, wf_val_label_path, "new_widerface_val")
    new_fc_metrics = {"ap_50": 0.0, "recall_50": 0.0}
    if has_fc:
        new_fc_metrics = _eval(onnx_path, fc_val_label_path, "new_fc_val")

    current_wf_metrics = {"ap_50": 0.0, "recall_50": 0.0}
    current_fc_metrics = {"ap_50": 0.0, "recall_50": 0.0}
    try:
        load_dotenv()
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
        svc       = BlobServiceClient(account_url=os.environ["AZURE_BLOB_URL"], credential=DefaultAzureCredential())
        container = svc.get_container_client(os.environ["AZURE_CONTAINER_NAME"])
        prod_blob = container.get_blob_client(current_production_pointer_blob).download_blob().readall().decode().strip()
        prod_suffix = Path(prod_blob).suffix.lower()
        if prod_suffix in {".onnx", ".pth"}:
            prod_model = os.path.join(eval_dir, f"current_production{prod_suffix}")
            with open(prod_model, "wb") as fh:
                fh.write(container.get_blob_client(prod_blob).download_blob().readall())
            current_wf_metrics = _eval(prod_model, wf_val_label_path, "current_widerface_val")
            if has_fc:
                current_fc_metrics = _eval(prod_model, fc_val_label_path, "current_fc_val")
        else:
            Task.current_task().get_logger().report_text(
                f"Production model pointer {current_production_pointer_blob} resolves to {prod_blob!r}. "
                "Skipping production metric evaluation because only .onnx and .pth models are supported."
            )
    except Exception as exc:
        if exc.__class__.__name__ not in {"ResourceNotFoundError", "BlobNotFoundError"}:
            raise
        Task.current_task().get_logger().report_text(
            f"No production model pointer found at {current_production_pointer_blob}; "
            "new model wins by default."
        )

    is_candidate = new_wf_metrics["ap_50"] > current_wf_metrics["ap_50"]

    logger = Task.current_task().get_logger()
    for split, m in [
        ("new_wf",      new_wf_metrics),
        ("new_fc",      new_fc_metrics),
        ("current_wf",  current_wf_metrics),
        ("current_fc",  current_fc_metrics),
    ]:
        for metric, value in m.items():
            logger.report_single_value(f"{split}_{metric}", value)
    logger.report_single_value("is_candidate", int(is_candidate))

    if is_candidate:
        model = OutputModel(task=Task.current_task(), name=f"scrfd-{version}")
        model.update_weights(weights_filename=onnx_path)
        model.tags = sorted(set((model.tags or []) + ["candidate"]))
        logger.report_text(
            f"scrfd-{version} tagged as candidate "
            f"(wf_ap_50 {new_wf_metrics['ap_50']:.4f} > {current_wf_metrics['ap_50']:.4f})"
        )
    else:
        logger.report_text(
            f"scrfd-{version} did not beat production "
            f"(wf_ap_50 {new_wf_metrics['ap_50']:.4f} <= {current_wf_metrics['ap_50']:.4f})"
        )

    return new_wf_metrics["ap_50"], new_wf_metrics["recall_50"], is_candidate, onnx_path


# ── Step 5: upload_model ───────────────────────────────────────────────────────

def upload_model(onnx_path: str, version: str) -> str:
    """
    Uploads ONNX to Azure models/scrfd/det_10g_<version>.onnx.
    Promotion (updating current_production.txt) is intentionally manual.
    """
    import os
    from clearml import Task
    from dotenv import load_dotenv

    scrfd_models_blob_dir           = "models/scrfd"
    current_production_pointer_blob = "models/scrfd/current_production.txt"

    load_dotenv()
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient, ContentSettings
    svc       = BlobServiceClient(account_url=os.environ["AZURE_BLOB_URL"], credential=DefaultAzureCredential())
    container = svc.get_container_client(os.environ["AZURE_CONTAINER_NAME"])

    blob_name = f"{scrfd_models_blob_dir}/det_10g_{version}.onnx"
    with open(onnx_path, "rb") as fh:
        container.get_blob_client(blob_name).upload_blob(
            fh, overwrite=False,
            content_settings=ContentSettings(content_type="application/octet-stream"),
        )

    Task.current_task().get_logger().report_text(
        f"Uploaded → azure://{blob_name}\n"
        f"To promote: update {current_production_pointer_blob} to '{blob_name}'"
    )
    return blob_name


# ── Pipeline controller ────────────────────────────────────────────────────────

class SCRFDRetrainingPipeline:
    def __init__(self, version: str, insightface_ref: str = INSIGHTFACE_REF):
        self.version         = version
        self.insightface_ref = insightface_ref

        # Compute absolute paths here (main process, __file__ is valid).
        # Pass them explicitly to steps that need them.
        _here         = os.path.dirname(os.path.abspath(__file__))
        repo_root     = os.path.dirname(_here)
        self.data_dir = os.path.join(repo_root, "retraining_data", version)
        _dataset      = os.path.join(_here, "dataset")
        self.scrfd_label_train = os.path.join(_dataset, "scrfd_label", "train", "labelv2.txt")
        self.scrfd_label_val   = os.path.join(_dataset, "scrfd_label", "val",   "labelv2.txt")
        self.wider_train_dir   = os.path.join(_dataset, "WIDER_train", "images")
        self.wider_val_dir     = os.path.join(_dataset, "WIDER_val",   "images")
        self.insightface_dir   = os.path.join(repo_root, "insightface_repo")

    def _build(self) -> PipelineController:
        pipe = PipelineController(
            name=f"scrfd-retrain-{self.version}",
            project="anonifyme-scrfd-retrain",
            version="1.0",
            add_pipeline_tags=True,
        )

        pipe.add_function_step(
            name="fetch_training_data",
            function=fetch_training_data,
            function_kwargs={"output_dir": self.data_dir},
            function_return=["failure_cases_dir", "n_failure_cases", "pretrained_path"],
        )

        # prepare_dataset builds the combined train labels used by HPO and final training.
        pipe.add_function_step(
            name="prepare_dataset",
            function=prepare_dataset,
            function_kwargs={
                "failure_cases_dir": "${fetch_training_data.failure_cases_dir}",
                "output_dir":        self.data_dir,
                "scrfd_label_train": self.scrfd_label_train,
                "scrfd_label_val":   self.scrfd_label_val,
            },
            function_return=["train_label_path", "combined_val_label_path", "fc_val_label_path", "wf_val_label_path", "dataset_stats"],
            parents=["fetch_training_data"],
        )

        pipe.add_function_step(
            name="hpo_search",
            function=hpo_search,
            function_kwargs={
                "pretrained_path":   "${fetch_training_data.pretrained_path}",
                "train_label_path":  "${prepare_dataset.train_label_path}",
                "val_label_path":    "${prepare_dataset.combined_val_label_path}",
                "output_dir":        self.data_dir,
                "wider_train_dir":   self.wider_train_dir,
                "wider_val_dir":     self.wider_val_dir,
                "insightface_dir":   self.insightface_dir,
                "insightface_ref":   self.insightface_ref,
                "hpo_max_jobs":      5,
                "hpo_trial_epochs":  3,
            },
            function_return=["best_lr", "best_weight_decay", "best_warmup_iters"],
            parents=["prepare_dataset"],
        )

        pipe.add_function_step(
            name="upload_dataset_version",
            function=upload_dataset_version,
            function_kwargs={
                "train_label_path":  "${prepare_dataset.train_label_path}",
                "combined_val_label_path": "${prepare_dataset.combined_val_label_path}",
                "fc_val_label_path": "${prepare_dataset.fc_val_label_path}",
                "dataset_stats":     "${prepare_dataset.dataset_stats}",
                "version":           self.version,
                "insightface_ref":   self.insightface_ref,
            },
            function_return=["dataset_blob_dir"],
            parents=["prepare_dataset"],
        )

        pipe.add_function_step(
            name="train_scrfd",
            function=train_scrfd,
            function_kwargs={
                "train_label_path": "${prepare_dataset.train_label_path}",
                "val_label_path":   "${prepare_dataset.combined_val_label_path}",
                "pretrained_path":  "${fetch_training_data.pretrained_path}",
                "output_dir":       self.data_dir,
                "lr":               "${hpo_search.best_lr}",
                "weight_decay":     "${hpo_search.best_weight_decay}",
                "warmup_iters":     "${hpo_search.best_warmup_iters}",
                "wider_train_dir":  self.wider_train_dir,
                "wider_val_dir":    self.wider_val_dir,
                "insightface_dir":  self.insightface_dir,
                "insightface_ref":  self.insightface_ref,
                "max_epochs":       30,
            },
            function_return=["checkpoint_path"],
            parents=["prepare_dataset", "hpo_search", "upload_dataset_version"],
        )

        pipe.add_function_step(
            name="evaluate_and_tag",
            function=evaluate_and_tag,
            function_kwargs={
                "checkpoint_path":   "${train_scrfd.checkpoint_path}",
                "wf_val_label_path": "${prepare_dataset.wf_val_label_path}",
                "fc_val_label_path": "${prepare_dataset.fc_val_label_path}",
                "output_dir":        self.data_dir,
                "version":           self.version,
                "wider_val_dir":     self.wider_val_dir,
                "insightface_dir":   self.insightface_dir,
                "insightface_ref":   self.insightface_ref,
            },
            function_return=["ap_50", "recall_50", "is_candidate", "onnx_path"],
            parents=["train_scrfd"],
        )

        pipe.add_function_step(
            name="upload_model",
            function=upload_model,
            function_kwargs={
                "onnx_path": "${evaluate_and_tag.onnx_path}",
                "version":   self.version,
            },
            function_return=["azure_blob_path"],
            parents=["evaluate_and_tag"],
        )

        return pipe

    def run(self):
        try:
            pipe = self._build()
            # Runs all steps locally on this machine (requires GPU + mmdet installed).
            # To dispatch to a remote ClearML agent instead, use pipe.start() and set
            # execution_queue on each step.
            pipe.start_locally(run_pipeline_steps_locally=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Removes per-run local retraining artifacts after success or failure."""
        import shutil
        from pathlib import Path

        for path in [
            self.data_dir,
            "./scrfd_work_dir",
            "./eval_results",
            "./eval_results_hpo",
            f"./det_10g_{self.version}.onnx",
            "./current_production.onnx",
        ]:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)

        for pattern in ["./hpo_trial_*", "./hpo_trial_*.onnx"]:
            for p in Path(".").glob(pattern):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink(missing_ok=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def make_training_version() -> str:
    """Builds a sortable version suffix for retraining artifacts."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def trigger_retraining(insightface_ref: str = INSIGHTFACE_REF):
    """
    Launch the full retraining pipeline.

    Args:
        insightface_ref: InsightFace branch, tag, or commit to checkout.
            Use a commit hash for reproducible training.
    """
    version = make_training_version()
    SCRFDRetrainingPipeline(version=version, insightface_ref=insightface_ref).run()
