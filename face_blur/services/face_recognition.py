import os
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace
from insightface.utils.face_align import norm_crop


class FaceRecognizer:
    def __init__(
        self,
        face_det_app,
        in_dir,
        whitelist_threshold: float = 0.55,
    ):
        self.face_det_app = face_det_app
        self.in_dir = in_dir
        self.whitelist_threshold = whitelist_threshold
        self.whitelist_embeddings = self.build_whitelist(self.in_dir)
        
    def build_whitelist(self, folder: str = None) -> list[np.ndarray]:
        """
        Enroll whitelisted faces from either:
        - A list of image file paths
        - A folder path where all images inside will be enrolled.

        Each image should contain exactly one face.
        Multiple images per person are fine, because all embeddings are stored
        and matching uses max similarity across all of them.

        Returns a list of unit-normalised FaceNet embeddings.
        """
        image_paths = None
        if folder is not None:
            image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
            image_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(image_extensions)
            ]
            print(f"[whitelist] Found {len(image_paths)} images in '{folder}'")

        if not image_paths:
            print("[whitelist] WARNING: no images provided.")
            return []

        embeddings = []
        for path in image_paths:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"[whitelist] WARNING: could not read '{path}', skipping.")
                continue

            # Upscale the image if the image resolution is too small, so that the FaceNet
            # Can detect faces more reliably.
            h, w = img_bgr.shape[:2]
            if max(h, w) < 640:
                scale = 640 / max(h, w)
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

            faces = self.face_det_app.get(img_bgr)

            if not faces:
                # If no face is detected, then fall back to full image without alignment
                print(f"[whitelist] NOTE: no face detected in '{path}', using full image (no alignment).")
                crop = img_bgr
            else:
                face = max(faces, key=lambda f: f.det_score)

                # Use aligned crop if landmarks are available, else fall back to bbox crop
                aligned = self.get_aligned_face(img_bgr, face)
                if aligned is not None:
                    crop = aligned
                    print(f"[whitelist] Enrolled with alignment: {path}")
                else:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
                    crop = img_bgr[y1:y2, x1:x2]
                    print(f"[whitelist] Enrolled without alignment (no landmarks): {path}")

            aug_embs = self.get_embeddings_augmented(crop)
            if not aug_embs:
                print(f"[whitelist] WARNING: embedding failed for '{path}', skipping.")
                continue

            embeddings.extend(aug_embs)

        print(f"[whitelist] Total enrolled embeddings: {len(embeddings)}")
        return embeddings

    def is_whitelisted(
        self,
        face_bgr: np.ndarray,
    ) -> bool:
        """
        Return True if the face matches any whitelist embedding above threshold.
        """
        if not self.whitelist_embeddings:
            return False

        query_emb = self.get_embedding(face_bgr)
        if query_emb is None:
            return False

        # Stack all whitelist embeddings: shape (N, D)
        wl_matrix = np.stack(self.whitelist_embeddings, axis=0)

        # Cosine similarities — dot product because both sides are L2-normalised
        similarities = wl_matrix @ query_emb  # shape (N,)
        return float(similarities.max()) >= self.whitelist_threshold

    def get_embeddings_augmented(self, face_bgr: np.ndarray) -> list[np.ndarray]:
        """
        Generate embeddings for multiple augmented versions of a face crop.
        """
        if face_bgr is None or face_bgr.size == 0:
            return []

        h, w = face_bgr.shape[:2]
        augments = [
            face_bgr,
            cv2.flip(face_bgr, 1),
            cv2.convertScaleAbs(face_bgr, alpha=1.2, beta=10),
            cv2.convertScaleAbs(face_bgr, alpha=0.8, beta=-10),
            face_bgr[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)],
        ]

        embeddings: list[np.ndarray] = []
        for aug in augments:
            emb = self.get_embedding(aug)
            if emb is not None:
                embeddings.append(emb)

        return embeddings

    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a normalised FaceNet embedding from a BGR face crop.

        Uses DeepFace with detector_backend='skip' because face detection is already
        handled upstream by InsightFace (buffalo_l)

        Returns a unit-normalised numpy array, or None on failure.
        """

        if face_bgr is None or face_bgr.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            result = DeepFace.represent(
                img_path=rgb,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="skip",
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return emb / (np.linalg.norm(emb) + 1e-10)
        except Exception as e:
            print(f"[embedding] ERROR: {e}")
            return None

    def get_aligned_face(self, img_bgr: np.ndarray, face) -> np.ndarray | None:
        """
        Align a detected face using the 5 landmark points from InsightFace.
        norm_crop produces a 112x112 aligned face — the standard input for
        ArcFace/FaceNet recognition models.
        """
        if face.kps is None:
            return None
        return norm_crop(img_bgr, face.kps)