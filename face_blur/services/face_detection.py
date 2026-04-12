from deep_sort_realtime.deepsort_tracker import DeepSort


class Detection:
    def __init__(self, face_det_app, fps, min_conf_threshold=0.3, device_id=0):
        self.fps = fps
        self.tracker = DeepSort(
                            max_age=fps // 2,
                            n_init=fps // 2,
                            max_iou_distance=0.9,
                            max_cosine_distance=0.1,
                            embedder='mobilenet',
                            embedder_gpu=True
                        )
        self.face_det_app = face_det_app
        self.min_conf_threshold = min_conf_threshold

    def get_detections(self, frame):
        faces = self.face_det_app.get(frame)

        detections = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            conf = float(face.det_score)

            # set detections to match DeepSORT format
            detections.append(([x1, y1, w, h], conf, 'face', face.kps))
        return detections

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return frame

        detections = self.get_detections(frame)

        others = []
        if detections:
            for det in detections:
                x, y, w, h = det[0]
                # append bbox, confidence, and kps to supplementary info
                others.append([[x, y, x + w, y + h], det[1], det[3]])

        tracks = self.tracker.update_tracks(detections, frame=frame, others=others)

        detections = []
        for track in tracks:
            supplementary = track.get_det_supplementary()

            if track.is_confirmed() and track.time_since_update == 0:
                x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]

            else:
                if supplementary is None or supplementary[1] < self.min_conf_threshold:
                    continue

                # this is the same as the original bounding box
                original_bbox = supplementary[0]
                x1, y1, x2, y2 = [int(v) for v in original_bbox]

            kps = supplementary[2]
            detections.append({"bbox":[x1, y1, x2, y2], "kps": kps})

        return detections


