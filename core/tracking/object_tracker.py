from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:

    def __init__(self):

        print("🧠 Initializing tracker...")

        self.tracker = DeepSort(
            max_age=30
        )

        print("✅ Tracker initialized")

    def update(self, detections, frame):

        tracker_inputs = []

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            width = x2 - x1
            height = y2 - y1

            confidence = det["confidence"]

            class_name = det["class"]

            tracker_inputs.append(
                (
                    [x1, y1, width, height],
                    confidence,
                    class_name
                )
            )

        tracks = self.tracker.update_tracks(
            tracker_inputs,
            frame=frame
        )

        tracked_objects = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)

            class_name = track.get_det_class()

            tracked_objects.append({
                "id": track_id,
                "class": class_name,
                "bbox": [x1, y1, x2, y2]
            })

        return tracked_objects