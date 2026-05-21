from ultralytics import YOLO


class YOLODetector:

    def __init__(self):

        print("🧠 Loading YOLO model...")

        self.model = YOLO("yolov8n.pt")

        print("✅ YOLO loaded successfully")

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections