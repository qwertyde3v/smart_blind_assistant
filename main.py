import cv2

from core.camera.ip_camera import IPCamera
from core.detection.yolo_detector import YOLODetector
from core.tracking.object_tracker import ObjectTracker


def main():

    camera = IPCamera()

    detector = YOLODetector()

    tracker = ObjectTracker()

    frame_count = 0

    last_tracked_objects = []

    try:

        camera.connect()

        while True:

            frame = camera.get_frame()

            if frame is None:

                print("⚠ Failed to get frame")
                continue

            frame_count += 1

            # Run YOLO every 120th frame
            if frame_count % 120 == 0:

                detections = detector.detect(frame)

                last_tracked_objects = tracker.update(
                    detections,
                    frame
                )

            # Draw tracked objects
            for obj in last_tracked_objects:

                x1, y1, x2, y2 = obj["bbox"]

                label = (
                    f"{obj['class']}_{obj['id']}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "Smart Blind Assistant - Tracking",
                frame
            )

            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    except Exception as e:

        print(f"❌ Error: {e}")

    finally:

        camera.release()


if __name__ == "__main__":
    main()