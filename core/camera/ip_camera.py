import cv2
from config import IP_CAMERA_URL


class IPCamera:

    def __init__(self):

        self.url = IP_CAMERA_URL
        self.cap = None

    def connect(self):

        self.cap = cv2.VideoCapture(self.url)

        if not self.cap.isOpened():
            raise Exception("Failed to connect to IP Camera")

        print("✅ IP Camera Connected")

    def get_frame(self):

        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()