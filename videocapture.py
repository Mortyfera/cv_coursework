import cv2
import logging

class VideoCaptureModule:
    def __init__(self, max_cameras_to_check=5):
        self.cap = None
        self.is_running = False
        
        for index in range(max_cameras_to_check):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.is_running = True
                    break
            cap.release()

        if not self.is_running:
            logging.error("Помилка: Не знайдено жодної підключеної та робочої камери.")
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        if not self.is_running or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logging.warning("Увага: Втрачено зв'язок з відеопотоком.")
            return False, None
            
        return True, frame

    def release(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.is_running = False