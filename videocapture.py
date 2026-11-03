"""
Логіка роботи модуля video_capture.py

Ініціалізація: Використання класу cv2.VideoCapture
з бібліотеки OpenCV для відкриття вебкамери користувача.

Зчитування кадрів: Безперервне отримання послідовності
відеокадрів у вигляді масивів даних.

Обробка помилок (FR-16): Перевірка доступності пристрою
та коректна реакція на переривання відеопотоку (наприклад,
якщо камеру випадково відключили під час роботи).

Повернення даних: Передача захоплених кадрів до головного циклу,
звідки вони спрямовуються до інших локальних компонентів:
MediaPipe (для аналізу наявності руки) та PyQt (для
відображення у графічному інтерфейсі).
"""

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
            import logging
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
        if self.cap.isOpened():
            self.cap.release()
        self.is_running = False