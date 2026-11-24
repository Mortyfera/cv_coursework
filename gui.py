import cv2
from PyQt6.QtWidgets import (QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QMessageBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


class GestureRecognitionGUI(QMainWindow):
    def __init__(self, camera, tracker, classifier):
        super().__init__()
        self.setWindowTitle("Система розпізнавання жестів")
        self.setGeometry(100, 100, 1000, 750)

        self.current_gesture = None
        self.gesture_counter = 0
        self.CONFIRMATION_FRAMES = 10

        self.camera = camera
        self.tracker = tracker
        self.classifier = classifier

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        if not self.camera.is_running:
            self.show_error("Камеру не знайдено або вона недоступна.")
            self.status_label.setText("Камера: Помилка ініціалізації")
        elif self.classifier.model is None:
            self.show_error("Файл моделі 'model.pkl' не знайдено. Спочатку запустіть train_model.py.")
            self.status_label.setText("Камера: Готова (модель відсутня)")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.video_label = QLabel("Відеопотік відсутній")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        self.video_label.setMinimumSize(800, 550)
        layout.addWidget(self.video_label, stretch=1)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Камера: Готово до запуску")
        self.hand_status_label = QLabel("Рука в кадрі: Ні")
        self.current_symbol_label = QLabel("Поточний жест: —")

        for lbl in (self.status_label, self.hand_status_label, self.current_symbol_label):
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
            status_layout.addWidget(lbl)

        layout.addLayout(status_layout)

        self.text_buffer = QTextEdit()
        self.text_buffer.setReadOnly(True)
        self.text_buffer.setFixedHeight(60)
        self.text_buffer.setStyleSheet("font-size: 36px; font-weight: bold;")
        layout.addWidget(self.text_buffer)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Запустити")
        self.btn_stop = QPushButton("Зупинити")
        self.btn_stop.setEnabled(False)
        self.btn_clear = QPushButton("Очистити текст")

        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop.clicked.connect(self.stop_capture)
        self.btn_clear.clicked.connect(self.text_buffer.clear)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

    def start_capture(self):
        if self.camera.is_running:
            self.timer.start(30)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_label.setText("Камера: Розпізнавання...")

    def stop_capture(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.setText("Відеопотік зупинено")
        self.status_label.setText("Камера: Зупинено")
        self.hand_status_label.setText("Рука в кадрі: Ні")
        self.current_symbol_label.setText("Поточний жест: —")
        self.current_gesture = None
        self.gesture_counter = 0

    def process_frame(self):
        success, frame = self.camera.get_frame()

        if not success:
            self.stop_capture()
            self.show_error("Увага: Втрачено зв'язок з відеопотоком.")
            self.status_label.setText("Камера: Помилка відеопотоку")
            return

        frame = cv2.flip(frame, 1)
        landmarks, annotated_frame = self.tracker.process_frame(frame)

        recognized_symbol = None
        if landmarks:
            self.hand_status_label.setText("Рука в кадрі: Так")
            recognized_symbol = self.classifier.predict(landmarks)
            self.current_symbol_label.setText(
                f"Поточний жест: {recognized_symbol if recognized_symbol else '—'}"
            )
        else:
            self.hand_status_label.setText("Рука в кадрі: Ні")
            self.current_symbol_label.setText("Поточний жест: —")

        self.process_text_logic(recognized_symbol)
        self.display_image(annotated_frame)

    def process_text_logic(self, symbol):
        if symbol:
            if symbol == self.current_gesture:
                self.gesture_counter += 1
                if self.gesture_counter == self.CONFIRMATION_FRAMES:
                    current_text = self.text_buffer.toPlainText()
                    self.text_buffer.setText(current_text + symbol)
            else:
                self.current_gesture = symbol
                self.gesture_counter = 1
        else:
            self.current_gesture = None
            self.gesture_counter = 0

    def display_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    def show_error(self, message):
        QMessageBox.critical(self, "Помилка", message)

    def closeEvent(self, event):
        self.camera.release()
        event.accept()