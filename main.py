import sys
import logging
from PyQt6.QtWidgets import QApplication

# Імпорт усіх незалежних модулів системи
from videocapture import VideoCaptureModule
from tracker import HandTracker
from classifier import GestureClassifier
from gui import GestureRecognitionGUI


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Запуск системи розпізнавання жестів...")

    app = QApplication(sys.argv)

    camera = VideoCaptureModule(max_cameras_to_check=5)
    tracker = HandTracker(model_path="hand_landmarker.task", max_num_hands=1)
    classifier = GestureClassifier(model_path="static_model.pkl", confidence_threshold=0.6)

    window = GestureRecognitionGUI(
        camera=camera,
        tracker=tracker,
        classifier=classifier
    )
    window.show()

    exit_code = app.exec()
    logging.info("Роботу системи завершено.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()