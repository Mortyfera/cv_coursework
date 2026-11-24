import pickle
import numpy as np
import logging
from tracker import normalize_landmarks

class GestureClassifier:
    def __init__(self, model_path="static_model.pkl", confidence_threshold=0.6):
        self.model = None
        self.confidence_threshold = confidence_threshold
        
        try:
            with open(model_path, mode='rb') as f:
                self.model = pickle.load(f)
            logging.info(f"Модель успішно завантажена з {model_path}")
        except FileNotFoundError:
            logging.error(f"Файл моделі {model_path} не знайдено. Спочатку запустіть навчання.")
        except Exception as e:
            logging.error(f"Помилка при завантаженні моделі: {e}")

    def predict(self, landmarks_list):
        if self.model is None or not landmarks_list:
            return None

        feature_vector = normalize_landmarks(landmarks_list)
        if not feature_vector:
            return None

        input_data = np.array([feature_vector])

        probabilities = self.model.predict_proba(input_data)[0]
        max_prob = np.max(probabilities)

        if max_prob < self.confidence_threshold:
            return None

        predicted_class_index = np.argmax(probabilities)
        predicted_label = self.model.classes_[predicted_class_index]
        
        return str(predicted_label)