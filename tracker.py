import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task', max_num_hands=1):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.landmarker.detect(mp_image)
        
        landmarks_list = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for lm in hand_landmarks:
                    landmarks_list.append([lm.x, lm.y, lm.z])
                
                self._draw_landmarks(frame, hand_landmarks)
                
        return landmarks_list, frame
        
    def _draw_landmarks(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        points_px = []
        
        for lm in hand_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points_px.append((px, py))
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        
        for connection in self.HAND_CONNECTIONS:
            p1 = points_px[connection[0]]
            p2 = points_px[connection[1]]
            cv2.line(frame, p1, p2, (0, 255, 0), 2)


def normalize_landmarks(landmarks_list):
    if not landmarks_list or len(landmarks_list) != 21:
        return None
    
    base_x, base_y, base_z = landmarks_list[0]
    normalized_list = []
    
    for point in landmarks_list:
        norm_x = point[0] - base_x
        norm_y = point[1] - base_y
        norm_z = point[2] - base_z
        normalized_list.append([norm_x, norm_y, norm_z])
        
    normalized_array = np.array(normalized_list)
    
    max_value = np.max(np.abs(normalized_array))
    if max_value > 0:
        normalized_array = normalized_array / max_value
        
    feature_vector = normalized_array.flatten().tolist()
    
    return feature_vector