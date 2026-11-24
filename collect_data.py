import cv2
import csv
from tracker import HandTracker, normalize_landmarks

def collect_dataset(label, samples_to_collect=500, filename="data/static_dataset.csv"):
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    collected = 0
    is_recording = False
    window_name = "Data Collection Window"
    
    print(f"Готово до запису літери '{label}'. Клікніть на вікно з відео та натисніть 'S' для старту.")
    
    while cap.isOpened() and collected < samples_to_collect:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        landmarks, drawn_frame = tracker.process_frame(frame)
        
        if not is_recording:
            cv2.putText(drawn_frame, "Press 'S' to START recording", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(drawn_frame, f"RECORDING: {collected}/{samples_to_collect}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if landmarks:
                feature_vector = normalize_landmarks(landmarks)
                if feature_vector:
                    with open(filename, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + feature_vector)
                    collected += 1
        
        cv2.imshow(window_name, drawn_frame)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key in [ord('s'), ord('S'), ord('і'), ord('І')]:
            is_recording = True
            print("Запис розпочато!")
        elif key in [ord('q'), ord('Q'), ord('й'), ord('Й'), 27]:
            print("Вихід з програми.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Збір даних для літери '{label}' завершено. Всього зібрано: {collected} зразків.")

if __name__ == "__main__":
    collect_dataset(label="Я", samples_to_collect=500)