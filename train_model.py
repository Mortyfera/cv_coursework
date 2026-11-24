import csv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(dataset_path="data/dataset.csv", model_output_path="model.pkl"):
    features = []
    labels = []

    print(f"Завантаження даних із {dataset_path}...")
    try:
        with open(dataset_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 64:
                    continue
                labels.append(row[0])
                features.append([float(val) for val in row[1:64]])
    except FileNotFoundError:
        print(f"Помилка: Файл {dataset_path} не знайдено! Спочатку зберіть дані.")
        return

    X = np.array(features)
    y = np.array(labels)

    print(f"Завантажено зразків: {len(X)}. Унікальних класів (літер): {len(set(y))} -> {sorted(list(set(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Навчання моделі Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*45)
    print(f"ТОЧНІСТЬ НА КОНТРОЛЬНІЙ ВИБІРЦІ: {accuracy * 100:.2f}%")
    print("="*45)
    print("\nДетальний звіт по кожній літері:")
    print(classification_report(y_test, y_pred))

    with open(model_output_path, mode='wb') as f:
        pickle.dump(model, f)
    print(f"Навчену модель успішно збережено у файл: {model_output_path}")

if __name__ == "__main__":
    train_and_evaluate(dataset_path="data/static_dataset.csv", model_output_path="static_model.pkl")