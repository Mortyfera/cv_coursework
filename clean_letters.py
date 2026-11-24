import csv

letters_to_remove = {"К", "П"}
filename = "data/static_dataset.csv"

with open(filename, mode="r", encoding="utf-8") as f:
    rows = [row for row in csv.reader(f) if row and row[0] not in letters_to_remove]

with open(filename, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Видалено старі записи для: {letters_to_remove}. Залишилось рядків: {len(rows)}")