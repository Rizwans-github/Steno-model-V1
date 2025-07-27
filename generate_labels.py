import os
import csv

input_dir = "augmented"
output_file = "labels/labels.csv"
os.makedirs("labels", exist_ok=True)

rows = []

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        name = filename.split("_")[0]  # e.g., pee_0 → pee
        rows.append([filename, name])

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "text"])
    writer.writerows(rows)

print(f"✅ labels.csv created with {len(rows)} entries at {output_file}")

