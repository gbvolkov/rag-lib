import os

output_dir = "results/tasks_output_v4_filtered"
files = os.listdir(output_dir)
files.sort()

print(f"Total files: {len(files)}")
for f in files:
    if "Анализ" in f or "Множества" in f or "Операции" in f or "Классическое" in f:
        print(f)
