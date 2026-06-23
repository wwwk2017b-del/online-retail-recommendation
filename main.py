"""
main.py
Run this to execute the full Online Retail Recommendation System pipeline.
"""
print("=" * 62)
print(" ONLINE RETAIL RECOMMENDATION SYSTEM")
print(" CodTech IT Solutions — ML Internship")
print(" Intern   : Abhishek Prasad")
print(" Intern ID: CITS5099")
print("=" * 62)

import subprocess, sys

steps = [
    ("Step 1: Generating retail dataset...",      "generate_data.py"),
    ("Step 2: Creating visualizations...",         "visualize.py"),
    ("Step 3: Building recommendation engines...", "model.py"),
]

for msg, script in steps:
    print(f"\n{'─'*62}")
    print(f" {msg}")
    print(f"{'─'*62}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Stopping.")
        sys.exit(1)

print("\n" + "=" * 62)
print(" ✅ ALL DONE! Check the outputs/ folder for images.")
print("=" * 62)
