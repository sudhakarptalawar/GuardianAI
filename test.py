from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Analyze image
results = model("images/test.jpg")

# Store detected events
events = set()

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])
        obj = result.names[cls]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        current_hour = datetime.now().hour

current_hour = datetime.now().hour

if obj == "person":

    if current_hour >= 22 or current_hour <= 5:

        events.add(
            f"{timestamp} - [HIGH] Potential Intruder Detected"
        )

    else:

        events.add(
            f"{timestamp} - [MEDIUM] Person Detected"
        )

# Print Report
print("\nGUARDIANAI REPORT\n")
print("=" * 30)

for event in events:
    print(event)

print("\nSUMMARY")
print("=" * 30)
print(f"Total Events: {len(events)}")

# Save Report
with open("report.txt", "w") as file:
    with open("alerts.txt", "w") as file:

    for event in events:

        if "[HIGH]" in event:
            file.write(event + "\n")

    print("Critical alerts saved as alerts.txt")

    file.write("GUARDIANAI REPORT\n")
    file.write("=" * 30 + "\n")

    for event in events:
        file.write(event + "\n")

    file.write("\nSUMMARY\n")
    file.write("=" * 30 + "\n")

    file.write(f"Total Events: {len(events)}")

print("\nReport saved as report.txt")