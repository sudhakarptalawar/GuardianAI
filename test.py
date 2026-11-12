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

        if obj == "person":
            events.add(
                f"{timestamp} - Person Detected"
            )

        elif obj == "dog":
            events.add(
                f"{timestamp} - Dog Detected"
            )

        elif obj == "car":
            events.add(
                f"{timestamp} - Vehicle Detected"
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

    file.write("GUARDIANAI REPORT\n")
    file.write("=" * 30 + "\n")

    for event in events:
        file.write(event + "\n")

    file.write("\nSUMMARY\n")
    file.write("=" * 30 + "\n")

    file.write(f"Total Events: {len(events)}")

print("\nReport saved as report.txt")