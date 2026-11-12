from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("images/test.jpg")

events = set()

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])

        obj = result.names[cls]

        if obj == "person":
            events.add(
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Person Detected"
)


        elif obj == "dog":
            events.add(
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Dog Detected"
)

        elif obj == "car":
            events.add(
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Vehicle Detected"
)

print("\nGUARDIANAI REPORT\n")
print("=" * 30)

for event in events:
    print(event)

print("\nSUMMARY")
print("=" * 30)
print(f"Total Events: {len(events)}")
with open("report.txt", "w") as file:

    file.write("GUARDIANAI REPORT\n")
    file.write("=" * 30 + "\n")

    for event in events:
        file.write(event + "\n")

    file.write("\nSUMMARY\n")
    file.write("=" * 30 + "\n")

    file.write(f"Total Events: {len(events)}")

print("\nReport saved as report.txt")