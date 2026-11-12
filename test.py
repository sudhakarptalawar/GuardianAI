from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("images/test.jpg")

events = set()

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])

        obj = result.names[cls]

        if obj == "person":
            events.add("Person Detected")

        elif obj == "dog":
            events.add("Dog Detected")

        elif obj == "car":
            events.add("Vehicle Detected")

print("\nGUARDIANAI REPORT\n")
print("=" * 30)

for event in events:
    print(event)

print("\nSUMMARY")
print("=" * 30)
print(f"Total Events: {len(events)}")