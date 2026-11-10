from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("videos/test.mp4")

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

print("\nEVENTS FOUND:\n")

for event in events:
    print(event)