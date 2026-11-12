from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("videos/test.mp4")

events = []
detected_objects = set()

fps = 30
frame_no = 0

for result in results:

    frame_no += 1

    current_time = frame_no / fps

    for box in result.boxes:

        cls = int(box.cls[0])
        obj = result.names[cls]

        if obj not in detected_objects:

            detected_objects.add(obj)

            if obj == "person":
                events.append(
                    f"{current_time:.1f}s : Person Appeared"
                )

            elif obj == "dog":
                events.append(
                    f"{current_time:.1f}s : Dog Appeared"
                )

            elif obj == "car":
                events.append(
                    f"{current_time:.1f}s : Vehicle Appeared"
                )

print("\nGUARDIAN AI TIMELINE\n")

for event in events:
    print(event)