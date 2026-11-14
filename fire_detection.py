from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("videos/fire/fire.mp4")

fire_keywords = [
    "fire",
    "smoke"
]

fire_found = False

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])
        obj = result.names[cls]

        print(obj)

        if obj.lower() in fire_keywords:
            fire_found = True

print("\nFIRE REPORT")
print("=" * 30)

if fire_found:
    print("[CRITICAL] FIRE DETECTED")
else:
    print("No fire detected")