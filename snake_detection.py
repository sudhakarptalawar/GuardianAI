from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("videos/snake/snake.mp4")

snake_found = False

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])
        obj = result.names[cls]

        print(obj)

        if obj.lower() == "snake":
            snake_found = True

print("\nSNAKE REPORT")
print("=" * 30)

if snake_found:
    print("[CRITICAL] SNAKE DETECTED")
else:
    print("No snake detected")