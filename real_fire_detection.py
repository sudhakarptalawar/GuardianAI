from ultralytics import YOLO

# Fire model (will add later)
model = YOLO("models/fire.pt")

results = model("videos/fire/fire.mp4")

fire_found = False

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])
        obj = result.names[cls]

        print(obj)

        if obj.lower() in ["fire", "smoke"]:

            fire_found = True

print("\nFIRE REPORT")
print("=" * 30)

if fire_found:

    print("[CRITICAL] FIRE DETECTED")

else:

    print("No fire detected")
    