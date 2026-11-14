from ultralytics import YOLO

model = YOLO("models/fire.pt")

results = model("videos/fire/fire.mp4")

events = set()

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])

        obj = result.names[cls]

        print(obj)

        if obj.lower() == "fire":

            events.add(
                "[CRITICAL] FIRE DETECTED"
            )

        elif obj.lower() == "smoke":

            events.add(
                "[HIGH] SMOKE DETECTED"
            )


print("\nFIRE REPORT")
print("=" * 30)

if events:

    for event in events:
        print(event)

else:

    print("No fire or smoke detected")