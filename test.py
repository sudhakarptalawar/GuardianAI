from ultralytics import YOLO
from datetime import datetime
import os

os.makedirs("reports", exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

results = model("videos/fire/fire.mp4")


# Store detected events
events = set()
person_count = 0
vehicle_count = 0
animal_count = 0
fire_detected = False

for result in results:

    for box in result.boxes:

        cls = int(box.cls[0])
        obj = result.names[cls]

        print(obj)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_hour = datetime.now().hour

        if obj == "person":
            if person_count > 0:
                fire_detected = True
            person_count += 1

            if current_hour >= 22 or current_hour <= 5:
                events.add(
                            f"{timestamp} - [HIGH] Potential Intruder Detected"
                        )


        

            else:
                events.add(
                            f"{timestamp} - [MEDIUM] Person Detected"
                        )

        

        elif obj == "dog":
            animal_count += 1

            events.add(
                f"{timestamp} - [LOW] Dog Detected"
            )

        elif obj == "car":
            vehicle_count += 1

            events.add(
                f"{timestamp} - [MEDIUM] Vehicle Detected"
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
with open(
    f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    "w"
) as file:

    file.write("GUARDIANAI REPORT\n")
    file.write("=" * 30 + "\n")

    for event in events:
        file.write(event + "\n")

    file.write("\nSUMMARY\n")
    file.write("=" * 30 + "\n")

    file.write(f"Total Events: {len(events)}")
    print("\nINCIDENT SUMMARY")
    print("\nALERTS")
print("=" * 30)

if fire_detected:
    print("[CRITICAL] FIRE DETECTED")
    print("=" * 30)

    print(f"Person Detections: {person_count}")
    print(f"Vehicle Detections: {vehicle_count}")
    print(f"Animal Detections: {animal_count}")
print("\nReport saved as report.txt")