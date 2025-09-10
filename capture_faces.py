import cv2
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Capture face images for dataset")
    parser.add_argument("--name", required=True, help="Person name (dataset/<name>)")
    parser.add_argument("--count", type=int, default=100, help="Number of images to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (0 default)")
    args = parser.parse_args()

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Save folder
    save_dir = Path("dataset") / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open webcam. Try --camera 1 or --camera 2")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            if count < args.count:
                filename = save_dir / f"{args.name}_{count:04d}.jpg"
                cv2.imwrite(str(filename), face)
                count += 1

        cv2.putText(frame, f"Captured: {count}/{args.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture Faces (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= args.count:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} images to {save_dir}")

if __name__ == "__main__":
    main()
