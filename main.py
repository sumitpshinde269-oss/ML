"""
YOLO11 + OpenCV Object Detection Runner
ML Project Entrypoint
"""

import sys
import os
from pathlib import Path
import argparse

# Ensure proper path resolution so 'ultralytics' imports the package correctly
WORKSPACE_DIR = Path(__file__).resolve().parent
ULTRALYTICS_DIR = WORKSPACE_DIR / "ultralytics"
if str(ULTRALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_DIR))

import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 Object Detection with OpenCV")
    parser.add_argument(
        "--source",
        type=str,
        default=str(ULTRALYTICS_DIR / "bus.jpg"),
        help="Path to image, video, directory, or '0' for webcam (default: bus.jpg)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(ULTRALYTICS_DIR / "yolo11n.pt"),
        help="Path to YOLO model weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_DIR / "results"),
        help="Directory to save output annotated images"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run real-time detection on webcam"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        # Fallback to checking inside ULTRALYTICS_DIR
        if (ULTRALYTICS_DIR / args.model).exists():
            model_path = ULTRALYTICS_DIR / args.model
        else:
            print(f"Error: Model not found at {model_path}")
            return 1

    print("=" * 60)
    print(f"Loading YOLO Model: {model_path.name}")
    print("=" * 60)
    model = YOLO(str(model_path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Webcam Mode
    if args.webcam or args.source == "0":
        print("Starting Webcam stream (press 'q' to quit)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam.")
            return 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=args.conf, verbose=False)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO11 Real-Time Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stream stopped.")
        return 0

    # Image / File Mode
    source_path = Path(args.source)
    if not source_path.exists():
        if (ULTRALYTICS_DIR / args.source).exists():
            source_path = ULTRALYTICS_DIR / args.source
        else:
            print(f"Error: Source file or directory not found: {source_path}")
            return 1

    print(f"Running inference on: {source_path}")
    results = model.predict(source=str(source_path), conf=args.conf, save=False)

    for i, result in enumerate(results):
        orig_name = Path(result.path).name if hasattr(result, 'path') and result.path else f"result_{i+1}.jpg"
        save_path = output_dir / f"detected_{orig_name}"

        # Get annotated image
        annotated_img = result.plot()
        cv2.imwrite(str(save_path), annotated_img)

        # Print detection details
        boxes = result.boxes
        num_objects = len(boxes)
        print("\n" + "-" * 50)
        print(f"File: {orig_name}")
        print(f"Total Detections: {num_objects}")
        print(f"Saved Annotated Result To: {save_path}")
        print("-" * 50)

        if num_objects > 0:
            print(f"{'Class':<15} | {'Confidence':<12} | {'Bounding Box (x1, y1, x2, y2)'}")
            print("-" * 50)
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                conf = float(box.conf[0].item())
                xyxy = [round(coord, 1) for coord in box.xyxy[0].tolist()]
                print(f"{cls_name:<15} | {conf:.2%}       | {xyxy}")
        else:
            print("No objects detected above confidence threshold.")

    print("\n" + "=" * 60)
    print(f"Inference complete! Results saved in: {output_dir}")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
