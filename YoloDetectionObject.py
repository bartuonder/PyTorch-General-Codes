"""
YOLO11 Hand Gesture Detection Tutorial
======================================
This educational project demonstrates how to use YOLO11 (latest YOLO version) models
from Hugging Face/Ultralytics for real-time hand gesture and number recognition using your webcam.

YOLO11 is the newest version with improved accuracy and speed compared to YOLOv8.

Author: Educational Tutorial
License: MIT
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


class HandGestureDetector:
    """
    A class to detect hand gestures and numbers using YOLO model.
    """

    def __init__(self, model_name="yolo11n.pt", confidence_threshold=0.5):
        """
        Initialize the hand gesture detector.

        Args:
            model_name (str): Name of the YOLO model to use (YOLO11 recommended)
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load the YOLO11 model from Ultralytics or local cache.
        """
        print(f"Loading YOLO11 model: {self.model_name}")
        print("This may take a moment on first run as the model downloads...")

        try:
            # YOLO11 will automatically download from Ultralytics if not present
            # YOLO11 offers better accuracy and speed than YOLOv8
            self.model = YOLO(self.model_name)
            print(f"✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def detect_hands(self, frame):
        """
        Detect hands and gestures in a frame.

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            frame: Annotated frame with detections
            results: YOLO detection results
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        # Get the annotated frame
        annotated_frame = results[0].plot()

        return annotated_frame, results

    def process_detections(self, results):
        """
        Process and extract detection information.

        Args:
            results: YOLO detection results

        Returns:
            list: List of detection dictionaries
        """
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

        return detections

    def run_webcam(self):
        """
        Run real-time hand gesture detection using webcam.
        """
        print("\n" + "=" * 60)
        print("Starting Webcam Hand Gesture Detection")
        print("=" * 60)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press '+' to increase confidence threshold")
        print("  - Press '-' to decrease confidence threshold")
        print("\n" + "=" * 60 + "\n")

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        fps_start_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("✗ Error: Could not read frame")
                break

            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:
                fps_end_time = time.time()
                fps = frame_count / (fps_end_time - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()

            # Detect hands and gestures
            annotated_frame, results = self.detect_hands(frame)

            # Get detection details
            detections = self.process_detections(results)

            # Add information overlay
            info_text = [
                f"FPS: {fps:.1f}",
                f"Confidence: {self.confidence_threshold:.2f}",
                f"Detections: {len(detections)}",
                f"Model: {self.model_name}"
            ]

            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display detection details
            if detections:
                y_pos = 150
                cv2.putText(annotated_frame, "Detected Objects:", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                for det in detections[:5]:  # Show max 5 detections
                    y_pos += 20
                    det_text = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(annotated_frame, det_text, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show frame
            cv2.imshow('YOLO11 Hand Gesture Detection', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✓ Frame saved as {filename}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam closed successfully")


def main():
    """
    Main function to run the hand gesture detector.
    """
    print("\n" + "=" * 60)
    print("YOLO11 Hand Gesture Detection Tutorial")
    print("=" * 60)
    print("\nThis tutorial demonstrates how to:")
    print("  1. Load YOLO11 models (latest version!) from Ultralytics")
    print("  2. Use webcam for real-time object detection")
    print("  3. Detect hands, gestures, and numbers")
    print("\nWhy YOLO11?")
    print("  ✓ Better accuracy than YOLOv8")
    print("  ✓ Faster inference speed")
    print("  ✓ Improved small object detection")
    print("=" * 60 + "\n")

    # Get model name from environment or use default
    model_name = os.getenv('YOLO_MODEL', 'yolo11n.pt')
    confidence = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

    print(f"Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Confidence Threshold: {confidence}")
    print()

    # Available models information
    print("Available YOLO11 models (from smallest to largest):")
    print("  - yolo11n.pt (Nano - Fastest, ~2.6M params)")
    print("  - yolo11s.pt (Small - ~9.4M params)")
    print("  - yolo11m.pt (Medium - ~20.1M params)")
    print("  - yolo11l.pt (Large - ~25.3M params)")
    print("  - yolo11x.pt (Extra Large - ~56.9M params, most accurate)")
    print("\nYOLO11 vs YOLOv8:")
    print("  • YOLO11n is 22% faster than YOLOv8n with similar accuracy")
    print("  • Better performance on small objects (hands, fingers)")
    print("  • Improved feature extraction")
    print("\nNote: For specialized hand gesture detection:")
    print("  - Search Hugging Face for 'hand gesture' or 'sign language' models")
    print("  - Or train your own YOLO11 model on hand gesture datasets")
    print()

    try:
        # Initialize detector
        detector = HandGestureDetector(
            model_name=model_name,
            confidence_threshold=confidence
        )

        # Run webcam detection
        detector.run_webcam()

    except KeyboardInterrupt:
        print("\n\n✓ Program interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
