
# SOFTWARE & SETUP:
# pip install opencv-python torch torchvision ultralytics

import cv2
from ultralytics import YOLO
import argparse
import os

def run_hazard_detector(model_path, video_path):
    """
    Initializes AI models and video file, then runs the main detection loop.
    Args:
        model_path (str): Path to the trained YOLO model file (.pt).
        video_path (str): Path to the video file to process.
    """
    # Validate Input Paths
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Load AI Models
    try:
        # Load the YOLO model for pothole detection
        pothole_model = YOLO(model_path)
        print("Pothole detection model loaded successfully.")

        # CONCEPTUAL: Load a segmentation model for road ends
        # road_end_model = YOLO('path/to/road_segmentation_model.pt')
        # print("Road end segmentation model loaded.")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Video File Reader ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'.")
        return

    frame_count = 0
    print("Video file loaded. Starting detection... (Press 'q' to quit)")

    # --- 4. Main Detection Loop ---
    while True:
        success, frame = cap.read()
        if not success:
            print("End of video file reached.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Perform Pothole Detection
        pothole_results = pothole_model(frame)
        annotated_frame = pothole_results[0].plot()

        #Perform Road End Detection (Conceptual)
        # As before, this section is a placeholder for a segmentation model.
        # It would identify the drivable area.
        # road_end_results = road_end_model(frame)
        # (logic to overlay segmentation mask)

        #Display the Output
        cv2.imshow("Road Hazard Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #Set up Command-Line Argument Parser 
    parser = argparse.ArgumentParser(description="Detect potholes and other road hazards in a video file.")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO model file (e.g., best.pt).")
    parser.add_argument("--video", required=True, help="Path to the video file to be processed.")
    
    args = parser.parse_args()
    
    run_hazard_detector(args.model, args.video)
