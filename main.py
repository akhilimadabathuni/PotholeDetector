# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Road Hazard Detector using Python, OpenCV, and AI Models (v2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# OVERVIEW:
# This script processes a video file to detect road hazards. It uses a
# YOLO object detection model to find potholes and is structured to allow
# for a second model (like a segmentation model) to identify road ends.
#
# ---
#
# HOW TO RUN:
# You run this script from your terminal and provide the paths to your
# model and video files as command-line arguments.
#
# Example:
# python your_script_name.py --model path/to/best.pt --video path/to/road.mp4
#
# ---
#
# HOW IT WORKS:
# 1. Argument Parsing: It takes the model and video file paths from the command line.
# 2. Video Input: It reads the specified video file frame by frame.
# 3. Pothole Detection: An object detection model (YOLO) scans each frame
#    to draw bounding boxes around potholes.
# 4. Road End Detection (Conceptual): To detect road ends, you would typically
#    use a 'semantic segmentation' model. This type of model "paints" the
#    pixels of the drivable area. The code includes a placeholder for this logic.
#
# ---
#
# SOFTWARE & SETUP:
# pip install opencv-python torch torchvision ultralytics
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    # --- 1. Validate Input Paths ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # --- 2. Load AI Models ---
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

    # --- 3. Initialize Video File Reader ---
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

        # --- 5. Perform Pothole Detection ---
        pothole_results = pothole_model(frame)
        annotated_frame = pothole_results[0].plot()

        # --- 6. Perform Road End Detection (Conceptual) ---
        # As before, this section is a placeholder for a segmentation model.
        # It would identify the drivable area.
        # road_end_results = road_end_model(frame)
        # ... (logic to overlay segmentation mask) ...

        # --- 7. Display the Output ---
        cv2.imshow("Road Hazard Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 8. Cleanup ---
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- Set up Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(description="Detect potholes and other road hazards in a video file.")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO model file (e.g., best.pt).")
    parser.add_argument("--video", required=True, help="Path to the video file to be processed.")
    
    args = parser.parse_args()
    
    run_hazard_detector(args.model, args.video)
