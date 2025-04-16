import cv2
import os

# Hardcoded paths
VIDEO_PATH = r"C:\Users\liell\Downloads\phantommenace_lightsabre.mp4"
OUTPUT_FOLDER = r"D:\phantom_menance_frames"  # Change this to your desired output folder

def extract_frames():
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return
    
    frame_count = 0
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # If frame is not read correctly, break the loop
        if not ret:
            break
        
        # Save the frame
        frame_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    extract_frames()
