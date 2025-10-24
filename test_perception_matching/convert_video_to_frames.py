import cv2
import os
import glob

# Hardcoded paths for frames to video
VIDEO_PATH = r"C:\Users\liell\Downloads\phantommenace_lightsabre.mp4"
OUTPUT_FOLDER = r"D:\phantom_menance_frames"

# Hardcoded paths for video to frames
FRAMES_FOLDER = r"C:\Working\GIT\DJI_UE4_poc\test_perception_matching\image_sequence"
OUTPUT_VIDEO_PATH = r"C:\Users\liell\Downloads\output_video.mp4"
FPS = 30

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

def frames_to_video():
    # Get all image files from the folder (supports jpg, jpeg, png)
    frame_files = sorted(glob.glob(os.path.join(FRAMES_FOLDER, "*.jpg")))
    frame_files += sorted(glob.glob(os.path.join(FRAMES_FOLDER, "*.jpeg")))
    frame_files += sorted(glob.glob(os.path.join(FRAMES_FOLDER, "*.png")))
    frame_files = sorted(list(set(frame_files)))  # Remove duplicates and sort
    
    if not frame_files:
        print(f"Error: No image files found in {FRAMES_FOLDER}")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"Error: Could not read first frame {frame_files[0]}")
        return
    
    height, width, _ = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_VIDEO_PATH}")
        return
    
    print(f"Creating video from {len(frame_files)} frames at {FPS} FPS...")
    
    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping...")
            continue
        
        out.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames...")
    
    out.release()
    print(f"Video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    # extract_frames()
    frames_to_video()
