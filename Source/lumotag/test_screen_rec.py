import numpy as np
import time
import cv2
from video_recorder import VideoRecorder

def create_text_frame(width, height, text, x, y):
    """Create a frame with text at specified position"""
    # Create frame in BGR format (3 channels, uint8)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw text using OpenCV for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)  # White text
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate text position to keep it within frame
    x_start = max(0, min(x, width - text_width))
    y_start = max(text_height, min(y, height))
    
    # Draw text
    cv2.putText(frame, text, (x_start, y_start), font, font_scale, color, thickness)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), font, 0.5, color, 1)
    
    return frame

def main():
    # Video parameters
    width = 640
    height = 480
    fps = 30
    # Initialize video recorder
    recorder = VideoRecorder(width, height)
    
    try:
        recorder.start_recording("test_recording.mp4")
        
        # Text parameters
        text = "Test Recording"
        x = width // 2
        y = height // 2
        x_vel = 2
        y_vel = 1
        
        print("Starting recording... Press Ctrl+C to stop")
        frame_count = 0
        start_time = time.time()
        last_frame_time = time.time()
        
        while True:
            # Calculate time since last frame
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # Only process frame if enough time has passed
            if elapsed >= 1.0/fps:
                # Update text position
                x += x_vel
                y += y_vel
                
                # Bounce off edges
                if x <= 0 or x >= width - 200:
                    x_vel = -x_vel
                if y <= 0 or y >= height:
                    y_vel = -y_vel
                
                # Create frame with moving text
                frame = create_text_frame(width, height, text, x, y)
                
                try:
                    # Write frame to recorder
                    recorder.write_frame(frame)
                    frame_count += 1
                    
                    # Print progress every second
                    if current_time - start_time >= 1.0:
                        print(f"Recording at {frame_count/(current_time - start_time):.1f} FPS")
                        frame_count = 0
                        start_time = current_time
                        
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    break
                    
                last_frame_time = current_time
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        recorder.stop_recording()
        print("Recording saved!")

if __name__ == "__main__":
    main() 