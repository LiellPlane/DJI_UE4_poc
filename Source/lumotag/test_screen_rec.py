import numpy as np
import time
from video_recorder import VideoRecorder

def create_text_frame(width, height, text, x, y):
    """Create a frame with text at specified position"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw text using numpy operations
    text_size = 20
    text_width = len(text) * text_size
    text_height = text_size
    
    # Calculate text position
    x_start = max(0, min(x, width - text_width))
    y_start = max(0, min(y, height - text_height))
    
    # Draw text (simple rectangle for now)
    frame[y_start:y_start+text_height, x_start:x_start+text_width] = 255
    return frame

def main():
    # Video parameters
    width = 640
    height = 480
    fps = 30
    
    # Initialize video recorder
    recorder = VideoRecorder(width, height, fps)
    recorder.start_recording("test_recording.mp4")
    
    # Text parameters
    text = "Test Recording"
    x = width // 2
    y = height // 2
    x_vel = 2
    y_vel = 1
    
    try:
        print("Starting recording... Press Ctrl+C to stop")
        frame_count = 0
        start_time = time.time()
        
        while True:
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
            
            # Write frame to recorder
            recorder.write_frame(frame)
            
            # Print progress every second
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                print(f"Recording at {frame_count/elapsed:.1f} FPS")
                frame_count = 0
                start_time = time.time()
            
            # Control frame rate
            time.sleep(1/fps)
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        recorder.stop_recording()
        print("Recording saved!")

if __name__ == "__main__":
    main() 