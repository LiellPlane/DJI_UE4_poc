import subprocess
import numpy as np
import time
import os
from pathlib import Path
import threading

class VideoRecorder:
    def __init__(self, width, height, fps=30):
        # Initialize all attributes first
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_recording = False
        self.chunk_duration = 10      # seconds
        self.last_chunk_time = None
        self.last_frame_time = 0
        self.frame_interval = 1.0 / fps  # time between frames
        
        # Always use home directory for recordings
        home_dir = Path.home()
        self.output_dir = home_dir / "recordings"
        self.output_dir.mkdir(exist_ok=True)
        print(f"Recordings will be saved to: {self.output_dir}")
        
    def start_recording(self, filename=None):
        if self.is_recording:
            print("Already recording!")
            return
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            
        output_path = self.output_dir / filename
        
        # FFmpeg command with standard encoding
        command = [
            'ffmpeg',
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # input from pipe
            '-c:v', 'libx264',  # software encoder - watch out
            '-b:v', '2M',  # bitrate
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',  # faster encoding preset
            '-loglevel', 'error',  # Only show errors
            str(output_path)
        ]
        
        try:
            # Create a pipe for stderr to capture FFmpeg errors
            stderr_pipe = subprocess.PIPE
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_pipe
            )
            
            # Check if FFmpeg started successfully
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode()
                raise Exception(f"FFmpeg failed to start: {error}")
                
            self.is_recording = True
            self.last_chunk_time = time.time()
            print(f"Started recording to {output_path}")
            
            # Start a thread to monitor FFmpeg's stderr
            def monitor_stderr():
                while self.is_recording and self.process.poll() is None:
                    line = self.process.stderr.readline().decode().strip()
                    if line:
                        print(f"FFmpeg: {line}")
            
            self.stderr_monitor = threading.Thread(target=monitor_stderr, daemon=True)
            self.stderr_monitor.start()
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            self.process = None
            raise  # Re-raise the exception to handle it in the calling code
        
    def write_frame(self, frame):
        if not self.is_recording or self.process is None:
            print("Not recording or process not started!")
            return
            
        # Validate frame format
        if not isinstance(frame, np.ndarray):
            print("Error: Frame must be a numpy array")
            return
            
        if frame.shape != (self.height, self.width, 3):
            print(f"Error: Frame shape {frame.shape} does not match expected shape {(self.height, self.width, 3)}")
            return
            
        if frame.dtype != np.uint8:
            print(f"Error: Frame dtype {frame.dtype} is not uint8")
            return
            
        try:
            # Control frame rate
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time
            
            # If we're falling behind, log a warning
            if time_since_last_frame > self.frame_interval * 2:
                print(f"Warning: Frame rate falling behind. Last frame was {time_since_last_frame:.3f}s ago")
            
            # Skip frame if too soon
            if time_since_last_frame < self.frame_interval:
                return  # Skip this frame to maintain target FPS
                
            # Write frame and flush to ensure it's sent
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            
            # Update last frame time
            self.last_frame_time = current_time
            
            # Check if FFmpeg is still running
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode()
                raise Exception(f"FFmpeg process died: {error}")
                
        except Exception as e:
            print(f"Error writing frame: {e}")
            self.stop_recording()
            raise  # Re-raise the exception to handle it in the calling code
                
        if time.time() - self.last_chunk_time > self.chunk_duration:
            print("Starting new chunk...")
            self.stop_recording()
            self.start_recording()  # new chunk
                
    def stop_recording(self):
        if self.is_recording and self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait()
                print("Recording stopped")
            except Exception as e:
                print(f"Error stopping recording: {e}")
            finally:
                self.process = None
                self.is_recording = False
                
    def __del__(self):
        if hasattr(self, 'is_recording'):  # Check if attribute exists
            self.stop_recording() 