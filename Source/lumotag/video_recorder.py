import subprocess
import numpy as np
import time
import os
from pathlib import Path

class VideoRecorder:
    def __init__(self, width, height, fps=30):
        # Initialize all attributes first
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_recording = False
        self.chunk_duration = 60  # seconds
        self.last_chunk_time = None
        
        # Always use home directory for recordings
        home_dir = Path.home()
        self.output_dir = home_dir / "recordings"
        self.output_dir.mkdir(mode=0o755, exist_ok=True)
        print(f"Recordings will be saved to: {self.output_dir}")
        
    def start_recording(self, filename=None):
        if self.is_recording:
            return
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            
        output_path = self.output_dir / filename
        
        # FFmpeg command with Raspberry Pi 5 hardware acceleration
        command = [
            'ffmpeg',
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # input from pipe
            '-c:v', 'h264_v4l2m2m',  # Raspberry Pi 5 hardware encoder
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.is_recording = True
            self.last_chunk_time = time.time()
            print(f"Started recording to {output_path}")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            self.process = None
        
    def write_frame(self, frame):
        if not self.is_recording or self.process is None:
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
            self.process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"Error writing frame: {e}")
            self.stop_recording()
                
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