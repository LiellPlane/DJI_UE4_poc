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
        
        # FFmpeg command with minimal settings
        command = [
            'ffmpeg',
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # input from pipe
            '-c:v', 'libx264',  # software encoder
            '-b:v', '500k',  # lower bitrate
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',  # fastest encoding preset
            '-loglevel', 'debug',  # Show all debug info
            str(output_path)
        ]
        
        try:
            print(f"Starting FFmpeg with command: {' '.join(command)}")
            # Create a pipe for stderr to capture FFmpeg errors
            stderr_pipe = subprocess.PIPE
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_pipe,
                text=False,  # Keep binary mode for stdin
                bufsize=10*1024*1024  # Increase buffer size
            )
            
            # Check if FFmpeg started successfully
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode('utf-8', errors='replace')
                print(f"FFmpeg startup error: {error}")
                raise Exception(f"FFmpeg failed to start: {error}")
                
            self.is_recording = True
            self.last_chunk_time = time.time()
            print(f"Started recording to {output_path}")
            
            # Start a thread to monitor FFmpeg's output
            def monitor_output():
                error_buffer = []
                while self.is_recording and self.process.poll() is None:
                    try:
                        # Read both stdout and stderr
                        stdout_line = self.process.stdout.readline()
                        stderr_line = self.process.stderr.readline()
                        
                        if stdout_line:
                            msg = stdout_line.decode('utf-8', errors='replace').strip()
                            print(f"FFmpeg stdout: {msg}")
                            error_buffer.append(f"stdout: {msg}")
                        if stderr_line:
                            msg = stderr_line.decode('utf-8', errors='replace').strip()
                            print(f"FFmpeg stderr: {msg}")
                            error_buffer.append(f"stderr: {msg}")
                            
                            # If we see a critical error, store it
                            if "error" in msg.lower():
                                self.last_error = msg
                    except Exception as e:
                        print(f"Error reading FFmpeg output: {e}")
                
                # If process died, print all accumulated errors
                if self.process.poll() is not None:
                    print("\nFFmpeg process died. Last errors:")
                    for error in error_buffer[-10:]:  # Show last 10 messages
                        print(f"  {error}")
                    
                    # Try to read any remaining error output
                    try:
                        remaining_error = self.process.stderr.read().decode('utf-8', errors='replace')
                        if remaining_error:
                            print("\nRemaining error output:")
                            print(remaining_error)
                    except Exception as e:
                        print(f"Error reading remaining output: {e}")
            
            self.output_monitor = threading.Thread(target=monitor_output, daemon=True)
            self.output_monitor.start()
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            self.process = None
            raise  # Re-raise the exception to handle it in the calling code
        
    def write_frame(self, frame):
        if not self.is_recording or self.process is None:
            raise RuntimeError("Not recording or process not started!")
            
        # Validate frame format
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
            
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape {frame.shape} does not match expected shape {(self.height, self.width, 3)}")
            
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype {frame.dtype} is not uint8")
            
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
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
            except BrokenPipeError as e:
                if self.process.poll() is not None:
                    error = self.process.stderr.read().decode('utf-8', errors='replace')
                    if not error:
                        error = "Unknown error - FFmpeg process died unexpectedly"
                    raise RuntimeError(f"FFmpeg process died with error: {error}") from e
                raise RuntimeError("Broken pipe to FFmpeg process") from e
            except Exception as e:
                raise RuntimeError(f"Error writing to FFmpeg: {str(e)}") from e
            
            # Update last frame time
            self.last_frame_time = current_time
            
            # Check if FFmpeg is still running
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode('utf-8', errors='replace')
                if not error:
                    error = "Unknown error - FFmpeg process died unexpectedly"
                raise RuntimeError(f"FFmpeg process died: {error}")
                
        except Exception as e:
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
            except Exception as e:
                # Try to kill the process if it's still running
                if self.process.poll() is None:
                    try:
                        self.process.kill()
                    except Exception as kill_error:
                        raise RuntimeError(f"Failed to stop recording and kill process: {str(e)}, kill error: {str(kill_error)}") from e
                raise RuntimeError(f"Failed to stop recording: {str(e)}") from e
            finally:
                self.process = None
                self.is_recording = False
                
    def __del__(self):
        if hasattr(self, 'is_recording'):  # Check if attribute exists
            self.stop_recording() 