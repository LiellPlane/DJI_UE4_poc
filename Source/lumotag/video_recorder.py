import subprocess
import numpy as np
import time
from pathlib import Path

class VideoRecorder:
    def __init__(self, width, height, fps=29.98):
        # Initialize all attributes first
        self.sequence_number = 0
        self.width = width
        self.height = height
        self.target_fps = fps  # Target FPS
        self.process = None
        self.is_recording = False
        self.chunk_duration = 30      # seconds
        self.overlap_duration = 1     # 1 second overlap between chunks
        self.last_chunk_time = None
        self.frame_buffer = []        # buffer for overlap frames
        self.max_buffer_frames = int(self.target_fps * self.overlap_duration)  # number of frames to buffer
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = None  # Will be calculated based on actual frame delivery
        self.fps_window = 100  # Number of frames to use for FPS calculation
        self.frame_times = []  # Store recent frame times for FPS calculation
        self.last_frame = None  # Store the last frame for potential duplication
        self.min_frame_interval = 1.0 / (self.target_fps * 1.5)  # Minimum time between frames (50% faster than target)
        self.max_frame_interval = 1.0 / (self.target_fps * 0.5)  # Maximum time between frames (50% slower than target)
        
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
            filename = f"recording_{timestamp}_{self.sequence_number}.mp4"
            self.sequence_number += 1
            
        output_path = self.output_dir / filename
        
        print(f"Starting FFmpeg with dimensions: {self.width}x{self.height}")
        
        # FFmpeg command with hardware acceleration
        command = [
            'ffmpeg',
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.target_fps),
            '-i', '-',  # input from pipe
            '-c:v', 'libx264',  # software encoder
            '-preset', 'ultrafast',  # fastest encoding preset
            '-tune', 'zerolatency',  # minimize latency
            '-b:v', '1000k',  # reduced bitrate
            '-pix_fmt', 'yuv420p',
            '-threads', '2',  # reduced thread count
            '-cpu-used', '8',  # maximum CPU usage reduction
            '-loglevel', 'warning',
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
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            self.process = None
            raise  # Re-raise the exception to handle it in the calling code
        
    def write_frame(self, frame):
        if not self.is_recording or self.process is None:
            raise RuntimeError("Not recording or process not started!")
            
        # Initialize start time if this is the first frame
        if self.start_time is None:
            self.start_time = time.time()
            
        # Validate frame format
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
            
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape mismatch. Got: {frame.shape}, Expected: {(self.height, self.width, 3)}")
            
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype {frame.dtype} is not uint8")
            
        try:
            # Check if FFmpeg is still running before writing
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode('utf-8', errors='replace')
                if not error:
                    error = f"FFmpeg process terminated unexpectedly. Frame dimensions: {frame.shape}, FFmpeg expected: {self.width}x{self.height}"
                raise RuntimeError(f"FFmpeg process died: {error}")
            
            # Write frame and flush to ensure it's sent
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
                
                # Store frame in buffer for overlap
                self.frame_buffer.append(frame.copy())
                if len(self.frame_buffer) > self.max_buffer_frames:
                    self.frame_buffer.pop(0)
                
            except BrokenPipeError as e:
                # Check FFmpeg's status and get any error output
                if self.process.poll() is not None:
                    error = self.process.stderr.read().decode('utf-8', errors='replace')
                    if not error:
                        error = f"FFmpeg process terminated unexpectedly. Frame dimensions: {frame.shape}, FFmpeg expected: {self.width}x{self.height}"
                    raise RuntimeError(f"FFmpeg process died: {error}") from e
                raise RuntimeError("Broken pipe to FFmpeg process") from e
            except Exception as e:
                raise RuntimeError(f"Error writing to FFmpeg: {str(e)}") from e
            
            # Update frame count
            self.frame_count += 1
            
        except Exception as e:
            self.stop_recording()
            raise  # Re-raise the exception to handle it in the calling code
                
        if time.time() - self.last_chunk_time > self.chunk_duration:
            # First stop the current recording
            self.stop_recording()
            
            # Then start a new recording
            self.start_recording()
            
            # Write buffered frames to the new recording
            for buffered_frame in self.frame_buffer:
                try:
                    self.process.stdin.write(buffered_frame.tobytes())
                    self.process.stdin.flush()
                except Exception as e:
                    print(f"Error writing buffered frame: {e}")
                    break  # Stop if we encounter an error
                
    def stop_recording(self):
        if self.is_recording and self.process is not None:
            try:
                # Check if FFmpeg is still running
                if self.process.poll() is None:
                    try:
                        self.process.stdin.close()
                        self.process.wait(timeout=2)  # Wait up to 2 seconds for FFmpeg to finish
                    except subprocess.TimeoutExpired:
                        print("FFmpeg did not finish in time, forcing termination")
                        self.process.kill()
                        self.process.wait()
                else:
                    # FFmpeg already died, get the error
                    error = self.process.stderr.read().decode('utf-8', errors='replace')
                    if error:
                        print(f"FFmpeg terminated with error: {error}")
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