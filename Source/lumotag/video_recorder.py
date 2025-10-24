import subprocess
import numpy as np
import time
from pathlib import Path
import threading
from collections import deque

# Correct Samsung Galaxy S20 FPS - NTSC standard
SAMSUNG_GALAXY_S20_FPS = 29.98  # Corrected to match actual Samsung timing

class VideoRecorder:
    def __init__(self, width, height, fps=SAMSUNG_GALAXY_S20_FPS):
        # Initialize all attributes first
        self.sequence_number = 0
        self.width = width
        self.height = height
        self.target_fps = fps 
        self.frame_interval = 1.0 / fps  # Time between frames
        self.process = None
        self.is_recording = False
        self.chunk_duration = 60 * 3     # seconds
        self.overlap_duration = 1     # 1 second overlap between chunks
        self.last_chunk_time = None
        self.frame_buffer = []        # buffer for overlap frames
        self.max_buffer_frames = int(self.target_fps * self.overlap_duration)
        self.frame_count = 0
        self.start_time = None
        
        # Timing control variables
        self.last_frame_time = None
        self.recording_start_time = None
        self.frame_queue = deque()
        self.frame_thread = None
        self.stop_thread = False
        
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
        
        print(f"Starting FFmpeg with dimensions: {self.width}x{self.height} at {self.target_fps} fps")
        print(f"Output file: {output_path}")
        
        # FFmpeg command - let FFmpeg handle frame rate conversion
        command = [
            'ffmpeg',
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-i', '-',  # input from pipe - no input frame rate specified
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-r', str(self.target_fps),  # Output frame rate
            '-b:v', '1000k',
            '-pix_fmt', 'yuv420p',
            '-threads', '2',
            '-loglevel', 'error',  # Only show errors
            str(output_path)
        ]
        
        try:
            print(f"Starting FFmpeg...")
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=10*1024*1024
            )
            
            # Give FFmpeg a moment to start
            time.sleep(0.1)
            
            # Check if FFmpeg started successfully
            if self.process.poll() is not None:
                error = self.process.stderr.read().decode('utf-8', errors='replace')
                print(f"FFmpeg startup error: {error}")
                raise Exception(f"FFmpeg failed to start: {error}")
                
            self.is_recording = True
            self.last_chunk_time = time.time()
            self.recording_start_time = time.time()
            self.last_frame_time = None
            
            # Clear any existing queue
            self.frame_queue.clear()
            
            # Start frame timing thread
            self.stop_thread = False
            self.frame_thread = threading.Thread(target=self._frame_writer_thread)
            self.frame_thread.daemon = True
            self.frame_thread.start()
            
            print(f"Successfully started recording to {output_path}")
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            if self.process:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
            raise
    
    def _frame_writer_thread(self):
        """Thread that writes frames to FFmpeg at the correct timing."""
        thread_start_time = time.time()
        expected_frame_time = thread_start_time
        
        while not self.stop_thread and self.is_recording:
            current_time = time.time()
            
            # Check if it's time for the next frame
            if current_time >= expected_frame_time:
                if self.frame_queue:
                    try:
                        frame = self.frame_queue.popleft()
                        if self.process and self.process.poll() is None:
                            self.process.stdin.write(frame.tobytes())
                            self.process.stdin.flush()
                            
                            # Store frame in buffer for overlap
                            self.frame_buffer.append(frame.copy())
                            if len(self.frame_buffer) > self.max_buffer_frames:
                                self.frame_buffer.pop(0)
                                
                            self.frame_count += 1
                            
                    except Exception as e:
                        print(f"Error writing frame in thread: {e}")
                        break
                
                # Calculate next frame time
                expected_frame_time += self.frame_interval
                
                # If we're falling behind significantly, skip to current time
                if expected_frame_time < current_time - (self.frame_interval * 2):
                    expected_frame_time = current_time
                    print("Warning: Frame timing fell behind, resynchronizing")
            
            # Small sleep to prevent busy waiting
            time.sleep(0.001)  # 1ms
        
    def write_frame(self, frame):
        """Queue a frame for writing at the correct timing."""
        # Better error reporting
        if not self.is_recording:
            raise RuntimeError("Not recording! Call start_recording() first.")
        
        if self.process is None:
            raise RuntimeError("FFmpeg process not started!")
            
        # Check if FFmpeg process died
        if self.process.poll() is not None:
            try:
                error = self.process.stderr.read().decode('utf-8', errors='replace')
                error_msg = f"FFmpeg process died unexpectedly. Exit code: {self.process.returncode}"
                if error:
                    error_msg += f"\nFFmpeg error: {error}"
                raise RuntimeError(error_msg)
            except Exception as e:
                raise RuntimeError(f"FFmpeg process died and couldn't read error: {e}")
            
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
        
        # Add frame to queue (limit queue size to prevent memory issues)
        if len(self.frame_queue) < 100:  # Limit queue size
            self.frame_queue.append(frame.copy())
        else:
            print("Warning: Frame queue full, dropping frame")
                
        # Check for chunk duration
        if time.time() - self.last_chunk_time > self.chunk_duration:
            try:
                self._new_chunk()
            except Exception as e:
                print(f"Error during chunk transition: {e}")
                # Try to restart recording
                try:
                    self.start_recording()
                except Exception as restart_error:
                    print(f"Failed to restart recording: {restart_error}")
                    raise RuntimeError(f"Chunk transition failed and couldn't restart: {restart_error}")
    
    def _new_chunk(self):
        """Start a new chunk while maintaining overlap."""
        if not self.is_recording:
            return
            
        print(f"Starting new chunk after {self.chunk_duration} seconds")
        
        # Save current state
        saved_buffer = self.frame_buffer.copy()
        current_sequence = self.sequence_number
        
        # Stop current recording cleanly
        self._stop_current_chunk()
        
        # Temporarily disable recording to prevent frame writes during transition
        self.is_recording = False
        
        try:
            # Start new recording
            self.start_recording()
            
            # Re-queue buffered overlap frames at the beginning
            temp_queue = deque()
            for buffered_frame in saved_buffer:
                temp_queue.append(buffered_frame.copy())
            
            # Add current queue contents after the overlap frames
            while self.frame_queue:
                temp_queue.append(self.frame_queue.popleft())
                
            self.frame_queue = temp_queue
            
            print(f"New chunk started with {len(saved_buffer)} overlap frames")
        except Exception as e:
            print(f"Error during chunk transition: {e}")
            # Try to restart recording
            try:
                self.start_recording()
            except Exception as restart_error:
                print(f"Failed to restart recording: {restart_error}")
                raise RuntimeError(f"Chunk transition failed and couldn't restart: {restart_error}")
        finally:
            # Re-enable recording
            self.is_recording = True
    
    def _stop_current_chunk(self):
        """Stop current chunk without clearing frame queue."""
        if not self.is_recording:
            return
            
        # Stop the frame writer thread
        self.stop_thread = True
        
        # Wait for frame thread to finish
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=2.0)
        
        # Close FFmpeg process
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    try:
                        self.process.stdin.close()
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        print("FFmpeg did not finish in time, forcing termination")
                        self.process.kill()
                        self.process.wait()
                else:
                    error = self.process.stderr.read().decode('utf-8', errors='replace')
                    if error:
                        print(f"FFmpeg terminated with error: {error}")
            except Exception as e:
                print(f"Error stopping chunk: {e}")
                if self.process.poll() is None:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
            finally:
                self.process = None
                
    def stop_recording(self):
        if self.is_recording:
            self._stop_current_chunk()
            self.is_recording = False
            self.frame_queue.clear()
            print("Recording stopped")
                
    def get_status(self):
        """Get current recording status for debugging."""
        status = {
            'is_recording': self.is_recording,
            'process_alive': self.process is not None and self.process.poll() is None,
            'thread_alive': self.frame_thread is not None and self.frame_thread.is_alive(),
            'queue_size': len(self.frame_queue),
            'frame_count': self.frame_count,
            'buffer_size': len(self.frame_buffer)
        }
        
        if self.process is not None:
            status['process_returncode'] = self.process.returncode
            
        return status
    
    def __del__(self):
        if hasattr(self, 'is_recording'):
            self.stop_recording()