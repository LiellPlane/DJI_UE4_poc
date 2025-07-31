import subprocess
import factory


class Voice(factory.VoiceBase):
    # Note - if this fails it might be because the speaker isn't plugged in
    # on Raspberry pi 5 you need to USB to 3.5mm jack converter, and it should
    # just work
    
    def speaker(self, in_box):
        """Ideally this should inherit from threading.thread
        and override init and run - but w/e this works for now"""
        
        while True:
            message = in_box.get(block=True)
            try:
                # Use espeak directly via subprocess - simple and works!
                subprocess.run(['espeak', '-s', '200', message], check=True)
            except Exception as e:
                print(f"espeak failed: {e}, message was: {message}")
