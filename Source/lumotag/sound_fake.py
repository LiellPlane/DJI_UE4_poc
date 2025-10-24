import factory
import time

class Voice(factory.VoiceBase):

    def speaker(self, in_box):
        #  TODO
        """Ideally this should inherit from threading.thread
        and override init and run - but w/e this works for now"""
        while True:
            message = in_box.get(block=True)
            self.speak_blocking(message)
            time.sleep(len(message)*0.030)
            
    def speak_blocking(self, message):
        print(f"Pretending to say {message}")