import factory
import time

class Voice(factory.VoiceBase):

    def speaker(self, in_box):
        #  TODO
        """Ideally this should inherit from threading.thread
        and override init and run - but w/e this works for now"""
        while True:
            message = in_box.get(block=True)
            print(f"Pretending to say {message}")
            time.sleep(len(message)*0.030)
            
