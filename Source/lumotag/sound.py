#pip install pyttsx3
import pyttsx3
import factory


class Voice(factory.VoiceBase):
    # Note - if this fails it might be because the speaker isn't plugged in
    # on Raspberry pi 5 you need to USB to 3.5mm jack converter, and it should
    # just work
    def speaker(self, in_box):
        #  TODO
        """Ideally this should inherit from threading.thread
        and override init and run - but w/e this works for now"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        engine.setProperty('volume', 10)
        while True:
            message = in_box.get(block=True)
            engine.say(message)
            engine.runAndWait()
