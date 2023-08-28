#pip install pyttsx3
import threading
from queue import Queue
import pyttsx3


class Voice():

    def __init__(self) -> None:
        """Class to provide synthetic
        voice prompts or alerts"""
        self.in_box = Queue(maxsize = 10)
        self.t = threading.Thread(
            target=speaker,args=(self.in_box,))
        self.t.start()

    def speak(
            self,
            message: str):
        # use  in_box._qsize() to prevent
        # blowing it up
        if self.in_box._qsize() >= self.in_box.maxsize - 1:
            self.in_box.queue.clear()
            self.in_box.put(
                "Voice buffer overflow",
                block=False)
        else:
            self.in_box.put(
                message,
                block=False)


def speaker(in_box):
    #TODO
    """Ideally this should inherit from threading.thread
    and override init and run - but w/e this works for now"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 10)
    while True:
        message = in_box.get(block=True)
        engine.say(message)
        engine.runAndWait()
