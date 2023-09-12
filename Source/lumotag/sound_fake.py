import factory


class Voice(factory.VoiceBase):

    def speaker(in_box):
        #  TODO
        """Ideally this should inherit from threading.thread
        and override init and run - but w/e this works for now"""
        while True:
            message = in_box.get(block=True)
            print(f"Pretending to say {message}")
