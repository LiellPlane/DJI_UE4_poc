import subprocess
import random
import factory


class Voice(factory.VoiceBase):
    # Note - if this fails it might be because the speaker isn't plugged in
    # on Raspberry pi 5 you need USB to 3.5mm jack converter, and it should
    # just work

    # Class-level profiles of eSpeak voice settings
    # Each profile is a flat list of CLI args (excluding the message)
    VOICE_PROFILES = [
        # female voices - faster speed
        ['-v', 'en+f2', '-s', '205', '-a', '185', '-p', '55'],
        ['-v', 'en+f3', '-s', '215', '-a', '180', '-p', '60'],
    ]

    def __init__(self) -> None:
        # Choose a random profile and keep it for this instance
        self._selected_profile = random.choice(self.VOICE_PROFILES)
        try:
            print(f"Voice: selected profile {self._selected_profile}")
        except Exception:
            pass
        # Start the speaker thread from the base class AFTER selecting voice
        super().__init__()

    def _build_cmd(self, message: str) -> list[str]:
        # self._selected_profile = random.choice(self.VOICE_PROFILES)
        return ['espeak', *self._selected_profile, message]

    def speaker(self, in_box):
        """Ideally this should inherit from threading.thread
        and override init and run - but w/e this works for now"""
        
        while True:
            message = in_box.get(block=True)
            self.speak_blocking(message)

    def speak_blocking(self, message):
            try:
                subprocess.run(self._build_cmd(message), check=True)
            except Exception as e:
                print(f"espeak failed: {e}, message was: {message}")