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
        # male voices - slower speed for better short word articulation
        ['-v', 'en+m1', '-s', '120', '-a', '180', '-p', '45'],
        ['-v', 'en+m7', '-s', '130', '-a', '190', '-p', '35'],  # low male
        # female voices - slower speed for better short word articulation
        ['-v', 'en+f2', '-s', '125', '-a', '185', '-p', '55'],
        ['-v', 'en+f3', '-s', '135', '-a', '180', '-p', '60'],
        # novelty voices - slower speed for better short word articulation
        ['-v', 'en+croak', '-s', '110', '-a', '170', '-p', '45'],
        ['-v', 'en+whisper', '-s', '100', '-a', '160', '-p', '50'],
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
            try:
                subprocess.run(self._build_cmd(message), check=True)
            except Exception as e:
                print(f"espeak failed: {e}, message was: {message}")
