import time
import cv2
import numpy as np
from abc import ABC

def num_to_range(num, inMin, inMax, outMin, outMax):
      return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))


class TimeDiffObject:
    """stopwatch function"""

    def __init__(self) -> None:
        self._start_time = time.perf_counter()

    def get_dt(self) -> float:
        """gets time in seconds since last reset/init"""
        self._stop_time = time.perf_counter()
        difference_secs = self._stop_time-self._start_time
        return difference_secs

    def reset(self):
        self._start_time = time.perf_counter()

class SpeedElement():
    def __init__(self, name, response_sec_per_m_per_sec):
        self.name=name
        self.current_speed_m = 0
        self.target_speed_m = 0
        self.last_steadystate = 0
        self.response_sec_per_m_per_sec = response_sec_per_m_per_sec # it will take N seconds to reach N m/sec
        self.position_m = 200 # position along conveyor
        self.timer_speed = TimeDiffObject()
        self.timer_pos = TimeDiffObject()
        self.timer_speed.reset()
        self.timer_pos.reset()
        self.set_insta_speed(0)

    def set_insta_speed(self, target_speed_m_sec: int) -> bool:
        self.current_speed_m = target_speed_m_sec
        self.current_speed_m = target_speed_m_sec
        self.last_steadystate = target_speed_m_sec
        return True

    def set_speed(self, target_speed_m_sec:int) -> bool:
        if target_speed_m_sec == self.target_speed_m:
            return False
        if abs(self.current_speed_m - self.target_speed_m) > 0.01:
            return False # not completed last state
        self.last_steadystate = self.current_speed_m # this might be a bit weird and break the lerping potentially
        self.timer_speed.reset()
        self.target_speed_m = target_speed_m_sec
        return True

    def update_state(self, outsideforce_m_s: float):
        delta = self.timer_speed.get_dt()
        time_to_ss = self.response_sec_per_m_per_sec * abs(self.last_steadystate - self.target_speed_m)
        if time_to_ss == 0:
            print("time to ss is zero")
            return
        lerp_position = delta / time_to_ss
        lerp_position = min(1, lerp_position)
        lerp_position = max(-1, lerp_position)
        lerp_position = round(lerp_position ** 3,3) # <<<--- lerp formula here
        # map speed
        if lerp_position < 0:
            self.current_speed_m = num_to_range(
                num=lerp_position,
                outMin=self.last_steadystate,
                outMax=self.target_speed_m,
                inMin=0,
                inMax=-1)

        else:

            self.current_speed_m = num_to_range(
                num=lerp_position,
                outMin=self.last_steadystate,
                outMax=self.target_speed_m,
                inMin=0,
                inMax=1)
        delta_postime = self.timer_pos.get_dt() * 10 
        self.position_m += ((self.current_speed_m + outsideforce_m_s) * delta_postime)
        self.timer_pos.reset()
    
class Conveyor():
    def __init__(self):
        self.responsetime = None #not sure what to do with this yet
        self.timer = TimeDiffObject()
        self.current_speed_m = 0
        self.target_speed_m = 0

    def setSpeed(target_speed_m_sec:int, responsetime_ms: int):
        pass

class HUD():
    def __init__(self):
        self.background_img = np.full((500, 1000, 3), 255, dtype=np.uint8)
        self.hamster_img = np.full((50, 50, 3), 0, dtype=np.uint8)
    
    def update(self, hamster_position):
        img = self.background_img.copy()
        x, y = (int(round(coord)) for coord in hamster_position)
        hamster_h, hamster_w = self.hamster_img.shape[:2]
        bg_h, bg_w = img.shape[:2]
        if (
            x < 0
            or y < 0
            or x + hamster_w > bg_w
            or y + hamster_h > bg_h
        ):
            raise ValueError("hamster_position would place hamster_img outside the background image")
        img[y:y + hamster_h, x:x + hamster_w] = self.hamster_img
        return img

def main():
    scambi = SpeedElement("scambi", 0.1)
    hud = HUD()

    scambi.set_speed(15)
    while True:
        scambi.update_state(outsideforce_m_s=5)
        frame = hud.update(hamster_position=(scambi.position_m,40))
        cv2.imshow("Hamster HUD", frame)
        cv2.waitKey(1)
        print(f"current speed {scambi.current_speed_m}")
        time.sleep(0.1)
        if scambi.current_speed_m > 10:
            scambi.set_speed(-20)
        if scambi.current_speed_m < -10:
            scambi.set_speed(0)

if __name__ == "__main__":
    main()
