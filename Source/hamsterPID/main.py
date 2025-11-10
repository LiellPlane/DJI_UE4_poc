import time


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

class Hamster():
    def __init__(self):
        self.hamsterName="scambi"
        self.current_speed_m = 0
        self.target_speed_m = 0
        self.last_steadystate = 0
        self.response_sec_per_m_per_sec = 0.1 # it will take N seconds to reach N m/sec
        self.position_m = 0 # position along conveyor
        self.timer = TimeDiffObject()
        self.timer.reset()

    def start_running(self, target_speed_m_sec:int) -> None:
        if target_speed_m_sec == self.target_speed_m:
            return
        self.last_steadystate = self.current_speed_m # this might be a bit weird and break the lerping potentially
        self.timer.reset()
        self.target_speed_m = target_speed_m_sec

    def update_state(self):
        delta = self.timer.get_dt()
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


    
class Conveyor():
    def __init__(self):
        self.responsetime = None #not sure what to do with this yet
        self.timer = TimeDiffObject()
        self.current_speed_m = 0
        self.target_speed_m = 0

    def setSpeed(target_speed_m_sec:int, responsetime_ms: int):
        pass


def main():
    scambi = Hamster()
    conveyor_belt = Conveyor()

    scambi.start_running(15)
    print(scambi.current_speed_m)
    while True:
        scambi.update_state()
        print(scambi.current_speed_m)
        time.sleep(0.1)
        if scambi.current_speed_m > 10:
            scambi.start_running(-20)
        if scambi.current_speed_m < -10:
            scambi.start_running(0)

if __name__ == "__main__":
    main()
