import time
import cv2
import numpy as np
from abc import ABC
import random




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



class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, min_dt=0.02):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.min_dt = min_dt  # Minimum time between updates (default 20ms)
        
        self.prev_error = 0
        self.integral = 0
        
        # Create timer - automatically tracks dt between update calls
        self.timer = TimeDiffObject()
        
        # Cache for when updates are too fast
        self.last_output = 0.0
    
    def update(self, measured_value):
        """
        Update PID controller with new measurement.
        If called too quickly (< min_dt), returns cached output.
        
        Args:
            measured_value: Current measured value (e.g., position offset)
        
        Returns:
            PID control output (new or cached)
        """
        # Get time since last update
        dt = self.timer.get_dt()
        
        # If called too fast, return cached output
        if dt < self.min_dt:
            return self.last_output
        
        # Reset timer for next call
        self.timer.reset()
        
        # Calculate error (setpoint - measured)
        error = self.setpoint - measured_value
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term (accumulated error over time)
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = self.Kd * derivative
        
        # Update previous error for next iteration
        self.prev_error = error
        
        # Calculate and cache output
        self.last_output = P + I + D
        
        return self.last_output
    
    def reset(self):
        """Reset controller state (useful when starting fresh)"""
        self.prev_error = 0
        self.integral = 0
        self.last_output = 0.0
        self.timer.reset()
        
class SpeedElement():
    def __init__(self, name, response_sec_per_m_per_sec):
        self.name=name
        self.current_speed_m = 0
        self.target_speed_m = 0
        self.last_steadystate = 0
        self.response_sec_per_m_per_sec = response_sec_per_m_per_sec # it will take N seconds to reach N m/sec
        self.timer_speed = TimeDiffObject()
        self.timer_pos = TimeDiffObject()
        self.timer_speed.reset()
        self.timer_pos.reset()
        self.set_insta_speed(0)
    
    def _num_to_range(self, num, inMin, inMax, outMin, outMax):
        return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax- outMin))

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

    def updateAndGetDeltaPos(self, outsideforce_m_s: float):
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
            self.current_speed_m = self._num_to_range(
                num=lerp_position,
                outMin=self.last_steadystate,
                outMax=self.target_speed_m,
                inMin=0,
                inMax=-1)

        else:

            self.current_speed_m = self._num_to_range(
                num=lerp_position,
                outMin=self.last_steadystate,
                outMax=self.target_speed_m,
                inMin=0,
                inMax=1)
        delta_postime = self.timer_pos.get_dt() * 10 
        delta_position = ((self.current_speed_m + outsideforce_m_s) * delta_postime)
        self.timer_pos.reset()
        return delta_position
    

class HUD():
    def __init__(self):
        self.background_img = np.full((500, 1000, 3), 255, dtype=np.uint8)
        for i in range(0, self.background_img.shape[1], 20):
            print(i)
            if (i // 20) % 2 == 0:
                if i+20 < self.background_img.shape[1]:
                    self.background_img[:, i:i+20 ,:] = 200

        self.hamster_img = np.full((50, 50, 3), 0, dtype=np.uint8)
        self.scroll_position = 0
        self.hamster_position = 200
        self.timer = TimeDiffObject()
        self.timer.reset()
    
    def _scroll_background(self, conveyor_speed_m_s):
        delta_time = self.timer.get_dt() * 10
        self.scroll_position += conveyor_speed_m_s * delta_time
        self.timer.reset()
        return np.roll(self.background_img.copy(), int(self.scroll_position), axis=1)
    
    def update(self, hamster_delta, conveyor_speed_m_s):
        self.hamster_position += hamster_delta
        x = int(self.hamster_position)
        y = 40
        img = self._scroll_background(conveyor_speed_m_s)
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
    def get_hamster_offset(self):
        return self.background_img.shape[1] //2 

def main():
    scambi = SpeedElement("scambi", 0.1)
    conveyor =  SpeedElement("scambi", 0.01)
    hud = HUD()

    scambi.set_speed(15)
    conveyor.set_speed(5)
    while True:
        scambiPosDelta = scambi.updateAndGetDeltaPos(outsideforce_m_s=conveyor.current_speed_m)
        conveyor.updateAndGetDeltaPos(outsideforce_m_s=0)
        print(f"conveyor.current_speed_m {conveyor.current_speed_m}")
        frame = hud.update(scambiPosDelta, conveyor_speed_m_s=conveyor.current_speed_m)
        cv2.imshow("Hamster HUD", frame)
        cv2.waitKey(1)
        print(f"current speed {scambi.current_speed_m}")
        time.sleep(0.1)
        if scambi.current_speed_m > 10:
            scambi.set_speed(-20)
            conveyor.set_speed(-5)
        if scambi.current_speed_m < -10:
            scambi.set_speed(0)
            conveyor.set_speed(10)

if __name__ == "__main__":
    main()
