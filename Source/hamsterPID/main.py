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
    def __init__(self, Kp, Ki, Kd, setpoint=0, min_dt=0.02, max_integral=20.0, integral_error_threshold=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.min_dt = min_dt  # Minimum time between updates (default 20ms)
        self.max_integral = max_integral  # Maximum integral value for anti-windup
        self.integral_error_threshold = integral_error_threshold  # Only accumulate integral when error exceeds this
        
        self.prev_error = 0
        self.integral = 0
        
        # Create timer - automatically tracks dt between update calls
        self.timer = TimeDiffObject()
        
        # Cache for when updates are too fast
        self.last_output = 0.0
        self.last_P = 0.0
        self.last_I = 0.0
        self.last_D = 0.0
        self.last_error = 0.0
        self.last_dt = 0.0
    
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
        
        # Integral term (accumulated error over time) with anti-windup
        # Only accumulate integral when error is significant to reduce overshoot
        if abs(error) > self.integral_error_threshold:
            self.integral += error * dt
        # Clamp integral to prevent windup (limit to reasonable range)
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        I = self.Ki * self.integral
        
        # Derivative term (rate of change of error)
        # Clamp dt to prevent huge derivatives from very small time steps
        dt_for_derivative = max(dt, self.min_dt)
        derivative = (error - self.prev_error) / dt_for_derivative if dt_for_derivative > 0 else 0
        D = self.Kd * derivative
        
        # Update previous error for next iteration
        self.prev_error = error
        
        # Calculate and cache output
        self.last_output = P + I + D
        self.last_P = P
        self.last_I = I
        self.last_D = D
        self.last_error = error
        self.last_dt = dt
        
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
            # print(i)
            if (i // 20) % 2 == 0:
                if i+20 < self.background_img.shape[1]:
                    self.background_img[:, i:i+20 ,:] = 200

        self.hamster_img = np.full((50, 50, 3), 0, dtype=np.uint8)
        self.scroll_position = 0
        self.hamster_position = 200
        self.timer = TimeDiffObject()
        self.timer.reset()
        
        # Text overlay cache
        self.overlay_text = None
        self.overlay_text_img = None
    
    def set_overlay_text(self, text):
        """Set overlay text to display at top of image. Cached and reused until changed."""
        if text != self.overlay_text:
            self.overlay_text = text
            # Create text image with black background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Create text image with padding
            padding = 10
            self.overlay_text_img = np.full((text_height + baseline + padding * 2, text_width + padding * 2, 3), bg_color, dtype=np.uint8)
            
            # Draw text
            cv2.putText(self.overlay_text_img, text, (padding, text_height + padding), 
                       font, font_scale, color, thickness, cv2.LINE_AA)
    
    def _scroll_background(self, conveyor_speed_m_s):
        delta_time = self.timer.get_dt() * 10
        self.scroll_position += conveyor_speed_m_s * delta_time
        self.timer.reset()
        return np.roll(self.background_img.copy(), int(self.scroll_position), axis=1)
    
    def update(self, hamster_delta, conveyor_speed_m_s):
        self.hamster_position += hamster_delta
        x = int(self.hamster_position)
        y = 250
        img = self._scroll_background(conveyor_speed_m_s)
        img[:, 1000//2, :] = (0,0,255)
        
        # Add overlay text at top if it exists
        if self.overlay_text_img is not None:
            overlay_h, overlay_w = self.overlay_text_img.shape[:2]
            bg_h, bg_w = img.shape[:2]
            # Center the overlay horizontally, place at top
            overlay_x = max(0, min(bg_w - overlay_w, (bg_w - overlay_w) // 2))
            img[0:overlay_h, overlay_x:overlay_x + overlay_w] = self.overlay_text_img
        
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
        return self.hamster_position - (self.background_img.shape[1] //2 )
    
    def display(self, frame):
        """Display the frame in a window."""
        cv2.imshow("Hamster HUD", frame)
        cv2.waitKey(1)

def main():
    scambi = SpeedElement("scambi", 0.1)
    conveyor =  SpeedElement("scambi", 0.01)
    pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.0, setpoint=0.0, max_integral=20.0, integral_error_threshold=1.0)
    hud = HUD()

    scambi.set_speed(15)
    conveyor.set_speed(5)
    while True:
        scambiPosDelta = scambi.updateAndGetDeltaPos(outsideforce_m_s=conveyor.current_speed_m)
        position_offset = hud.get_hamster_offset()
        correction = pid.update(position_offset)
        target_speed = (conveyor.current_speed_m + correction) / 10
        conveyor.set_speed(target_speed)
        hud.set_overlay_text(f"correction: {round(correction,2)} | target_speed: {round(target_speed,2)}")
        conveyor.updateAndGetDeltaPos(outsideforce_m_s=0)
        frame = hud.update(scambiPosDelta, conveyor_speed_m_s=conveyor.current_speed_m)
        hud.display(frame)
        # print(f"current speed {scambi.current_speed_m}")
        time.sleep(0.1)
        # if scambi.current_speed_m > 10:
        #     scambi.set_speed(-20)
        #     conveyor.set_speed(-5)
        # if scambi.current_speed_m < -10:
        #     scambi.set_speed(0)
        #     conveyor.set_speed(10)
        scambi.set_speed(random.randint(-20,20))

if __name__ == "__main__":
    main()
