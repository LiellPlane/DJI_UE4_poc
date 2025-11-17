import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.prev_error = 0
        self.integral = 0
    
    def update(self, measured_value, dt):
        error = self.setpoint - measured_value
        
        P = self.Kp * error
        
        self.integral += error * dt
        I = self.Ki * self.integral
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = self.Kd * derivative
        
        self.prev_error = error
        
        return P + I + D
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0

class ConveyorSystem:
    def __init__(self, use_feedforward=True, use_feedback=True):
        self.position = 0  # meters from center (positive = forward, negative = backward)
        self.conveyor_speed = 5.0
        
        self.use_feedforward = use_feedforward
        self.use_feedback = use_feedback
        
        # PID tuned for position control
        self.pid = PIDController(Kp=2.0, Ki=0.3, Kd=0.8, setpoint=0)
    
    def simulate_step(self, person_speed, dt):
        # Physics: person's position changes based on speed difference
        # If person runs 7 m/s and conveyor moves 5 m/s, they drift forward at 2 m/s
        speed_difference = person_speed - self.conveyor_speed
        self.position += speed_difference * dt
        
        # Control: determine what conveyor speed should be
        target_speed = 0
        
        if self.use_feedforward:
            # Feedforward: match the person's speed as baseline
            target_speed = person_speed
        
        if self.use_feedback:
            # Feedback: adjust based on position error
            # Negative feedback: if position > 0 (forward), increase conveyor speed
            correction = -self.pid.update(self.position, dt)
            target_speed += correction
        
        self.conveyor_speed = target_speed
        
        return self.position, self.conveyor_speed

# Simulation
dt = 0.02
time = np.arange(0, 20, dt)

def person_speed_profile(t):
    """Person changes their running speed over time"""
    if t < 4:
        return 5.0
    elif t < 7:
        return 5.0 + (t - 4) * 2.0  # Accelerate to 11 m/s
    elif t < 11:
        return 11.0
    elif t < 14:
        return 11.0 - (t - 11) * 2.0  # Decelerate to 5 m/s
    else:
        return 5.0

# Test three control strategies
configs = [
    ('Feedback Only', False, True),
    ('Feedforward Only', True, False),
    ('Feedforward + Feedback', True, True)
]

results = {}

for name, use_ff, use_fb in configs:
    system = ConveyorSystem(use_feedforward=use_ff, use_feedback=use_fb)
    
    positions = []
    conveyor_speeds = []
    person_speeds = []
    
    for t in time:
        ps = person_speed_profile(t)
        pos, cs = system.simulate_step(ps, dt)
        
        positions.append(pos)
        conveyor_speeds.append(cs)
        person_speeds.append(ps)
    
    results[name] = {
        'position': np.array(positions),
        'conveyor_speed': np.array(conveyor_speeds),
        'person_speed': np.array(person_speeds)
    }

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Top plot: Position on conveyor
colors = ['#e74c3c', '#3498db', '#2ecc71']
for i, name in results.keys():
    axes[0].plot(time, results[name]['position'], label=name, 
                linewidth=2.5, color=colors[i])

axes[0].axhline(y=0, color='black', linestyle='--', linewidth=2, 
               alpha=0.6, label='Target (Center)', zorder=10)
axes[0].fill_between(time, -0.1, 0.1, alpha=0.2, color='green', 
                     label='Acceptable range')
axes[0].set_ylabel('Position from Center (m)', fontsize=13, fontweight='bold')
axes[0].set_title('Person\'s Position on Conveyor (Goal: Stay at 0)', 
                 fontsize=15, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 20])

# Bottom plot: Speeds
axes[1].plot(time, results['Feedforward + Feedback']['person_speed'], 
            'k--', linewidth=3, label='Person Speed', alpha=0.7, zorder=5)

for i, name in enumerate(results.keys()):
    axes[1].plot(time, results[name]['conveyor_speed'], 
                label=f'{name} Conveyor Speed', linewidth=2.5, color=colors[i])

axes[1].set_xlabel('Time (s)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Speed (m/s)', fontsize=13, fontweight='bold')
axes[1].set_title('Conveyor Speed Response to Person\'s Speed Changes', 
                 fontsize=15, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 20])

plt.tight_layout()
plt.show()

# Performance summary
print("\n" + "="*75)
print("CONTROL STRATEGY PERFORMANCE SUMMARY")
print("="*75)
for name in results.keys():
    pos = results[name]['position']
    max_err = np.max(np.abs(pos))
    rms_err = np.sqrt(np.mean(pos**2))
    final_err = abs(pos[-1])
    
    print(f"\n{name}:")
    print(f"  Max deviation from center: {max_err:.3f} m")
    print(f"  RMS error:                 {rms_err:.3f} m")
    print(f"  Final position error:      {final_err:.3f} m")

print("\n" + "="*75)