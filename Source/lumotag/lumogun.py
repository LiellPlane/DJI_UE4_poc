import os
import factory
from functools import partial
#  detect what OS we are on - test environment on production (real hardware)
RASP_PI_4_OS = "armv7l"
if hasattr(os, 'uname') is False:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
elif os.uname()[-1] == RASP_PI_4_OS:
    print("raspberry presence detected, loading hardware libraries")
    import raspberry_hardware as lumogun
else:
    raise Exception("Could not detect platform")



def main():
    # initialise components of lumogun
    test_config = factory.TZAR_config()
    relay = lumogun.Relay(test_config)
    triggers = lumogun.Triggers(test_config)
    accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.CSI_Camera()
    display = lumogun.display()

    # set partial functions
    set_torch = partial(relay.set_relay, relaypos=1)
    set_laser = partial(relay.set_relay, relaypos=2)
    set_clicker = partial(relay.set_relay, relaypos=3)

    while True:
        test_config.loop_wait()
        
        vel = accelerometer.update_vel()
        results_trig_positions = (triggers.test_states())

        is_torch_reqd = results_trig_positions[test_config.rly_torch]
        is_trigger_reqd = results_trig_positions[test_config.rly_triggerclick]

        set_torch(state=is_torch_reqd)
        set_laser(state=is_torch_reqd)
        set_clicker(state=is_trigger_reqd)

        if is_torch_reqd is True:
            display.display_output(next(image_capture))
        else:
            display.display_output(accelerometer.get_visual())

        

        print(f"{vel} {results_trig_positions}")

if __name__ == '__main__':
    main()