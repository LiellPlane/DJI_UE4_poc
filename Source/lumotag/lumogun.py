import os
import factory

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
    test_config = factory.leviathon_config()
    config = lumogun.config()
    relay = lumogun.Relay()
    triggers = lumogun.Triggers()
    accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.CSI_Camera()
    display = lumogun.display()

    while True:
        config.loop_wait()
        
        vel = accelerometer.update_vel()
        results_trig_positions = (triggers.test_states())

        req_torch = results_trig_positions[config.torch]
        req_trig = results_trig_positions[config.triggerclick]

        relay.set_relay(relaypos=1, state=req_torch)
        relay.set_relay(relaypos=2, state=req_torch)
        relay.set_relay(relaypos=3, state=req_trig)

        if req_torch is True:
            display.display_output(next(image_capture))
        else:
            display.display_output(accelerometer.get_visual())

        

        print(f"{vel} {results_trig_positions}")

if __name__ == '__main__':
    main()