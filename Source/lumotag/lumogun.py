import os
import factory
from functools import partial
#  detect what OS we are on - test environment on production (real hardware)
RASP_PI_4_OS = "armv7l"
if hasattr(os, 'uname') is False:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
    GUN_CONFIGURATION = factory.simitzar_config()
elif os.uname()[-1] == RASP_PI_4_OS:
    print("raspberry presence detected, loading hardware libraries")
    import raspberry_hardware as lumogun
    GUN_CONFIGURATION = factory.TZAR_config()
else:
    raise Exception("Could not detect platform")



def main():

    # initialise components of lumogun
    
    relay = lumogun.Relay(GUN_CONFIGURATION)
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.CSI_Camera()
    display = lumogun.display()

    # set partial functions
    set_torch = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["torch"])

    set_laser = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["laser"])

    set_clicker = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["clicker"])


    # if user is holding down trigger on boot up, quit
    # application
    is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.rly_triggerclick]
    if is_trigger_reqd:
        raise Exception("Trigger detected on boot-up - exit app")

    while True:

        GUN_CONFIGURATION.loop_wait()

        vel = accelerometer.update_vel()
        results_trig_positions = (triggers.test_states())

        is_torch_reqd = results_trig_positions[GUN_CONFIGURATION.rly_torch]
        is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.rly_triggerclick]

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
