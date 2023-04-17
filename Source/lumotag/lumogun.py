import os
import factory
import sound
from functools import partial
import rabbit_mq
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
    voice = sound.Voice()
    voice.speak(f"{GUN_CONFIGURATION.model_name} START")
    relay = lumogun.Relay(GUN_CONFIGURATION)
    voice.speak("reelaay")
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    voice.speak("triggers")
    accelerometer = lumogun.Accelerometer()
    voice.speak("accelerometer")
    image_capture = lumogun.CSI_Camera()
    voice.speak("CSI")
    display = lumogun.display()
    voice.speak("display")
    messenger = rabbit_mq.messenger(GUN_CONFIGURATION)
    voice.speak("messenger")
    voice.speak("all devices healthy")

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
    results_trig_positions = (triggers.test_states())
    is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.rly_triggerclick]
    if is_trigger_reqd:
        raise Exception("Trigger detected on boot-up - exit app")

    cnt = 0 
    while True:
        cnt += 1
        print(f"In Box: {messenger.check_in_box()}")
        messenger.send_message(f"{GUN_CONFIGURATION.model_name} says F U x {cnt}")
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
