import os
import factory
import sound
from functools import partial
import rabbit_mq
import msgs
from dataclasses import asdict
import json
import time
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
    voice.speak(f"{GUN_CONFIGURATION.model_name},  LOAD.")
    relay = lumogun.Relay(GUN_CONFIGURATION)
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.CSI_Camera()
    display = lumogun.display()
    #messenger = rabbit_mq.messenger(GUN_CONFIGURATION)
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
        
        # #in_msg = messenger.check_in_box()
        # if in_msg is not None:
        #     #print(in_msg)
        #     break
        #     msg = json.loads(msgs.bytes_to_str(in_msg))
        #     msg = msgs.Report(**msg)
        #     if msg.my_id == GUN_CONFIGURATION.my_id:
        #         break
        #     if msg.img_as_str is None:
        #         pass
        #     display.display_output(msgs.decode_image_from_str(msg.img_as_str))
        #     time.sleep(1)
        GUN_CONFIGURATION.loop_wait()

        accelerometer.update_vel()
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

        if is_trigger_reqd is True:
            msg_to_send = msgs.Report(
                my_id=GUN_CONFIGURATION.my_id,
                target="UNKNOWN",
                timestamp="TBD",
                img_as_str=msgs.encode_img_to_str(display.last_img)
            )

            msg = msgs.str_to_bytes(json.dumps(asdict(msg_to_send)))
            msg_dict = json.loads(msgs.bytes_to_str(msg))
            msg_dataclass = msgs.Report(**msg_dict)
            img_decoded = msgs.decode_image_from_str(msg_dataclass.img_as_str)
            #https://amroamroamro.github.io/mexopencv/matlab/cv.imencode.html
            display.display_output(img_decoded)
            #messenger.send_message("plop")

if __name__ == '__main__':
    main()
