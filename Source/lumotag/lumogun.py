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
    GUN_CONFIGURATION = factory.stryker_config()
else:
    raise Exception("Could not detect platform")



def main():

    # initialise components of lumogun
    voice = sound.Voice()
    voice.speak(f"{GUN_CONFIGURATION.model_name}")
    relay = lumogun.Relay(GUN_CONFIGURATION)
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    #accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.CSI_Camera()
    display = lumogun.display(GUN_CONFIGURATION)
    messenger = rabbit_mq.messenger(GUN_CONFIGURATION)
    voice.speak("all devices healthy")

    # set partial functions
    set_torch = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["torch"])

    set_laser = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["laser"])

    set_trigger= partial(
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
        
        in_msg = messenger.check_in_box()
        if in_msg is not None:
            msg = json.loads(msgs.bytes_to_str(in_msg))
            msg = msgs.Report(**msg)
            in_ts = msg.timestamp
            received_ts = msgs.get_epoch_ts()
            #print("hit report lag", received_ts-in_ts)

            if msg.msg_type == msgs.MessageTypes.HELLO.value:
                if msg.my_id == GUN_CONFIGURATION.my_id:
                    voice.speak("CONNECTED")
                else:
                    voice.speak("Player connected, " + msg.msg_string)

            if msg.msg_type == msgs.MessageTypes.ERROR.value:
                print(f"Message ERROR (is me={msg.my_id==GUN_CONFIGURATION.my_id}): {msg.msg_string}")

            if msg.my_id != GUN_CONFIGURATION.my_id:
                if msg.msg_type == msgs.MessageTypes.HIT_REPORT.value:
                    if msg.img_as_str is not None:
                        display.display_output(
                            msgs.decode_image_from_str(msg.img_as_str))
                    time.sleep(1)

        GUN_CONFIGURATION.loop_wait()

        #accelerometer.update_vel()
        results_trig_positions = (triggers.test_states())

        is_torch_reqd = results_trig_positions[GUN_CONFIGURATION.rly_torch]
        is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.rly_triggerclick]

        #set_torch(state=is_torch_reqd)
        #set_laser(state=is_torch_reqd)

        if is_trigger_reqd:
            result=GUN_CONFIGURATION.trigger_debounce.trigger_oneshot(
                True,
                msgs.package_send_report,
                msgs.MessageTypes.HIT_REPORT.value,
                image_capture.last_img,
                "some twat",
                messenger,
                GUN_CONFIGURATION,
                "lol QQ l2p"
            
            )
            print(result)
        trigger_ready = set_trigger(state=is_trigger_reqd)
        #if trigger_ready and is_trigger_reqd:
        #    voice.speak("BANG")
        #if is_torch_reqd is True:
        display.display_output(next(image_capture))
        #else:
           #display.display_output(accelerometer.get_visual())

        # if is_trigger_reqd is True:
        #     msgs.package_send_report(
        #         type_=msgs.MessageTypes.HIT_REPORT.value,
        #         image=image_capture.last_img,
        #         gun_config=GUN_CONFIGURATION,
        #         messenger=messenger,
        #         target="some twat",
        #         message_str="lol QQ l2p"
        #     )

    raise RuntimeError("something broke out of loop")

if __name__ == '__main__':
    main()
