import os
import factory
import sound
from functools import partial
import msgs
import time
import decode_clothID_v2 as decode_clothID
#  detect what OS we are on - test environment (Windows) or production (pi hardware)
RASP_PI_4_OS = "armv7l"

if hasattr(os, 'uname') is False:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
elif os.uname()[-1] == RASP_PI_4_OS:
    print("raspberry presence detected, loading hardware libraries")
    import raspberry_hardware as lumogun
else:
    raise Exception("Could not detect platform")

# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)


def main():
    # initialise components of lumogun
    voice = sound.Voice()
    voice.speak(f"{GUN_CONFIGURATION.model}")
    relay = lumogun.Relay(GUN_CONFIGURATION)
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    #accelerometer = lumogun.Accelerometer()
    image_capture = lumogun.CSI_Camera(factory.HQ_GS_Cam_vidmodes)
    voice.speak("cam")
    img = next(image_capture)
    if img is None:
        raise Exception("Error with camera")
    voice.speak("ok display")
    display = lumogun.display(GUN_CONFIGURATION)
    display.display_output(img)
    voice.speak("ok")
    messenger = lumogun.Messenger(GUN_CONFIGURATION)
    workingdata = decode_clothID.WorkingData()
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
    
    trigger_debounce = GUN_CONFIGURATION.trigger_debounce.trigger_oneshot_simple

    # if user is holding down trigger on boot up, quit
    # application
    results_trig_positions = (triggers.test_states())
    is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.rly_triggerclick]
    if is_trigger_reqd:
        raise Exception("Trigger detected on boot-up - exit app")

    cnt = 0 
    while True:
        cnt += 1

        for msg in messenger.check_in_box():
            in_msg = msgs.parse_input_msg(msg)
            if in_msg.success is False:
                errmsg = in_msg.error
                print("Input Message Err:", errmsg)
            else:
                msg_body = in_msg.msg_body

                if msg_body.msg_type == msgs.MessageTypes.HEARTBEAT.value:
                    print(f"heartbeat in from {msg_body.my_id}")
                    continue

                if msg_body.msg_type == msgs.MessageTypes.HELLO.value:
                    if msg_body.my_id == GUN_CONFIGURATION.my_id:
                        voice.speak("CONNECTED")
                    else:
                        voice.speak("new player, " + msg_body.msg_string)
                    continue

                if msg_body.msg_type == msgs.MessageTypes.TEST.value:
                    print("test input message OK")
                    continue

                if msg_body.msg_type == msgs.MessageTypes.ERROR.value:
                    print(f"Message ERROR (is me={msg_body.my_id==GUN_CONFIGURATION.my_id}): {msg_body.msg_string}")
                    continue

                if msg_body.my_id != GUN_CONFIGURATION.my_id:
                    if msg_body.msg_type == msgs.MessageTypes.HIT_REPORT.value:
                        if msg_body.img_as_str is not None:
                            display.display_output(
                                msgs.decode_image_from_str(msg_body.img_as_str))
                        time.sleep(1)
                    continue

        GUN_CONFIGURATION.loop_wait()

        #accelerometer.update_vel()
        results_trig_positions = (triggers.test_states())

        is_torch_reqd = results_trig_positions[GUN_CONFIGURATION.rly_torch]
        is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.rly_triggerclick]

        set_torch(state=is_torch_reqd, strobe_cnt=0)
        set_laser(state=is_torch_reqd, strobe_cnt=0)

        # if user presses trigger - use one-shot debounce (so not constantly firing
        # when active). Relays also have debounces for electrical stability
        # when user releases trigger - do not need a debounce - deactivate immediately
        result=trigger_debounce(is_trigger_reqd)
        if is_trigger_reqd is True:
            if result is True:
                set_trigger(state=True, strobe_cnt=0) # click noise from relay only
                msgs.package_send_report(
                    type_=msgs.MessageTypes.HIT_REPORT.value,
                    image=image_capture.last_img,
                    gun_config=GUN_CONFIGURATION,
                    messenger=messenger,
                    target="some twat",
                    message_str="lol QQ l2p"
                )
                voice.speak("BANG")
        else:
            set_trigger(state=False, strobe_cnt=0) # click noise from relay only

        cam_img = next(image_capture)
        #img_with_analysis = decode_clothID.find_lumotag(cam_img, workingdata)
        display.display_output(cam_img)

    raise RuntimeError("something broke out of loop")

if __name__ == '__main__':
    main()

