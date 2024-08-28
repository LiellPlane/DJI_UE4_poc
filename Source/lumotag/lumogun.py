# need to add switch for reversed camera in shared memory
# its a bad technique but will have to make do

import factory
import random
from functools import partial
import msgs
import queue
import datetime
import time
#import decode_clothID_v2 as decode_clothID
import analyse_lumotag
import img_processing
from utils import time_it, get_platform
from my_collections import _OS
# need this import to detect lumogun types (subclasses)
import configs
#  detect what OS we are on - test environment (Windows) or production (pi hardware)
PLATFORM = get_platform()
PRINT_DEBUG = configs.get_lumofind_config(PLATFORM).PRINT_DEBUG

if PLATFORM == [_OS.WINDOWS]:
    print("raspberry presence failed, loading test libraries")
    import fake_raspberry_hardware as lumogun
    import sound_fake as sound
elif PLATFORM == _OS.RASPBERRY:
    print("raspberry presence detected, loading hardware libraries")
    import raspberry5_hardware as lumogun
    import sound as sound
elif PLATFORM == _OS.MAC_OS:
    print("disgusting Mac detected, loading fake hardware libraries")
    import fake_raspberry_hardware as lumogun
    import sound_fake as sound
else:
    import fake_raspberry_hardware as lumogun
    import sound_fake as sound

    #raise Exception("Could not detect platform")

# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)

class AnalysisTimeoutException(Exception):
    pass

def main():
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    # if user is holding down trigger on boot up, quit
    # application
    # initialise components of lumogun
    voice = sound.Voice()

    if get_platform() == _OS.RASPBERRY:
        for _ in range(0, 2):
            voice.speak("cancel")
            results_trig_positions = (triggers.test_states())
            if any([True for i in results_trig_positions.values() if i is True]):
                voice.speak("bye")
                time.sleep(3)
                raise Exception("Trigger detected on boot-up - exit app")
            time.sleep(2)

    voice.speak(f"{GUN_CONFIGURATION.model}")
    relay = lumogun.Relay(GUN_CONFIGURATION)
    
    #accelerometer = lumogun.Accelerometer()
    #image_capture_longrange = lumogun.CSI_Camera(GUN_CONFIGURATION.video_modes)
    #image_capture_longrange = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes)
    image_capture_longrange = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes)
    image_capture_shortrange = lumogun.CSI_Camera_async_flipflop(GUN_CONFIGURATION.video_modes_closerange)
    slice_details_close_range = img_processing.get_internal_section(
                        image_capture_shortrange.get_res(),
                        GUN_CONFIGURATION.internal_img_crop_sr)
    slice_details_long_range = img_processing.get_internal_section(
                        image_capture_longrange.get_res(),
                        GUN_CONFIGURATION.internal_img_crop_lr)
    image_analysis = []
    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_longrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_longrange.get_safe_mem_details,
        slice_details=slice_details_long_range,
        img_shrink_factor=None,
        OS_friendly_name="cam1inner",
        camera_source_class_ref = image_capture_longrange,
        config=configs.get_lumofind_config(PLATFORM)))

    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_longrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_longrange.get_safe_mem_details,
        slice_details=None,
        OS_friendly_name="cam1macro",
        img_shrink_factor=GUN_CONFIGURATION.img_subsmple_factor,
        camera_source_class_ref = image_capture_longrange,
        config=configs.get_lumofind_config(PLATFORM)))

    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_shortrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_shortrange.get_safe_mem_details,
        slice_details=None,
        OS_friendly_name="cam2inner",
        img_shrink_factor=GUN_CONFIGURATION.img_subsmple_factor,
        camera_source_class_ref = image_capture_shortrange,
        config=configs.get_lumofind_config(PLATFORM)))
    
    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_shortrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_shortrange.get_safe_mem_details,
        slice_details=slice_details_close_range,
        OS_friendly_name="cam2macro",
        img_shrink_factor=None,
        camera_source_class_ref = image_capture_shortrange,
        config=configs.get_lumofind_config(PLATFORM)))
    
    #time.sleep(100000)
    img = next(image_capture_longrange)
    if img is None:
        raise Exception("broken long-range image source")
    img2 = next(image_capture_shortrange)
    if img2 is None:
        raise Exception("broken close-range image source")
        
    voice.speak("cam")
    # img = next(image_capture_longrange)

    # if img is None:
    #     raise Exception("Error with camera")
    
    display = lumogun.display(GUN_CONFIGURATION)
    # this generates the affine transform dictionary key, which
    # is used by other processes for annotating the screen such as found targets
    # so best to do it before any other processes come back with data
    display.generate_output_affine(next(image_capture_longrange))
    display.generate_output_affine(next(image_capture_shortrange))
    voice.speak("ok display")

    messenger = lumogun.Messenger(GUN_CONFIGURATION)
    #workingdata = decode_clothID.WorkingData()
    file_system = lumogun.filesystem()
    voice.speak("all devices healthy")
    # display.display_output(img)
    # while True:
    #display.display_output(next(image_capture2))
    longrangedetails = img_processing.CamDisplayTransform(
        cam_image_shape= next(image_capture_longrange).shape
    )
    closerangedetails = img_processing.CamDisplayTransform(
        cam_image_shape= next(image_capture_shortrange).shape
    )
    transform_details = img_processing.TransformsDetails(
        longrange_to_shortrange_perwarp=file_system.get_closerange_to_longrange_transform(),
        closerange_to_display=closerangedetails,
        longrange_to_display=longrangedetails,
        transition_steps=99,
        transition_time_secs=0.3,
        display_image_shape=GUN_CONFIGURATION.screen_size,
        displayrotation=GUN_CONFIGURATION.screen_rotation,
        slice_details_close_range=slice_details_close_range,
        slice_details_long_range=slice_details_long_range
    )

    transform_manager = img_processing.TransformManager(transformdetails=transform_details)
    crosshair_lerper = img_processing.lerped_add_crosshair()


    # create demo player
    players = {
        "demoplayer":
        factory.PlayerInfoBox(
            playername="demoplayer",
            playergraphic=None,
            _gun_config=GUN_CONFIGURATION
        )}


    # set partial functions
    set_torch = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["torch"])

    set_laser = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["laser"])

    set_trigger = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["clicker"])
    
    trigger_debounce = GUN_CONFIGURATION.trigger_debounce
    zoom_debounce = GUN_CONFIGURATION.trigger_debounce
    #torch_debounce = GUN_CONFIGURATION.torch_debounce.trigger_oneshot_simple


    cnt = 0 
    TEMP_DEBUG_trigger_cnt = 0
    TEMP_fake_light = False
    while True:
        with time_it("TOTAL TIME FOR EVERYTHING", debug=PRINT_DEBUG):
            cnt += 1
            with time_it("get next image", debug=PRINT_DEBUG):
                cap_img = next(image_capture_longrange)
                cap_img_closerange = next(image_capture_shortrange)
                # this is bad code - should come as package with the image -
                # but in easy of modularity have to do it like this for now
            with time_it("start analysis", debug=PRINT_DEBUG):
                for img_analyser in image_analysis:
                    img_analyser.trigger_analysis()


            with time_it("check messaging", debug=PRINT_DEBUG):
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

            with time_it("gun states set", debug=PRINT_DEBUG):
                #accelerometer.update_vel()
                results_trig_positions = (triggers.test_states())

                is_torch_reqd = results_trig_positions[GUN_CONFIGURATION.button_torch]
                is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.button_trigger]
                is_zoom_reqd = results_trig_positions[GUN_CONFIGURATION.button_rear]
                

                #### testing cycle
                # is_torch_reqd = TEMP_fake_light
                # if random.randint(0,100) < 2:
                #     TEMP_fake_light = not TEMP_fake_light
                    
                # if random.randint(0,3000) < 2:
                #     is_trigger_reqd = not is_trigger_reqd
                # if random.randint(0, 100) < 4:
                #     transform_manager.trigger_transition()
                # in this case 
                # result = torch_debounce(is_torch_reqd)
                # if result is True:
                #     set_torch(state=is_torch_reqd, strobe_cnt=GUN_CONFIGURATION.light_strobe_cnt)
                #     set_laser(state=is_torch_reqd, strobe_cnt=0)
                set_torch(state=is_torch_reqd, strobe_cnt=0)

                set_laser(state=is_torch_reqd, strobe_cnt=0)


                result_zoom = zoom_debounce.trigger_1shot_simple_High(is_zoom_reqd)
                if result_zoom:
                    transform_manager.trigger_transition()
                # desired behaviour: 
                # User presses trigger - gun fires immediately
                # after 0.N seconds, relay clicks off
                # user can now fire again immediately
                # any other behaviour during refractory period is ignored
                result = trigger_debounce.trigger_1shot_simple_High(is_trigger_reqd)
                if result is True:
                    
                    # true will only be available as an impulse after
                    # pulling trigger, then go low again - but
                    # mem state of debouncer will remain high
                    msgs.package_send_report(
                        type_=msgs.MessageTypes.HIT_REPORT.value,
                        image=image_capture_longrange.last_img,
                        gun_config=GUN_CONFIGURATION,
                        messenger=messenger,
                        target="some twat",
                        message_str="lol QQ l2p"
                    )

                    # debugging code to capture images
                    #if cap_img is not None:
                    # TEMP_DEBUG_trigger_cnt += 1
                    # file_system.save_image(cap_img,message=f"_longrange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                    # file_system.save_image(cap_img_closerange,message=f"_closerange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                set_trigger(
                    state=trigger_debounce.get_heldstate(),
                    strobe_cnt=0
                    ) # click noise from relay only

            with time_it("gun image stuff", debug=PRINT_DEBUG):

                transition_i = transform_manager.get_deltatime_transition()
                transition_state = transform_manager.get_transition_state()
                display_active_image = cap_img_closerange
                if  transition_state == img_processing.CameraTransitionState.CLOSERANGE:
                    display_active_image = cap_img_closerange
                    output_image = display.generate_output_affine(display_active_image)
                    transition_i=0
                if transition_state == img_processing.CameraTransitionState.LONGRANGE:
                    display_active_image = cap_img
                    output_image = display.generate_output_affine(display_active_image)
                    transition_i=transform_manager.transformdetails.transition_steps-1
                else:
                    with time_it("execute affine transform", debug=PRINT_DEBUG):
                        mat = transform_manager.CR_all_transition_m[transition_i]
                        cr_img = img_processing.apply_perp_transform(mat, cap_img_closerange, display.emptyscreen)

                        mat = transform_manager.LR_all_transition_m[transition_i]
                        lr_img = img_processing.apply_perp_transform(mat, cap_img, display.emptyscreen)

                        percent_done = transition_i/(transform_manager.transformdetails.transition_steps-1)
                        cr_img = img_processing.darken_image(cr_img, 1-percent_done)
                        combo_image = img_processing.overlay_warped_image_alpha_feathered(cr_img, lr_img, percent_done)
                        
                        #combo_image = img_processing.radial_motion_blur(combo_image)
                        output_image = img_processing.gray2rgb(combo_image)

                
                # with time_it("execute affine transform", debug=PRINT_DEBUG):
                #     img = display.TESTgenerate_output_affine2cam(cap_img,cap_img_closerange)


                with time_it("wait for image analysis", debug=PRINT_DEBUG):
                    analysis = {}
                    for img_analyser in image_analysis:
                        # TODO: get this properly. Some complexity due to reversed shape so using
                        # protected member :(
                        res_for_affine_transform_lookup = img_analyser.camera_source_class_ref._store_res #BAD LIELL!!! 
                        try:
                            result = img_analyser.analysis_output_q.get(block=True, timeout=5)
                            if result:

                                if res_for_affine_transform_lookup not in analysis:
                                    analysis[res_for_affine_transform_lookup] = []
                                analysis[res_for_affine_transform_lookup].extend(result)
                        except queue.Empty:
                            raise AnalysisTimeoutException("Timeout occurred while waiting for image analysis.")

                if len(analysis) > 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    file_system.save_image(
                        cap_img,message=f"_longrange_cnt{timestamp}cnt"
                        )
                    file_system.save_image(
                        cap_img_closerange,message=f"_closerange_cnt{timestamp}cnt"
                        )
                    file_system.save_image(
                        cap_img,message=f"_CUNT"
                        )
                    file_system.save_image(
                        cap_img_closerange,message=f"_FUCKINGBICTHCUNT"
                        )
                    
                with time_it("add internal section", debug=PRINT_DEBUG):
                    display.add_internal_section_region(
                        display_active_image.shape,
                        output_image,
                        transform_manager.get_lerped_targetzone_slice(transition_i),
                        transform_manager.get_display_affine_transformation(transition_i))

                with time_it("add graphics: crosshair/analyics", debug=PRINT_DEBUG):

                    crosshair_lerper.add_cross_hair(
                        image=output_image,
                        adapt=True,
                        target_acquired=(len(analysis) > 0)
                    )


                    # use the image shape to determine image analysis provenance
                    # we don't want to draw for instance close-range target graphics
                    # on long-range active image and vice-versa
                    # don't draw if we are transitiong (for now)
                    # might be a nice effect though

                    if transition_state != img_processing.CameraTransitionState.TRANSITIONING:
                        if transition_state == img_processing.CameraTransitionState.CLOSERANGE:
                            # filter for close range origin analysis
                            display.add_target_tags(
                                output=output_image,
                                graphics={
                                    k: v for k, v in analysis.items()
                                    if k == image_capture_shortrange._store_res
                                    }
                                )
                        if transition_state == img_processing.CameraTransitionState.LONGRANGE:
                            # filter for long range origin analysis
                            display.add_target_tags(
                                output=output_image,
                                graphics={
                                    k: v for k, v in analysis.items()
                                    if k == image_capture_longrange._store_res
                                    }
                                )

                with time_it("add graphics: player info", debug=PRINT_DEBUG):
                    display.add_playerinfo_graphics(
                        output=output_image,
                        players=players,
                        analysis=analysis
                        )

                with time_it("display image", debug=PRINT_DEBUG):
                    display.display_method(output_image)
                
                # if len(analysis) > 0:
                #     file_system.save_image(cap_img)

if __name__ == '__main__':
    main()
