# need to add switch for reversed camera in shared memory
# its a bad technique but will have to make do

import factory
from functools import partial
import queue
import datetime
import time
from comms_http import HTTPComms
#import decode_clothID_v2 as decode_clothID
import analyse_lumotag
import img_processing
from decode_clothID_v2 import find_lumotag, find_lumotag_mser
from utils import time_it, get_platform
from my_collections import _OS, HeightWidth
# need this import to detect lumogun types (subclasses)
import configs
# Import fake websocket server for testing

#  detect what OS we are on - test environment (Windows) or production (pi hardware)
PLATFORM = get_platform()
PRINT_DEBUG = configs.get_lumofind_config(PLATFORM).PRINT_DEBUG

if PLATFORM == _OS.WINDOWS:
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
    raise ValueError(f"Unknown platform detected: {PLATFORM}")

# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)



class AnalysisTimeoutException(Exception):
    pass


def save_analysis(result):
    """Debug function"""
    import pickle
    timestamp = datetime.datetime.now().strftime("%Y%M%d%H%M%S%f")[:-3]
    filename = rf"D:\lumotag_training_data\false_positives{timestamp}.pc"
    all_results = []
    for res in result:
        all_results.append(res._2d_samples)
    with open(filename, 'wb') as file:
        pickle.dump(all_results, file)
    with open(filename, 'rb') as file:
        _ = pickle.load(file)


def save_images_if_barcode(analysis, file_system, cap_img, cap_img_closerange):
    """Debug function"""
    if len(analysis) > 0:
        timestamp = datetime.datetime.now().strftime("%H%M%S%f")[:-3]
        file_system.save_image(
            cap_img,message=f"_longrange_cnt{timestamp}cnt"
            )
        file_system.save_image(
            cap_img_closerange,message=f"_closerange_cnt{timestamp}cnt"
            )
                    
def main():

    perfmonitor = factory.Perfmonitor()
    triggers = lumogun.Triggers(GUN_CONFIGURATION)
    # if user is holding down trigger on boot up, quit
    # application
    # initialise components of lumogun
    voice = sound.Voice()
    file_system = lumogun.filesystem()
    status_bar = factory.LumoUI(filesystem_=file_system)
    if get_platform() == _OS.RASPBERRY:
        for _ in range(0, 2):
            voice.speak("cancel")
            voice.wait_for_speak()
            results_trig_positions = (triggers.test_states())
            if any([True for i in results_trig_positions.values() if i is True]):
                voice.speak("bye")
                time.sleep(3)
                raise Exception("Trigger detected on boot-up - exit app")
            time.sleep(2)

    voice.speak(f"{GUN_CONFIGURATION.model}, V zero five.")
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
        OS_friendly_name="longrange_inner_mser",
        camera_source_class_ref = image_capture_longrange,
        lumotag_func=find_lumotag_mser,
        config=configs.get_lumofind_config(PLATFORM)))
 
    # you can remove this one as its to test the mser
    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_longrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_longrange.get_safe_mem_details,
        slice_details=None,
        OS_friendly_name="longrange_macro_mser",
        img_shrink_factor=GUN_CONFIGURATION.img_subsmple_factor,
        camera_source_class_ref = image_capture_longrange,
        lumotag_func=find_lumotag_mser,
        config=configs.get_lumofind_config(PLATFORM)))

    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_longrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_longrange.get_safe_mem_details,
        slice_details=slice_details_long_range,
        img_shrink_factor=None,
        OS_friendly_name="longrange_inner",
        camera_source_class_ref = image_capture_longrange,
        lumotag_func=find_lumotag,
        config=configs.get_lumofind_config(PLATFORM)))

    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_longrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_longrange.get_safe_mem_details,
        slice_details=None,
        OS_friendly_name="longrange_macro",
        img_shrink_factor=GUN_CONFIGURATION.img_subsmple_factor,
        camera_source_class_ref = image_capture_longrange,
        lumotag_func=find_lumotag,
        config=configs.get_lumofind_config(PLATFORM)))

    
    image_analysis.append(analyse_lumotag.ImageAnalyser_shared_mem(
        sharedmem_buffs=image_capture_shortrange.get_mem_buffers(),
        safe_mem_details_func = image_capture_shortrange.get_safe_mem_details,
        slice_details=slice_details_close_range,
        OS_friendly_name="closerange_inner",
        img_shrink_factor=None,
        camera_source_class_ref = image_capture_shortrange,
        lumotag_func=find_lumotag,
        config=configs.get_lumofind_config(PLATFORM)))


    # game_client =(HTTPComms(
    #     sharedmem_buffs_closerange=image_capture_shortrange.get_mem_buffers(),
    #     safe_mem_details_func_closerange=image_capture_shortrange.get_safe_mem_details,
    #     sharedmem_buffs_longrange=image_capture_longrange.get_mem_buffers(),
    #     safe_mem_details_func_longrange=image_capture_longrange.get_safe_mem_details,
    #     images_url="http://LIELLOMEN:8080/api/v1/images/upload",
    #     events_url="http://LIELLOMEN:8080/api/v1/events",
    #     gamestate_url="http://LIELLOMEN:8080/api/v1/gamestate",
    #     OS_friendly_name="shortrange_img_uploader",
    #     user_id="player_1",
    #     upload_timeout=1.0,
    #     poll_interval_seconds=0.3
    # ))

    for image_analyser in image_analysis:
        print("placeholder for analysis time graphs otherwise they get spread out heuristically - put somewhere nicer")
        perfmonitor.manual_measure(f"{image_analyser.OS_friendly_name}", 10)
    perfmonitor.manual_measure("spacemaker", 0) # for visual break between analyusis and system metrics
    #time.sleep(100000)
    voice.speak("test cam 1")
    voice.wait_for_speak()
    img = next(image_capture_longrange)
    if img is None:
        raise Exception("broken long-range image source")
    voice.speak("test cam 2")
    voice.wait_for_speak()
    img2 = next(image_capture_shortrange)
    if img2 is None:
        raise Exception("broken close-range image source")
        
    voice.speak("cams OK")
    voice.wait_for_speak()
    # img = next(image_capture_longrange)

    # if img is None:
    #     raise Exception("Error with camera")
    
    display = lumogun.display(GUN_CONFIGURATION, configs.get_lumofind_config(PLATFORM).SAVE_STREAM)
    # this generates the affine transform dictionary key, which
    # is used by other processes for annotating the screen such as found targets
    # so best to do it before any other processes come back with data
    display.generate_output_affine(next(image_capture_longrange))
    display.generate_output_affine(next(image_capture_shortrange))
    voice.speak("ok display")
    voice.wait_for_speak()

    # messenger = lumogun.Messenger(GUN_CONFIGURATION)
    #workingdata = decode_clothID.WorkingData()
    

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
        factory.PlayerInfoBoxv2(
            playername="testplayer",
            avatar_canvas= HeightWidth(60,60), # get this from the status bar class
            info_box = HeightWidth(60,60)
        ),
        "testself":
        factory.LocalPlayerCard(
            playername="self"
        ),

        }


    # set partial functions
    set_torch = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["torch"])

    # very convoluted way to find out if the torch debouncer is allowing us to trigger
    #can_torch_trigger = relay.debouncers[relay.gun_config.RELAY_IO[GUN_CONFIGURATION.relay_map["torch"]]].can_trigger

    set_laser = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["laser"])

    set_trigger = partial(
        relay.set_relay,
        GUN_CONFIGURATION.relay_map["clicker"])
    

    test_dict = {}
    test_dict["torch"] = [partial(set_torch, (i%2==0), 0) for i in range(1,8)]
    test_dict["laser"] = [partial(set_laser, (i%2==0), 0) for i in range(1,8)]
    test_dict["trigger"] = [partial(set_trigger, (i%2==0), 0) for i in range(1,8)]
    voice.speak(f"testing {len(test_dict)} relays")
    for devicename, _function_list in test_dict.items():
        # voice.speak(devicename)
        # voice.speak("relay")
        # voice.speak("test")
        # voice.wait_for_speak()
        print(f"Testing : {devicename}")
        for _function in _function_list:
            _function()
            if PLATFORM == _OS.RASPBERRY: time.sleep(0.1)

    voice.speak("all devices healthy")
    voice.wait_for_speak()

    
    trigger_debounce = GUN_CONFIGURATION.trigger_debounce
    zoom_debounce = GUN_CONFIGURATION.zoom_debounce
    #torch_debounce = GUN_CONFIGURATION.torch_debounce.trigger_oneshot_simple


    cnt = 0 
    TEMP_DEBUG_trigger_cnt = 0
    TEMP_fake_light = False
    
    while True:
        imageIDs = []
        TEMP_DEBUG_trigger_cnt += 1
        #print(TEMP_DEBUG_trigger_cnt)
        with time_it("TOTAL TIME FOR EVERYTHING", debug=PRINT_DEBUG):
            cnt += 1
            with time_it("get next image", debug=PRINT_DEBUG), perfmonitor.measure("get next image"):
                cap_img = next(image_capture_longrange)
                cap_img_closerange = next(image_capture_shortrange)
                if "game_client" in locals():
                    # use these for uploading images of interest to the server
                    # IDs that don't match will be ignored, so just grab all valid ones for now
                    imageIDs.append(factory.decode_image_id(cap_img))
                    imageIDs.append(factory.decode_image_id(cap_img_closerange))
                # this is bad code - should come as package with the image -
                # but in easy of modularity have to do it like this for now
            with time_it("start analysis", debug=PRINT_DEBUG):
                for img_analyser in image_analysis:
                    img_analyser.trigger_analysis()
                if "game_client" in locals():
                    game_client.trigger_capture_close_range()
                    game_client.trigger_capture_long_range()

            GUN_CONFIGURATION.loop_wait()
            
            with time_it("gun states set", debug=PRINT_DEBUG), perfmonitor.measure("gun_states"):
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

                # here we check the torch debouncer and the input trigger
                # if we see that we have no energy we disable torch
                # update torch with latest energy
                players["testself"].torch_energy_update(is_torch_reqd) # this isnt quite right as needs to ask debouncer if can use torch
                #if players["me"].get_torch_energy() < 5:
                #    is_torch_reqd = False

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
                is_trigger_pressed = trigger_debounce.trigger_1shot_simple_High(is_trigger_reqd)
                if is_trigger_pressed is True:
                    players["testself"].update_ammo(-1)
                    #file_system.save_image(cap_img,message=f"quadro_longrange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                    #file_system.save_image(cap_img_closerange,message=f"quadro_closerange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                    #voice.speak("wut")
                    # true will only be available as an impulse after
                    # pulling trigger, then go low again - but
                    # mem state of debouncer will remain high


                    # debugging code to capture images
                    #if cap_img is not None:
                    TEMP_DEBUG_trigger_cnt += 1
                    # file_system.save_image(cap_img,message=f"_longrange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                    # file_system.save_image(cap_img_closerange,message=f"_closerange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                set_trigger(
                    state=trigger_debounce.get_heldstate(),
                    strobe_cnt=0
                    ) # click noise from relay only

            with time_it("gun image stuff", debug=PRINT_DEBUG), perfmonitor.measure("gun_image"):
                # output_image = display.emptyscreen
                transition_i = transform_manager.get_deltatime_transition()
                transition_state = transform_manager.get_transition_state()
                display_active_image = cap_img_closerange
                if  transition_state == img_processing.CameraTransitionState.CLOSERANGE:
                    display_active_image = cap_img_closerange
                    output_image = display.generate_output_affine(display_active_image)
                    transition_i=0
                elif transition_state == img_processing.CameraTransitionState.LONGRANGE:
                    display_active_image = cap_img
                    output_image = display.generate_output_affine(display_active_image)
                    transition_i=transform_manager.transformdetails.transition_steps-1
                else:
                    with time_it("execute affine transform", debug=PRINT_DEBUG):
                        mat = transform_manager.CR_all_transition_m[transition_i]
                        cr_img = img_processing.apply_perp_transform(mat, cap_img_closerange, display.emptyscreen)

                        mat = transform_manager.LR_all_transition_m[transition_i]
                        lr_img = img_processing.apply_perp_transform(mat, cap_img, display.emptyscreen)
                    with time_it("darken and overlay", debug=PRINT_DEBUG):
                        percent_done = transition_i/(transform_manager.transformdetails.transition_steps-1)
                        cr_img = img_processing.darken_image(cr_img, 1-percent_done)
                        combo_image = img_processing.overlay_warped_image_alpha_feathered(cr_img, lr_img, percent_done)
                        
                        #combo_image = img_processing.radial_motion_blur(combo_image)
                        output_image = img_processing.gray2rgb(combo_image)

                    
            
            with time_it("wait for image analysis", debug=PRINT_DEBUG), perfmonitor.measure("wait image_analysis"):
                analysis = {}
                for img_analyser in image_analysis:
                    if img_analyser.check_if_timed_out():
                        raise Exception(f"Analysis timed out for {img_analyser.OS_friendly_name} analysis")
                    # TODO: get this properly. Some complexity due to reversed shape so using
                    # protected member :(
                    res_for_affine_transform_lookup = img_analyser.camera_source_class_ref._store_res #BAD LIELL!!! 
                    try:
                        result: analyse_lumotag.AnalysisOutput = img_analyser.analysis_output_q.get(block=True, timeout=0)
                        if isinstance(result, Exception):
                            raise result# this is really shit but better than nothing or dying downstream in a confusing way
                        perfmonitor.manual_measure(f"{img_analyser.OS_friendly_name}", img_analyser.get_analysis_time_ms())

                        if result.Results:
                            
                            #file_system.save_barcodepair(result, message="falsepos")
                            #save_analysis(result)

                            if res_for_affine_transform_lookup not in analysis:
                                analysis[res_for_affine_transform_lookup] = []
                            analysis[res_for_affine_transform_lookup].extend(result.Results)
                    except queue.Empty:
                        # raise AnalysisTimeoutException("Timeout occurred while waiting for image analysis.")
                        # print(f"waiting for analysis {img_analyser.OS_friendly_name}")
                        pass # test asynchronous analysis

 

                #save_images_if_barcode(analysis,file_system,cap_img,cap_img_closerange)
            with perfmonitor.measure("graphics"):
                with time_it("add internal section", debug=PRINT_DEBUG):
                    display.add_internal_section_region(
                        display_active_image.shape,
                        output_image,
                        transform_manager.get_lerped_targetzone_slice(transition_i),
                        transform_manager.get_display_affine_transformation(transition_i))

            with time_it("add graphics: crosshair/analyics", debug=PRINT_DEBUG), perfmonitor.measure("graphics"):

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



                perfmonitor.manual_measure("check_scale", 25)

                # calculate the fade for test player - needs to be called to determine what direction of fade
                fade_norm = display.get_norm_fade_val(players["demoplayer"], analysis)




                
                with time_it("display image", debug=PRINT_DEBUG),  perfmonitor.measure("display"):
                    status_bar.load_player_image(players["demoplayer"].col_image, fade_norm)
                    status_bar.draw_status_bar(
                        output_image,
                        players["testself"].ammo,
                        players["testself"].get_normalised_torchenergy(),
                        players["testself"].get_healthpoints()
                        )
                    
                    # status_bar.draw_shieldtorch_bar(output_image, players["me"].get_normalised_torchenergy())
                    # original display output before new UI stuff (doom bar, graphic meters)
                    
                    image_actions = display.cardio_gram_display.update_metrics({i:perfmonitor.get_average(i) for i in perfmonitor.measurements.keys()})
                    output_image = display.cardio_gram_display.composite_onto_inplace(output_image, image_actions)
                    

                    # use the event comms and event sender to check connectivity
                    # both should have the same connection status, but we can check both for sanity


                    # check if local variables have been defined
                    if "game_client" in locals():
                        if not game_client.is_connected():
                            img_processing.draw_border_rectangle(output_image, thickness=10, color=(0, 0, 255))
                            players["testself"].set_healthpoints(None)
                        else:
                        # probably shoudl get the player card here
                            if game_client.get_latest_gamestate() is not None:
                                players["testself"].set_healthpoints(
                                    game_client.get_latest_gamestate().players.get("testself").health
                                    )


                    if is_trigger_pressed:
                        # screen flash on trigger - do we want this to hide the UI?
                        output_image[:] = 255
                    
                    display.display(output_image)
                perfmonitor.get_time("complete_cycle", reset=True)
                perfmonitor.manual_measure("check_scale2", 25)
                if len(analysis) > 0:
                    if is_trigger_pressed is True:
                        if "demoplayer" in players:
                            players["demoplayer"].update_healthpoints(diff=-10)


                if "game_client" in locals():
                    if is_trigger_pressed is True:
                        # upload all images during trigger event
                        for img_id in imageIDs:
                            game_client.upload_image_by_id(img_id)
                    else:
                        for img_id in imageIDs:
                            game_client.delete_image_by_id(img_id)


                # if is_trigger_pressed is True:
                #     file_system.save_image(cap_img,message=f"falsep_longrange_cnt{TEMP_DEBUG_trigger_cnt}cnt")
                #     file_system.save_image(cap_img_closerange,message=f"falsep_closerange_cnt{TEMP_DEBUG_trigger_cnt}cnt")

if __name__ == '__main__':
    main()
# dict_keys(['cam1inner_mser', 'cam1macro_mser', 'cam1inner', 'cam1macro', 'cam2inner', 'spacemaker', 'gun_image', 'complete_cycle', 'graphics'])