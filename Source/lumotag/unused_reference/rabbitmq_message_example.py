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









                                            msgs.package_send_report(
                        type_=msgs.MessageTypes.HIT_REPORT.value,
                        image=image_capture_longrange.last_img,
                        gun_config=GUN_CONFIGURATION,
                        messenger=messenger,
                        target="some twat",
                        message_str="lol QQ l2p"
                    )