event_good_session = {'body': '{"action":"","data":"doesnt matter for now but needs config id and maybe user","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_bad_session = {'body': '{"action":"","data":"doesnt matter for now but needs config id and maybe user","sessiontoken":"\\"fb1e6ssssb1es\\""}'}
event_good_login = {'body': '{"login":{"email":"test@email.tet","password":"secretshh"}}', 'isBase64Encoded': False}
event_bad_login_bad_password = {'body': '{"login":{"email":"test@email.tet","password":"wrongpassword"}}', 'isBase64Encoded': False}
event_bad_login_no_user = {'body': '{"login":{"email":"sss@sss.tet","password":"wrongpassword"}}', 'isBase64Encoded': False}
event_newuser_exists = {'body': '{"action":"newuser","data":"already@exists","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_newuser = {'body': '{"action":"newuser","data":"someone@new","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_get_config = {'body': '{"action":"getconfig","data":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

event_upload_image_perpwarp = {'body': '{"action":"perpwarp","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_upload_image_image_raw = {'body': '{"action":"image_raw","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_upload_image_image_overlay = {'body': '{"action":"image_overlay","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

event_upload_upload_image_perpwarp = {'body': '{"action":"getimage_perpwarp","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_upload_upload_image_raw = {'body': '{"action":"getimage_raw","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_upload_upload_image_overlay = {'body': '{"action":"getimage_overlay","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

event_get_event = {'body': '{"action":"check_event","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

event_set_event = {'body': '{"action":"reset","payload":"existinguser@wantsconfig","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

'[{"clickX": 176, "clickY": 116}, {"clickX": 176, "clickY": 116}, {"clickX": 176, "clickY": 116}]'
event_update_config_samples = {'body': '{"action":"sendposfinish","data":1,"sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

body_bad_pos_send = {'action': 'sendposfinish', 'data': [{'clickX': 243, 'clickY': 195}], 'sessiontoken': '"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es"'}

event_test_logged_in = {'body': '{"action":"check_logged_in","data":1,"sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

event_test_get_sim = {'body': '{"action":"get_region_sim","data":1,"sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}

