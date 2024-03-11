event_good_session = {'body': '{"action":"","data":"doesnt matter for now but needs config id and maybe user","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_bad_session = {'body': '{"action":"","data":"doesnt matter for now but needs config id and maybe user","sessiontoken":"\\"fb1e6ssssb1es\\""}'}
event_good_login = {'body': '{"login":{"email":"test@email.tet","password":"secretshh"}}', 'isBase64Encoded': False}
event_bad_login_bad_password = {'body': '{"login":{"email":"test@email.tet","password":"wrongpassword"}}', 'isBase64Encoded': False}
event_bad_login_no_user = {'body': '{"login":{"email":"sss@sss.tet","password":"wrongpassword"}}', 'isBase64Encoded': False}
event_newuser_exists = {'body': '{"action":"newuser","data":"already@exists","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}
event_newuser = {'body': '{"action":"newuser","data":"someone@new","sessiontoken":"\\"fb1e6ead-e6b5-4dea-8921-f60e4c40b1es\\""}'}