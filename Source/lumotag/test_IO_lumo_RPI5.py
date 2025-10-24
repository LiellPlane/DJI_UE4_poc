# need to add switch for reversed camera in shared memory
# its a bad technique but will have to make do

import factory
from functools import partial
from utils import time_it, get_platform

#  detect what OS we are on - test environment (Windows) or production (pi hardware)
PLATFORM = get_platform()


print("raspberry presence detected, loading hardware libraries")
import raspberry5_hardware as lumogun


# load config depending on if simulated, or if on hardware,
# model ID from file on device
model = lumogun.get_my_info(factory.gun_config.DETAILS_FILE)
GUN_CONFIGURATION  = factory.get_config(model)


def main():

    triggers = lumogun.Triggers(GUN_CONFIGURATION)

    relay = lumogun.Relay(GUN_CONFIGURATION)

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
    torch_debounce = GUN_CONFIGURATION.torch_debounce.trigger_oneshot_simple

    while True:
        GUN_CONFIGURATION.loop_wait()

        with time_it("gun states set"):
            #accelerometer.update_vel()
            results_trig_positions = (triggers.test_states())

            is_torch_reqd = results_trig_positions[GUN_CONFIGURATION.button_torch]
            is_trigger_reqd = results_trig_positions[GUN_CONFIGURATION.button_trigger]
            is_rear_reqd = results_trig_positions[GUN_CONFIGURATION.button_rear]

            # in this case 
            # result = torch_debounce(is_torch_reqd)
            # if result is True:
            #     set_torch(state=is_torch_reqd, strobe_cnt=GUN_CONFIGURATION.light_strobe_cnt)
            #     set_laser(state=is_torch_reqd, strobe_cnt=0)
            set_torch(state=is_torch_reqd, strobe_cnt=0)

            set_laser(state=is_torch_reqd, strobe_cnt=0)

            # desired behaviour: 
            # User presses trigger - gun fires immediately
            # after 0.N seconds, relay clicks off
            # user can now fire again immediately
            # any other behaviour during refractory period is ignored
            result = trigger_debounce.trigger_1shot_simple_High(is_trigger_reqd)
            if result is True:
                print("trigger pulled")
            # trigger
            set_trigger(
                state=trigger_debounce.get_heldstate(),
                strobe_cnt=0) # click noise from relay only




if __name__ == '__main__':
    main()
