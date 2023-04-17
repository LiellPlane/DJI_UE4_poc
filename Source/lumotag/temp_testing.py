import rabbit_mq
import factory
import messaging
import time

def main():
    # messagers are being popualted from external source and put on 
    #queue, next is to test error if queue is full (should be ok)
    # then test being in a loop and extracting hit reports.. 
    #then add SEND function which we currently don't have

    #then need to flesh out hit reports, images and IDs etc, and how to handle
    #or display any errors
    mssger = rabbit_mq.messenger(
        factory.TZAR_config())

    cnt = 0
    while True:
        cnt += 1
        time.sleep(0.1)
        print("-> checking in box")
        print(mssger.check_in_box())
        print("-> end of check")

        print("sending msg")
        mssger.send_message(f"fartzz uwu {cnt}")

if __name__ == '__main__':
    main()
