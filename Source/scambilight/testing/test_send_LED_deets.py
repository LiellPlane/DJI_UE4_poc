
import socket
import json
import random
import sys

class UDPMessageSender:
    def __init__(self, host='scambilightled.broadband', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_message(self, message):
        try:
            self.socket.sendto(message.encode(), (self.host, self.port))
        except Exception as e:
            print(f"Error sending message: {e}")

    def close(self):
        self.socket.close()


if __name__ == "__main__":
    # Create an instance of UDPMessageSender
    sender = UDPMessageSender()

    senddic_list = []



    single_Led_size_bytes = sys.getsizeof(json.dumps({300:(255,255,255)}))
    udp_payload_bytes = 700
    leds_per_packet = udp_payload_bytes // single_Led_size_bytes


    from time import perf_counter
    # Send the message
    while True:
        senddic_list = []
        for i in range(0, 300):
            if i%leds_per_packet == 0:
                senddic_list.append({})
            senddic_list[-1][i]= (
                random.randint(0,200),
                random.randint(0,200),
                random.randint(0,200)
            )
        t1_start = perf_counter() 
        #for led_packt in senddic_list:
        sender.send_message(json.dumps(senddic_list))
        t1_stop = perf_counter()
        print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)
        pause = input("press enter to blast UDP")
    # Close the socket
    sender.close()
