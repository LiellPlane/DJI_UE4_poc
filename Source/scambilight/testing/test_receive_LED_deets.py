import socket
import atexit
import json
import numpy as np
import struct


class UDPMessageReceiver:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        atexit.register(self.close)

    def receive_message_json(self, buffer_size=1024):
        try:
            data, addr = self.socket.recvfrom(buffer_size)

            return data.decode(), addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def receive_message_bytes(self, buffer_size=1024):
        try:
            data, addr = self.socket.recvfrom(buffer_size)

            data = data.split(b'|')

            posyes = [np.array(struct.unpack('{}H'.format(len(i)//2), i), dtype=np.uint16) for i in data[::2]]
            colours = [np.array(struct.unpack('{}B'.format(len(i)), i), dtype=np.uint8) for i in data[1::2]]

            positions = data[::2] 
            colour = data[1::2]



            return data, addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def close(self):
        self.socket.close()

if __name__ == "__main__":

    receiver = UDPMessageReceiver()
    while True:
    # Receive a message
        message, address = receiver.receive_message_bytes()
#
    receiver.close()