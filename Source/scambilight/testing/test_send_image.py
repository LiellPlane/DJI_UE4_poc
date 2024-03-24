import os
from pathlib import Path
import imp
from multiprocessing import Process, Queue
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
#sys.path.append(str(Path(parent).resolve().parents[0]) + "\\lumotag\\factory.py")
module_name = imp.load_source('my_collections', str(Path(parent).resolve().parents[0]) + "/lumotag/my_collections.py")
module_name = imp.load_source('img_processing', str(Path(parent).resolve().parents[0]) + "/lumotag/img_processing.py")
module_name = imp.load_source('factory', str(Path(parent).resolve().parents[0]) + "/lumotag/factory.py")



import socket
import factory




class ExternalDataWorker():
    def __init__(
            self,
            hostname):
        self.in_queue = Queue(maxsize=1)
        self.msg_queue = Queue(maxsize=1)
        self.hostname = hostname
    def _start(self):
    
        process = Process(
            target=self._run,
            args=(),
            daemon=True)

        process.start()


class UDPMessageSender:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_message(self, message):
        try:
            self.socket.sendto(message.encode(), (self.host, self.port))
            print(f"Sent message: {message}")
        except Exception as e:
            print(f"Error sending message: {e}")

    def close(self):
        self.socket.close()


class UDPMessageReceiver:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))

    def receive_message(self, buffer_size=1024):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            print(f"Received message: {data.decode()} from {addr}")
            return data.decode(), addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def close(self):
        self.socket.close()

if __name__ == "__main__":
    # Create an instance of UDPMessageSender
    sender = UDPMessageSender()

    # Define the message to send

    

    message = "This is a test message!"
    receiver = UDPMessageReceiver()

    # Receive a message
    message, address = receiver.receive_message()
    # Send the message
    sender.send_message(message)

    # Close the socket
    sender.close()
    receiver.close()