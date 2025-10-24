import numpy as np
from libs.collections import Scambi_unit_LED_only
from libs.utils import time_it_sparse
from factory import TimeDiffObject
from typing import Literal
import multiprocessing
import socket
import atexit
from abc import ABC, abstractmethod

UDP_DELIMITER:bytes = b'\xAB\xCD\xEF' # be careful changing this - can mess up delimiting if for instance | or null


class UDPMessageReceiver:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        buffer_size = 1024
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        self.socket.bind((self.host, self.port))
        atexit.register(self.socket.close)

    def receive_message(self, buffer_size=10000):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            return data.decode(), addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None

    def receive_bytes_message(self, buffer_size=10000):
        try:
            data, addr = self.socket.recvfrom(buffer_size)
            return data, addr
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None, None


class UDPListenerProcessWrapper:
    def __init__(self):
        self.queue = multiprocessing.Queue(maxsize=1)
        self.process = multiprocessing.Process(target=self.worker_process, args=(self.queue,))
        # Set daemon to True so that the process will be terminated when the main thread exits
        self.process.daemon = True
        self.process.start()

    def get_message(self):
        return self.queue.get(block=True, timeout=None)

    def worker_process(self, _queue):
        """we want to pull UDP messages off the buffer as fast as 
        possible so it doesn't fill up. We can also set a low buffer
        for the receiver"""
        receiver = UDPMessageReceiver()
        while True:
            message, _ = receiver.receive_bytes_message()
            if not _queue.full():
                _queue.put(message)


class UDPTransmitProcess(ABC):
    def __init__(self, host, port):
        self.host=host
        self.port=port
    @abstractmethod
    def send_scambis(self, scambis:list[Scambi_unit_LED_only]):
        ...


class UDPTrasmit_RUSTsync(UDPTransmitProcess):

    def __init__(self, *args, **kwargs):
        """rust implementation to encode the scambis and send them
        not asynchronous"""
        super().__init__(*args, **kwargs)
        import led_sender # this is a pyo3 rust module - import it here rather than top level so its easier to transplant
        self.sender = led_sender.UdpSender()

    def send_scambis(self, scambis: list[Scambi_unit_LED_only]):
        if scambis is not None and len(scambis)>0:
            self.sender.send_udp_scambis(
                scambis,
                f"{self.host}:{self.port}"
            )


class UDPTransmitProcessWrapper(UDPTransmitProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = multiprocessing.Queue(maxsize=1)
        self.process = multiprocessing.Process(target=self.worker_process, args=(self.queue,))
        # Set daemon to True so that the process will be terminated when the main thread exits
        self.process.daemon = True
        self.process.start()

    def send_scambis(self, scambis: list[Scambi_unit_LED_only]):
        return self.queue.put(scambis)

    def worker_process(self, _queue):
        """we want to pull UDP messages off the buffer as fast as 
        possible so it doesn't fill up. We can also set a low buffer
        for the receiver"""
        transmitter = UDPMessageSender(host=self.host, port=self.port)
        while True:
            scambis: list[Scambi_unit_LED_only] = _queue.get(block=True, timeout=None)
            leds_to_send = transform_scambits_for_UDP(scambis)
            transmitter.send_message(leds_to_send)


class UDPMessageSender:
    def __init__(self, host='scambilightled.broadband', port=12345):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.error_time = TimeDiffObject()
        self.error_backoff_s = 0

    def send_message(self, message:bytes):
        try:
            if self.error_time.get_dt() > self.error_backoff_s:
                self.socket.sendto(message, (self.host, self.port))
        except Exception as e:
            print(f"Error sending message: {e}")
            self.error_time.reset()

    def close(self):
        self.socket.close()

def  transform_scambits_for_UDP(scambis: list[Scambi_unit_LED_only])->bytes:
    """pack data for efficient delivery across network"""

    output_payload = []
    #with time_it_sparse("prep scambis for sending"):
    for scambiunit in scambis:
        pos_array = np.asarray(scambiunit.physical_led_pos, dtype="uint16")
        col_array = np.asarray(tuple(scambiunit.colour), dtype="uint8")
        output_payload.append(pos_array.tobytes())
        output_payload.append(col_array.tobytes())

    return UDP_DELIMITER.join(output_payload)


def transform_UDP_message_to_scambis(message: bytes)->list[Scambi_unit_LED_only]:
    """transform received UDP message to scambi LED information"""
    data = message.split(UDP_DELIMITER)

    scambiunits: list[Scambi_unit_LED_only] = []
    with time_it_sparse("decode remote scambis"):
        for i in range(0, len(data), 2):
            scambiunits.append(Scambi_unit_LED_only(
                colour=tuple(np.frombuffer(data[i+1], dtype="uint8")),
                physical_led_pos=[int(x) for x in np.frombuffer(data[i], dtype="uint16")]))
    return scambiunits