import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = b"Hello, World!"

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))



sock2 = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock2.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock2.recvfrom(1024) # buffer size is 1024 bytes
    print("received message: %s" % data)