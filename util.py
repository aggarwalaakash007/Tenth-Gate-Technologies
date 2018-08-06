import socket
import _pickle as cPickle
import struct


def send_one_message(sock, data):
    data = cPickle.dumps(data)
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        #if not newbuf: return None
        if not newbuf: continue
        buf += newbuf
        count -= len(newbuf)
    return buf
    
def recv_one_message(sock):
    lengthbuf = recvall(sock, 4)
    length, = struct.unpack('!I', lengthbuf)
    return cPickle.loads(recvall(sock, length))