import cv2
import pickle
import struct


def create_bts_socket_data(image):
    result, encode = cv2.imencode('.jpg', image, (cv2.IMWRITE_JPEG_QUALITY, 10))
    data = pickle.dumps(encode, 0)
    size = len(data)
    return data, size

def tcp_sender(socket, size, data, pack_format=">L"):
    socket.sendall(struct.pack(pack_format, size) + data)

def unpack_data(data, payload_size, format=">L"):
    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(format, packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    return data, msg_size


def bts2img(frame_data):
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return frame

def struct_calcsize(format=">L"):
    return struct.calcsize(format)