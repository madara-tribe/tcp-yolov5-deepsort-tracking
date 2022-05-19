import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import socket
import argparse

from tcp_utils.tcp_img_sender import unpack_data, bts2img, struct_calcsize
M_SIZE = 1024

class Application(tk.Frame):
    def __init__(self,master, host=None, port=None, video_source=0):
        super().__init__(master)

        self.master.geometry("1280x768")
        self.master.title("Tkinter with Video Streaming and Capture")
        self.font_frame = font.Font(family="Meiryo UI", size=15, weight="normal" )
    
        self.vcap = cv2.VideoCapture( video_source )
        self.width = self.vcap.get( cv2.CAP_PROP_FRAME_WIDTH )
        self.height = self.vcap.get( cv2.CAP_PROP_FRAME_HEIGHT )

        self.create_widgets()
        self.create_frame_button()
        self.create_socket(host, port)
        self.delay = 15 #[ms]

    def create_socket(self, host, port):
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        print('Socket created')

        self.sock.bind((host, port))
        print('Socket bind complete')
        self.sock.listen(10)
        print('Socket now listening')

    def create_widgets(self):
        #Frame_Camera
        self.frame_cam = tk.LabelFrame(self.master, text = 'Camera', font=self.font_frame)
        self.frame_cam.place(x = 10, y = 10)
        self.frame_cam.configure(width = self.width+30, height = self.height+50)
        self.frame_cam.grid_propagate(0)

        #Canvas
        self.canvas = tk.Canvas(self.frame_cam)
        self.canvas.configure(width=self.width, height=self.height-150)
        self.canvas.grid(column=0, row=0, padx = 10, pady=10)

    def create_frame_button(self):
        # Button Frame
        self.frame_btn = tk.LabelFrame(self.master, text='Control', font=self.font_frame)
        self.frame_btn.place(x=10, y=650)
        self.frame_btn.configure(width=100, height=120 )
        self.frame_btn.grid_propagate(0)

        # Close
        self.btn_close = tk.Button(self.frame_btn, text='Close', font=self.font_frame, command=self.close)
        self.btn_close.pack(fill = 'x', padx=20, side = 'left')
        
        # Start
        self.btn_close = tk.Button(self.frame_btn, text='Start', font=self.font_frame, command=self.start_tcp)
        self.btn_close.pack(fill = 'x', padx=20, side = 'left')

    def start_tcp(self):
        conn, addr=self.sock.accept()
        data = b""
        payload_size = struct_calcsize(format=">L")
        print("payload_size: {}".format(payload_size))
        while True:
            while len(data) < payload_size:
                print("Recv: {}".format(len(data)))
                data += conn.recv(4096)

            data, msg_size = unpack_data(data, payload_size, format=">L")
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = cv2.cvtColor(bts2img(frame_data), cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))

            #self.photo -> Canvas
            self.canvas.create_image(0,0, image= self.photo, anchor = tk.NW)
            self.canvas.update()

    def close(self):
        self.master.destroy()
        self.vcap.release()
        self.canvas.delete("o")


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root, host="192.168.10.107", port=8485)#Inherit
    app.mainloop()

    