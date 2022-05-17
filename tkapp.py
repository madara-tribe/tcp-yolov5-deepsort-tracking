import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import font
import time
from multiprocessing import Queue
from tcp_utils.byte2image import bts_to_img


class Application(tk.Frame):
    def __init__(self,master, q:Queue, video_source=0):
        super().__init__(master)

        self.master.geometry("1280x768")
        self.master.title("Tkinter with Video Streaming and Capture")
        self.font_frame = font.Font(family="Meiryo UI", size=15, weight="normal" )

        self.q = q
    
        self.vcap = cv2.VideoCapture( video_source )
        self.width = self.vcap.get( cv2.CAP_PROP_FRAME_WIDTH )
        self.height = self.vcap.get( cv2.CAP_PROP_FRAME_HEIGHT )

        self.create_widgets()
        self.create_frame_button()
        self.delay = 15 #[ms]
        self.update()


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
        self.btn_start = tk.Button(self.frame_btn, text='Start', font=self.font_frame, command=self.start)
        self.btn_start.pack(fill = 'x', padx=20, side = 'left')
        # Stop
        self.stop_btn = tk.Button(self.frame_btn, text='Stop', font=self.font_frame, command=self.stop)
        self.stop_btn.pack(fill = 'x', padx=20, side = 'left')


    def update(self):
        #Get a frame from the video source
        bts_frame = self.q.get() #cv2.imread(self.yolov5_img_path)
        frame = cv2.cvtColor(bts_to_img(bts_frame), cv2.COLOR_BGR2RGB)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))

        #self.photo -> Canvas
        self.canvas.create_image(0,0, image= self.photo, anchor = tk.NW)
        self.master.after(self.delay, self.update)

    def close(self):
        self.master.destroy()
        self.vcap.release()
        self.canvas.delete("o")

    def start(self):
        print("let's start")

    def stop(self):
        print("let's stop")

#if __name__ == "__main__":
 #   root = tk.Tk()
   # app = Application(master=root)#Inherit
   # app.mainloop()
