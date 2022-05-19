# yolov5-deepsort-tracking via TCP Trasmission

# Env and Versions

## Host OS(Mac)
- Mac OS 12.3.1
- python 3.6.10
- pytorch 1.10.2
- torchvision 0.11.3
- tkinter 8.6 

## guest OS(ubuntu)
- ubuntu 20.04
- python 3.8.1
- pytorch 1.10.2+cu102
- torchvision 0.11.3+cu102


## How to Start

```
# at host OS
$ python3 tkapp.py
# at guest OS
$ python3 yolov5_detect.py
```


# Stracture

# result
<b>Output movie<b>
  
![tcp_yolov5](https://user-images.githubusercontent.com/48679574/169310418-3506fcf4-b077-48a8-9e49-39322f4e2c6d.gif)
  
# References
- [Developing a Live Video Streaming Application using Socket Programming with Python](https://medium.com/nerd-for-tech/developing-a-live-video-streaming-application-using-socket-programming-with-python-6bc24e522f19)
