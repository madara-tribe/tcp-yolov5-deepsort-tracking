import sys, os
import cv2
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import Queue

from models.experimental import attempt_load
from option_parser import get_parser
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from Deepsort.parser import get_config
from Deepsort.deep_sort.deep_sort import DeepSort
from tcp_utils.byte2image import image_to_bts


class Yolov5_Detction:
    def __init__(self):
        self.cfg = get_config()
        self.vid_writer = self.cv2_video_writer(w=1280, h=720)
        self.device = select_device(device='')

    def preprocess_img(self, img, half):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def load_model(self, weights, half):
        self.cfg.merge_from_file('Deepsort/configs/deep_sort.yaml')
        deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                            max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        yolov5 = attempt_load(weights, map_location=self.device)  # load FP32 model
        if half:
            yolov5.half()  # to FP16
        return yolov5, deepsort


    def cv2_video_writer(self, w=1280, h=720):
        # camera init
        fps = 10 # int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vid_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
        return vid_writer

    def deepsort_detection(self, deepsort, annotator, det, img, im0, names, s):
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            # pass detections to deepsort
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            
            # draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()
        return deepsort, annotator, s

    @torch.no_grad()
    def yolov5_detection(self, q:Queue, opt):
        half, weights, video_source, imgsz, conf_thres = opt.half, opt.weights, opt.source, opt.imgsz, opt.conf_thres
        iou_thres, classes, agnostic_nms, augment = opt.iou_thres, opt.classes, opt.agnostic_nms, opt.augment

        # Load model
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        yolov5, deepsort = self.load_model(weights, half=half)
        stride = int(yolov5.stride.max())  # model stride
        names = yolov5.module.names if hasattr(yolov5, 'module') else yolov5.names  # get class names
        
        # Dataloader
        dataset = LoadImages(video_source, img_size=imgsz, stride=stride)

        # Run inference
        if self.device.type != 'cpu':
            yolov5(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(yolov5.parameters())))  # run once

        #t0 = time.time()
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            # Inference
            #t1 = time_sync()
            img = self.preprocess_img(img, half=half)
            pred = yolov5(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            #t2 = time_sync()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s
                s += '%gx%g ' % img.shape[2:]  # print string
                annotator = Annotator(im0, line_width=2, pil=not ascii)
                deepsort, annotator, s = self.deepsort_detection(deepsort, annotator, det, img, im0, names, s)
                # Print time (inference + NMS)
                #print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                im0 = annotator.result()
                # show on tkinter
                im_rgb = cv2.resize(im0, (1280, 720))
                self.vid_writer.write(im_rgb.astype(np.uint8))
                #im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
                q.put(image_to_bts(im_rgb))

                # opencv 
                #cv2.imshow(p, im0)
                #if cv2.waitKey(1) == ord('q'):  # q to quit
                    #raise StopIteration
                    
        print('Done. (%.3fs)' % (time.time() - t0))
        self.vid_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    q = Queue()
    opt = get_parser()
    yolov5_detect = Yolov5_Detction()
    yolov5_detect.yolov5_detection(q, opt)
    
