import os
import sys
import argparse
import numpy as np
import time
import cv2
import glob
import onnxruntime
import torch
import torch.nn.functional as F
import math

# prj_path = os.path.join(os.path.dirname(__file__), '..')
# if prj_path not in sys.path:
#     sys.path.append(prj_path)

from ortrack_nvinfer import ORTrackerNvinfer

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def get_frames(video_name):
    """获取视频帧

    Args:
        video_name (_type_): _description_

    Yields:
        _type_: _description_
    """
    if not video_name:
        rtsp = "rtsp://%s:%s@%s:554/cam/realmonitor?channel=1&subtype=1" % ("admin", "123456", "192.168.1.108")
        cap = cv2.VideoCapture(rtsp) if rtsp else cv2.VideoCapture()
        
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                # print('读取成功===>>>', frame.shape)
                yield cv2.resize(frame,(800, 600))
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame

class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        """初始化预处理图像. 
           需要注意的是: 如果按照如下方式处理则无法通过。原因是transpose之后的数据不连续，需要将其变为连续的数
           据，所以需要先进行ascontiguousarray
           '''
            img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
            img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
            return img_tensor_norm
           '''
            因此，等效的实现包含两种，一种是直接在返回的tensor上添加contiguous,另一种是在numpy.array数据的
            时候就进行ascontiguousarray.
        Args:
            img_arr (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        # 第一种方法
        # im = np.ascontiguousarray(img_arr.transpose((2, 0, 1))[::-1]) # HWC to CHW, BGR to RGB
        # im = torch.from_numpy(im).cuda().float()
        # im = ((im / 255.0) - self.mean) / self.std
        # if len(im.shape) == 3:
        #     im = im[None]
            
        # print(f"preprocessor im: {im.shape}")
        # return im
        
        # 第二种方法 
        # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm.contiguous()

class MFTrackerTRT:
    def __init__(self) -> None:
        self.debug = True        
        
        self.init_track_net()
        self.preprocessor = Preprocessor_wo_mask()
        self.max_score_decay = 1.0
        self.search_factor = 4.5
        self.search_size = 256
        self.template_factor = 2.0
        self.template_size = 128
        self.update_interval = 200
        self.online_size = 1

    def init_track_net(self):
        """使用设置的参数初始化tracker网络
        """        
        self.ortrack_tracker = ORTrackerNvinfer("./model_ortrack_distill_sim.trt")

    def track_init(self, frame, target_pos=None, target_sz = None):
        """使用第一帧进行初始化

        Args:
            frame (_type_): _description_
            target_pos (_type_, optional): _description_. Defaults to None.
            target_sz (_type_, optional): _description_. Defaults to None.
        """
        self.trace_list = []
        try:
            # [x, y, w, h]
            init_state = [target_pos[0], target_pos[1], target_sz[0], target_sz[1]]
            z_patch_arr, _, z_amask_arr = self.sample_target(frame, init_state, self.template_factor, output_sz=self.template_size)
            template = self.preprocessor.process(z_patch_arr)
            self.template = template
            self.online_template = template

            self.online_state = init_state
            self.online_image = frame
            self.max_pred_score = -1.0
            self.online_max_template = template
            self.online_forget_id = 0

            # save states
            self.state = init_state
            self.frame_id = 0
            print(f"第一帧初始化完毕！")
        except:
            print(f"第一帧初始化异常！")
            exit()

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = self.sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        # print(f">>>search: {search.shape}")
        # compute trt output prediction
        trt_outputs = self.ortrack_tracker.infer(self.template, search)
        # print(f">>> lenght trt_outputs: {trt_outputs}")
        pred_boxes = trt_outputs[2]
        pred_score = trt_outputs[1]

        # print(f">>> trt_outputs: {pred_boxes, pred_score}")
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        # print("pred_box", pred_box)
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
 
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)

        return {"target_bbox": self.state, "conf_score": pred_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box[0]
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def sample_target(self, im, target_bb, search_area_factor, output_sz=None, mask=None):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
        x2 = int(x1 + crop_sz)

        y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
        y2 = int(y1 + crop_sz)

        x1_pad = int(max(0, -x1))
        x2_pad = int(max(x2 - im.shape[1] + 1, 0))

        y1_pad = int(max(0, -y1))
        y2_pad = int(max(y2 - im.shape[0] + 1, 0))

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        if mask is not None:
            mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H,W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0
        if mask is not None:
            mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)


        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
            att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
            if mask is None:
                return im_crop_padded, resize_factor, att_mask
            mask_crop_padded = \
            F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded

        else:
            if mask is None:
                return im_crop_padded, att_mask.astype(np.bool_), 1.0
            return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded
        
    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W-margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H-margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2-x1)
        h = max(margin, y2-y1)
        return [x1, y1, w, h]
        
if __name__ == '__main__':
 
    Tracker = MFTrackerTRT()
    Tracker.video_name = "./untitled.mp4"
    # init_state = [282, 250, 23, 23]

    # warm_up = 500
    # warm_up_first = True
    # input0= torch.rand((112, 112, 3)).numpy()
    # input1= torch.rand((224, 224, 3)).numpy()
    # for i in range(warm_up):
    #     if warm_up_first == True:
    #         Tracker.track_init(input0, [20, 20], [50, 50])
    #         warm_up_first = False
    #     else:
    #         state = Tracker.track(input1)

    first_frame = True
    if Tracker.video_name:
        video_name = Tracker.video_name
    else:
        video_name = 'webcam'

    frame_id = 0
    total_time = 0
    for frame in get_frames(Tracker.video_name):
        # print(f"frame shape {frame.shape} {type(frame)}")
        tic = cv2.getTickCount()
        if first_frame:
            x, y, w, h = cv2.selectROI(video_name, frame, fromCenter=False)
            print(f">>>init state: {(x, y, w, h)}")
            target_pos = [x, y]
            target_sz = [w, h]
            # target_pos = [init_state[0], init_state[1]]
            # target_sz = [init_state[2], init_state[3]]
            Tracker.track_init(frame, target_pos, target_sz)
            first_frame = False
        else:
            state = Tracker.track(frame)
            frame_id += 1

        toc = cv2.getTickCount() - tic
        toc = int(1 / (toc / cv2.getTickFrequency()))
        total_time += toc
        print('Video: {:12s} {:3.1f}fps'.format('tracking', toc))
        cv2.imshow('Tracking', frame)
        cv2.waitKey(1)
    
    print('video: average {:12s} {:3.1f} fps'.format('finale average tracking fps', total_time/(frame_id - 1)))
    cv2.destroyAllWindows()
