import tensorrt as trt
from collections import OrderedDict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import logging
import logging.config
import time
import pkg_resources as pkg


TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
 
class ORTrackerNvinfer:
    """ORTracker Nvinfer
    """
    def __init__(self, engine_name="ORTracker") -> None:

      
        self.device = torch.device('cuda:0')
        
        
        self.engine_path = os.path.join( engine_name  )
        if not os.path.exists(self.engine_path):
            LOGGER.info(f"Error ENGINE_NAME: {engine_name}")
            sys.exit(1)
        
        # LOGGER.info(f"loading {self.engine_path} for TensorRT inference.")
        print(f"loading {self.engine_path} for TensorRT inference.")
        self.check_version(trt.__version__, '7.0.0', hard=True)

        # 定义绑定数据
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

        # 定义logger
        self.logger = trt.Logger(trt.Logger.INFO)

        # 反序列化engine文件
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
 
        # 创建上下文
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
 
        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            # LOGGER.info(f"name: {name, dtype}")
            print(f"name: {name, dtype}")
            # input
            if self.model.binding_is_input(i):
                if -1 in tuple(self.model.get_binding_shape(i)):
                    dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0,1)[2]))
                if dtype == np.float16:
                    fp16 = True
            else: # output
                # print(f">>>output name: {name} {self.model.get_binding_shape(i)} {dtype}")
                self.output_names.append(name)

            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype)).to(self.device)

            # 绑定输入输出数据
            self.bindings[name] = self.Binding(name, dtype, shape, im, int(im.data_ptr()))
 
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['search'].shape[0]

    def infer(self, im, im_0):
        """ 

        Args:
            im (_type_): _description_
            augment (bool, optional): _description_. Defaults to False.
            visualize (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
 
        self.binding_addrs['template'] = int(im.data_ptr())
        self.binding_addrs['search'] = int(im_0.data_ptr())
 
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
        
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


    def check_version(self, current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
        # Check version vs. required version
        current, minimum = (pkg.parse_version(x) for x in (current, minimum))
        result = (current == minimum) if pinned else (current >= minimum)  # bool
        s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
        if hard:
            assert result, s  # assert min requirements met
        if verbose and not result:
            LOGGER.warning(s)
        return result

 

if __name__=="__main__":
    det = MixformerNvinfer("./model_ortrack_distill_sim.trt" )

    input= torch.rand((1, 3, 128, 128)).cuda()
    # input0= torch.rand((1, 3, 112, 112)).cuda()
    input1= torch.rand((1, 3, 256, 256)).cuda()

    warmup_N = 100
    N = 1000
    for i in range(warmup_N):
        output = det.infer(input,  input1)
    
    start = time.time()
    for i in range(N):
        start_i = time.time()
        output = det.infer(input,  input1)
        # print(f">>>single infer time: {1 / (time.time() - start_i)} FPS")

    print(f">>>infer time: {1 / ((time.time() - start) / N)} FPS")

    print(f"output's length is {output[0].shape} {output[1].shape}")