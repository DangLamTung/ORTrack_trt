# ORTrack onnx c++ and TensorRT-py version
ORTrack:  Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking. [official pytorch](https://github.com/wuyou3474/ORTrack)

Here, the ORTrack tracking algorithm with onnx and TensorRT convert code is provided

Archived ~430FPS on GTX1650 and ~100FPS on Jetson Xavier NX (distilled model)


# 1. How to build and run it?
For ONNX, install the onnxruntime, change the path in ORTrack_pyonnx/or_tracker_ort.py and run.


# 2. MixformerV2 TensorRT version inference with CPP and python
Assume that you have configured Tensorrt, use onnx2trt to convert the onnx model to engine on your GPU platform, and then start compilation and execution.To create the TensorRT model, run the convert_model_trt.sh file to create TensorRT model, then build the program:

## cpp version 
build and run
```
$ cd ORTrack_trt
$ mkdir build && cd build
$ cmake .. && make
& ./ortracker-trt ../ortracker.trt  ../target.mp4
```
## python version
Modify the video path in ORTrack_trt/mixformer-pytrt/mf_tracker_trt.py，and mkdir model file_dir, then download the onnx file and put onnx file into file_dir.
```
$ cd ORTrack_pyonnx
$ python ORTrack_pytrt/mf_tracker_trt.py
```
Note: In addition to simplification when converting the onnx model, it is important to ensure that the shape of the data input to the engine model and the corresponding underlying data are continuous.

# Acknowledgments

Thanks for the [LightTrack-ncnn](https://github.com/Z-Xiong/LightTrack-ncnn.git), [lite.ai.tookit](https://github.com/DefTruth/lite.ai.toolkit), and [MixformerV2-onnx](https://github.com/maliangzhibi/MixformerV2-onnx) which helps us to quickly implement our ideas, and [official pytorch](https://github.com/wuyou3474/ORTrack) for this work.

