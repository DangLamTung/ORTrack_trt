//
// Create by Daniel Lee on 2023/9/22
// Modified by DLT

#ifndef ORTrackTRT_H
#define ORTrackTRT_H

#include <iostream>
#include <fstream>
#include <vector> 
#include <map>
#include <memory>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <half.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "cuda_fp16.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

# define USE_CUDA

using namespace nvinfer1;

struct DrBBox {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct DrOBB {
    DrBBox box;
    float score;
    int class_id;
};

struct Config {
    // std::vector<float> window;
    float template_factor = 2.0;
    float search_factor = 4.5; // 5.0
    int template_size = 128; //192
    int search_size = 256; // 384
    int stride = 16;
    int feat_sz = 14; // 24
    int update_interval = 200;
};
enum
{
    CHW = 0, HWC = 1
};


class ORTrackTRT {

private:
    Logger gLogger;

    const char* INPUT_BLOB_IMGT_NAME = "template";
    const char* INPUT_BLOB_IMGSEARCH_NAME = "search";

    const char* OUTPUT_BLOB_PREDBOXES_NAME = "pred_boxes"; 
    const char* OUTPUT_BLOB_PREDSCORES_NAME = "score_map"; 
    const char* OUTPUT_BLOB_BACKBONE_FEAT_NAME = "backbone_feat"; 
    const char* OUTPUT_BLOB_OFFSET_NAME = "offset_map"; 
    const char* OUTPUT_BLOB_SIZEMAP_NAME = "size_map"; 
    char *trt_model_stream = nullptr;
    
    size_t size{0};

    // define the TensorRT runtime, engine, context,stream
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;

    cudaStream_t stream;

    // init dynamic input dims
    std::vector<std::vector<int64_t>> input_node_dims = {
        {1, 3, 128, 128}, // z  (b=1,c,h,w)
        {1, 3, 256, 256} // x
    };

    // Define FP32 mean and scale values
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};  // RGB
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};

    float max_pred_score = -1.f;
    float max_score_decay = 1.f;

    cv::Mat z_patch; // template
 
    DrOBB object_box;
    int frame_id = 0;

    int output_pred_boxes_size = 1;
    int output_pred_scores_size = 1;


    float *input_imt = nullptr;
    float *input_imot = nullptr;
    float *input_imsearch = nullptr;
    float *output_pred_boxes = nullptr;
    float *output_pred_scores = nullptr;

    // 数据尺度的定义
    static const int INPUT_IMT_W = 128;
    static const int INPUT_IMSEARCH_W = 256;

    static const int INPUT_IMT_H = 128;
    static const int INPUT_IMSEARCH_H = 256;

private:

    void transform(cv::Mat &mat_z, cv::Mat &mat_x);

    void map_box_back(DrBBox &pred_box, float resize_factor);

    void clip_box(DrBBox &box, int height, int wight, int margin);

    void cal_bbox(float *boxes_ptr, float * scores_ptr, DrBBox &pred_box, float &max_score, float resize_factor);

    void sample_target(const cv::Mat &im, cv::Mat &croped, DrBBox target_bb, float search_area_factor, int output_sz, float &resize_factor);

public:

    ORTrackTRT(std::string &engine_name);

    ~ORTrackTRT(); //override

    void init(const cv::Mat &img, DrOBB bbox);    
    
    const DrOBB &track(const cv::Mat &img);
    
    // state  dynamic
    DrBBox state;
    
    // config static
    Config cfg; 

    void deserialize_engine(std::string &engine_name);

    void infer(
        float  *input_imt,
        float  *input_imsearch,
        float  *output_pred_boxes,
        float  *output_pred_scores,
        cv::Size input_imt_shape,
        cv::Size input_imsearch_shape);
    
    void blob_from_image_half(cv::Mat& img, float* output_data);

    void blob_from_image_half(cv::Mat& img, cv::Mat &imgx);
    
    void half_norm(const cv::Mat &img, float* input_data);

    
};

#endif 
