#ifndef DARKNET_LIB
#define DARKNET_LIB

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include "network.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh);
extern void run_voxel(int argc, char **argv);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
// my yolo2
extern void initParam(char *cfgfile, char *weightsfile);
extern int detectObject(image *yoloImg, box *rects, int *label, float *accuracy, int limit, float thresh, float nms );
extern void releaseYololib();

struct tagDetected
{
    cv::Rect box;
    float accuracy;
    int label;
};

class Yolo2
{
    public:
        Yolo2();
        Yolo2(std::string &namefile, std::string &cfgfile, std::string &weightsfile);
        virtual ~Yolo2();
    public:
        bool detect(cv::Mat &srcImg, std::vector<tagDetected> &detected, float thresh=0.25, float nms=0.4);
        std::string &getName(int n){ return m_names[n]; }
    private:
        image mat2YoloImage(cv::Mat &src);
        std::vector<std::string> m_names;
        bool readFile(std::string &filePath);
};

#endif
