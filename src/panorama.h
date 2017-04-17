#ifndef PANORAMA_H
#define PANORAMA_H

#include <iostream>
#include <time.h>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"


class Panorama
{
public:
    Panorama();

private:
    int m_nLeft;
    int m_nRight;
    std::vector<cv::Mat> m_leftHomo;
    std::vector<cv::Mat> m_rightHomo;
    std::vector<cv::cuda::GpuMat> m_leftHomoGPU;
    std::vector<cv::cuda::GpuMat> m_rightHomoGPU;
    std::vector<cv::Size> m_leftSize;
    std::vector<cv::Size> m_rightSize;        
    cv::Mat m_dstImg;
    cv::cuda::GpuMat m_dstImgGPU;
    int m_siftWidth;
    bool m_isGpu;
private:
    bool hFromRansac( cv::Mat &image1, cv::Mat &image2, cv::Mat &homography);
    cv::Size getDistSize(cv::Mat &H, cv::Mat &srcImg);
public:
    void setGpu(bool enable);
    int estimateAndCompose(std::vector<cv::Mat> &imgs, cv::Mat &result);
    int composePanorama(std::vector<cv::Mat> &imgs, cv::Mat &result);
};

#endif // PANORAMA_H
