#include <iostream>
#include <assert.h>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <cmath>

#include "beanstalk.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/tracking/tracker.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "cuda_runtime.h"

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudabgsegm.hpp"

#include "json11.hpp"
#include "util.h"
#include "fixedqueue.h"
#include "panorama.h"
#include "yolo2.hpp"


using namespace std;
using namespace cv;
using namespace Beanstalk;
using namespace json11;

// Global
bool g_isBebug = false;
vector<Rect> g_detectedArea;
Rect g_box;
vector<Rect> g_Boxes;
vector<Point> g_Points;
bool drawing_box = false;
bool set_box =false;


//
bool labelling(Mat &grayImg, Rect2d &dtcRect, int minArea);
//bool searchMotion(Mat &grayImg, Mat &dstImg, vector<Rect> dstRect, const vector<Rect> &currRect, int minArea, int maxArea);
double calcLength(const Point2f &p1, const Point2f &p2);
double calcLength(const Point &p1, const Point &p2);
double calcRectDiff(const Rect &rect1, const Rect &rect2);

int trackVehicle();
//int optFlowTracking(void);
//int pano_test();
//int pano_test2();




// コールバック関数
void rect_mouse_callback(int event, int x, int y, int flags, void* param){
    Mat* image = static_cast<Mat*>(param);

    switch (event){
    case EVENT_MOUSEMOVE:
        if (drawing_box){
            g_box.width = x - g_box.x;
            g_box.height = y - g_box.y;
        }
        break;

    case EVENT_LBUTTONDOWN:
        drawing_box = true;
        g_box = Rect(x, y, 0, 0);
        break;

    case EVENT_LBUTTONUP:
        drawing_box = false;
        if (g_box.width < 0){
            g_box.x += g_box.width;
            g_box.width *= -1;
        }
        if (g_box.height < 0){
            g_box.y += g_box.height;
            g_box.height *= -1;
        }
        rectangle(*image, g_box, Scalar(0xff, 0x00, 0x00));
        g_Boxes.push_back(g_box);

        break;
    }
}

int setBoundBox(string &name, Mat &srcImg)
{

    Mat temp = srcImg.clone();

    // ウィンドウを生成
    namedWindow(name, CV_WINDOW_AUTOSIZE);

    // コールバックを設定
    setMouseCallback(name, rect_mouse_callback, (void *)&srcImg);


    // Main loop
    while (1){
        // imageをtempにコピー
        srcImg.copyTo(temp);

        // マウスの左クリックを離すまでの間、矩形を一時的に描画
        if (drawing_box) {
            rectangle(temp, g_box, Scalar(0xff, 0x00, 0x00));
        }


        imshow(name, temp);

        // Escで終了
        if (waitKey(15) == 27)
            break;
    }
    destroyAllWindows();

    return 0;
}

// call back for cureve appro
void point_mouse_callback(int event, int x, int y, int flags, void* param){
    Mat* image = static_cast<Mat*>(param);

    switch (event){

    case EVENT_LBUTTONDOWN:
        Point tp(x,y);
        circle(*image, tp, 10, Scalar(255,0,0), -1);
        g_Points.push_back(tp);
        break;

    }
}

int setPoints(string &name, Mat &srcImg)
{

    Mat temp = srcImg.clone();

    // ウィンドウを生成
    namedWindow(name, CV_WINDOW_AUTOSIZE);

    // コールバックを設定
    setMouseCallback(name, point_mouse_callback, (void *)&srcImg);


    // Main loop
    while (1){
        // imageをtempにコピー
        srcImg.copyTo(temp);


        imshow(name, temp);

        // Escで終了
        if (waitKey(15) == 27)
            break;
    }
    destroyAllWindows();

    return 0;
}



bool labelling(Mat &grayImg, Rect2d &dtcRect, int minArea=100)
{
    Mat binImg;
    threshold(grayImg,binImg,0,255,THRESH_BINARY | THRESH_OTSU);

    Mat labelImage(binImg.size(), CV_32S);
    Mat stats;
    Mat centroids;

    int nLabels = connectedComponentsWithStats(binImg, labelImage, stats, centroids, 8);

    int padding = 10;
    bool isDetect = false;
    for (int i = 1; i < nLabels; ++i) {
        int *param = stats.ptr<int>(i);
        int area = param[ConnectedComponentsTypes::CC_STAT_AREA];
        int x = param[ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[ConnectedComponentsTypes::CC_STAT_TOP];
        int width = param[ConnectedComponentsTypes::CC_STAT_WIDTH];
        int height = param[ConnectedComponentsTypes::CC_STAT_HEIGHT];

        if(x<=padding || y <= padding || x+width >= binImg.cols-padding || y+height >= binImg.rows-padding){
            continue;
        }

        if( area > minArea && width < height ){
            dtcRect.x = (double)x;
            dtcRect.y = (double)y;
            dtcRect.width = (double)width;
            dtcRect.height = (double)height;
            minArea = area;
            isDetect = true;
        }
    }
    return isDetect;
}


bool searchMotion(Mat &grayImg, vector<Rect> &boundBoxes, int minArea=100, int maxArea=200000)
{
    Mat binImg;
    threshold(grayImg,binImg,0,255,THRESH_BINARY | THRESH_OTSU);

    Mat labelImage(binImg.size(), CV_32S);
    Mat stats;
    Mat centroids;
    int nLabels = connectedComponentsWithStats(binImg, labelImage, stats, centroids, 8);

    bool isDetect = false;
    int boxNum = boundBoxes.size();
    int padding = 1;

    for (int i = 1; i < nLabels; ++i) {
        int *param = stats.ptr<int>(i);
        int area = param[ConnectedComponentsTypes::CC_STAT_AREA];
        int x = param[ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[ConnectedComponentsTypes::CC_STAT_TOP];
        int width = param[ConnectedComponentsTypes::CC_STAT_WIDTH];
        int height = param[ConnectedComponentsTypes::CC_STAT_HEIGHT];

//        if(x<=padding || y <= padding || x+width >= binImg.cols-padding || y+height >= binImg.rows-padding){
//            continue;
//        }
        if(x<=padding || x+width >= binImg.cols-padding ){
            continue;
        }

        bool isNewBox = true;
        if( area > minArea && area < maxArea ){
            Rect boundBox;
            boundBox.x = x;
            boundBox.y = y;
            boundBox.width = width;
            boundBox.height = height;

            //search same box
            for(int i=0; i<boxNum; i++){
                double length = calcRectDiff(boundBox, boundBoxes[i]);
                if(length < 200){
                    isNewBox = false;
                    break;
                }
            }
            //add new bounding box
            if(isNewBox){
                boundBoxes.push_back(boundBox);
                isDetect = true;
            }
        }
    }

    return isDetect;
}


double calcLength(const Point2f &p1, const Point2f &p2)
{
    double length = pow( (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y), 0.5 );
    return length;
}

double calcLength(const Point &p1, const Point &p2)
{
    double length = pow( ((double)p2.x-(double)p1.x)*((double)p2.x-(double)p1.x) + ((double)p2.y-(double)p1.y)*((double)p2.y-(double)p1.y), 0.5 );
    return length;
}

double calcRectDiff(const Rect &rect1, const Rect &rect2)
{
//    Point sp1(rect1.x, rect1.y);
//    Point sp2(rect2.x, rect2.y);
//    double dis1 = calcLength(sp1, sp2);
//    Point ep1(rect1.x+rect1.width, rect1.y+rect1.height);
//    Point ep2(rect2.x+rect2.width, rect2.y+rect2.height);
//    double dis2 = calcLength(ep1, ep2);

//    return (dis1+dis2)/2.0;

    Point sp1(rect1.x+(rect1.width/2), rect1.y+(rect1.height/2));
    Point sp2(rect2.x+(rect2.width/2), rect2.y+(rect2.height/2));
    double dis = calcLength(sp1, sp2);

    return dis;
}


static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void drawArrows(Mat& srcImg, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, Scalar line_color)
{
    for (size_t i = 0; i <  prevPts.size(); ++i){
        if (status[i]){
            int line_thickness = 1;

            double length = calcLength(prevPts[i], nextPts[i]);
            if(length < 0.05) continue;

            Point p = prevPts[i];
            Point q = nextPts[i];

            double angle = atan2((double)p.y - q.y, (double)p.x - q.x);
            double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x));

            if (hypotenuse <  1.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
            q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(srcImg, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.
            p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
            line(srcImg, p, q, line_color, line_thickness);

            p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
            line(srcImg, p, q, line_color, line_thickness);
        }
    }
}


void estimateTrackTransform(Mat &srcImg, vector<Rect2d> &trackBoxes, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status)
{
    int width = srcImg.cols;
    int height = srcImg.rows;
    size_t boxSize = trackBoxes.size();
    size_t ptSize = prevPts.size();

    for (size_t i = 0; i < boxSize; ++i){
        vector<Point2f> inPrevPos(0);
        vector<Point2f> inNextPos(0);
        vector<Point2f> subPrevPos(0);
        vector<Point2f> subNextPos(0);

        for (size_t j = 0; j < ptSize; ++j){
            if (status[j] ){
                double length = calcLength(prevPts[i], nextPts[i]);
                if(length > 10) continue;
           //     if(length < 0.05) continue;

                if(trackBoxes[i].x < prevPts[j].x && trackBoxes[i].x+trackBoxes[i].width > prevPts[j].x &&
                    trackBoxes[i].y < prevPts[j].y && trackBoxes[i].y+trackBoxes[i].height > prevPts[j].y){
                    inPrevPos.push_back(prevPts[j]);
                    inNextPos.push_back(nextPts[j]);
                }
                if(trackBoxes[i].x < prevPts[j].x && trackBoxes[i].x+trackBoxes[i].width > prevPts[j].x &&
                    trackBoxes[i].y < prevPts[j].y && trackBoxes[i].y+trackBoxes[i].height > prevPts[j].y){
                    subPrevPos.push_back(prevPts[j]);
                    subNextPos.push_back(nextPts[j]);
                }
            }
        }

        int inSize = inNextPos.size();
        if(inSize >= 2){
            double prevX=0, prevY=0;
            double nextX=0, nextY=0;
            for(int j=0; j<inSize; j++){
                prevX += inPrevPos[j].x;
                prevY += inPrevPos[j].y;
                nextX += inNextPos[j].x;
                nextY += inNextPos[j].y;
            }
            double diffX = nextX/(double)inSize - prevX/(double)inSize;
            double diffY = nextY/(double)inSize - prevY/(double)inSize;
            cout << "diffX=" << diffX << ",diffY=" << diffY << endl;

            if(abs(diffX) > 0.05)
                trackBoxes[i].x += diffX;

//            if(abs(diffY) > 0.05)
                trackBoxes[i].y += diffY;

        }


        if(inSize > 3){
            RotatedRect prevBox = minAreaRect(Mat(subPrevPos));
            RotatedRect nextBox = minAreaRect(Mat(subNextPos));
            Point2f pvtx[4], nvtx[4];
            prevBox.points(pvtx);
            nextBox.points(nvtx);
            for (int j = 0; j < 4; j++){
                 line(srcImg, pvtx[j], pvtx[(j + 1) % 4], Scalar(255, 255, 0), 1, LINE_AA);
                 line(srcImg, nvtx[j], nvtx[(j + 1) % 4], Scalar(0, 255, 255), 1, LINE_AA);
            }

//            //New position
//            double minX(9999),minY(9999);
//            double maxX(0),maxY(0);
//            for(int j=0;j<4;j++){
//                if(minX > nvtx[j].x)    minX = nvtx[j].x;
//                if(minY > nvtx[j].y)    minY = nvtx[j].y;
//                if(maxX < nvtx[j].x)    maxX = nvtx[j].x;
//                if(maxY < nvtx[j].y)    maxY = nvtx[j].y;
//            }

//            if(minX < 0) minX = 0;
//            if(minY < 0) minY = 0;
//            if(maxX > width) maxX = width;
//            if(maxY > height) maxY = height;

//            trackBoxes[i].x = minX;
//            trackBoxes[i].y = minY;
//            trackBoxes[i].width = maxX - minX;
//            trackBoxes[i].height = maxY - minY;


//            Rect prevRect = boundingRect( Mat(subPrevPos) );
//            Rect nextRect = boundingRect( Mat(subNextPos) );

//            rectangle(srcImg, prevRect, Scalar(255,255,0));
//            rectangle(srcImg, nextRect, Scalar(0,255,255));

//            int width, height;
//            if(nextRect.width > 0){
//                width = (int)(((double)nextRect.width/(double)prevRect.width) * (double)trackBoxes[i].width);
//                std::cout << width << ", nextRect.width=" << nextRect.width << ", prevRect.width=" << prevRect.width << std::endl;
//            }

//            if(nextRect.height > 0){
//                height = (int)(((double)nextRect.height/(double)prevRect.height) * (double)trackBoxes[i].height);
//                std::cout << height << ", nextRect.height=" << nextRect.height << ", prevRect.height=" << prevRect.height << std::endl;
//            }else{
//                height = trackBoxes[i].height;
//            }


//            Rect boundRect;
//            boundRect.x = (nextRect.x+nextRect.width/2) - width/2;
//            boundRect.y = (nextRect.y+nextRect.height/2) - height/2;
//            boundRect.width = width;
//            boundRect.height = height;

//            trackBoxes[i] = boundRect;
        }

/*
        //if inside points over 4 calc transform
        if(inPrevPos.size() >= 4 ){
            Mat masks;
            Mat H = findHomography(inPrevPos, inNextPos, masks, RANSAC);
            vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(trackBoxes[i].x,trackBoxes[i].y);
            obj_corners[1] = cvPoint(trackBoxes[i].x+trackBoxes[i].width, trackBoxes[i].y );
            obj_corners[2] = cvPoint(trackBoxes[i].x+trackBoxes[i].width, trackBoxes[i].y+trackBoxes[i].height );
            obj_corners[3] = cvPoint(trackBoxes[i].x, trackBoxes[i].y+trackBoxes[i].height );
            vector<Point2f> scene_corners(4);

            perspectiveTransform( obj_corners, scene_corners, H);

//            Rect boundRect = boundingRect( Mat(scene_corners) );
//            //New position
//            if(boundRect.x < 0) boundRect.x = 0;
//            if(boundRect.y < 0) boundRect.y = 0;
//            if(boundRect.x+boundRect.width > width) boundRect.width = width - boundRect.x;
//            if(boundRect.y+boundRect.height > height) boundRect.height = height - boundRect.y;

//            trackBoxes[i] = boundRect;


            //New position
            int minX(9999),minY(9999);
            int maxX(0),maxY(0);
            for(size_t i=0;i<scene_corners.size();i++){
                if(minX > scene_corners[i].x)    minX = scene_corners[i].x;
                if(minY > scene_corners[i].y)    minY = scene_corners[i].y;
                if(maxX < scene_corners[i].x)    maxX = scene_corners[i].x;
                if(maxY < scene_corners[i].y)    maxY = scene_corners[i].y;
            }

            if(minX < 0) minX = 0;
            if(minY < 0) minY = 0;
            if(maxX > width) maxX = width;
            if(maxY > height) maxY = height;

            trackBoxes[i].x = minX;
            trackBoxes[i].y = minY;
            trackBoxes[i].width = maxX - minX;
            trackBoxes[i].height = maxY - minY;

        }
*/
    }
}


int yoloTest(void)
{
    string namefile = "../Tracking/yolo/coco.names";
    string cfgfile = "../Tracking/yolo/yolo.cfg";
    string weightsfile = "../Tracking/yolo/yolo.weights";
    Yolo2 yolo(namefile, cfgfile ,weightsfile);

    Mat img = imread("../Tracking/data/street4.jpg");
    vector<tagDetected> yoloResult;
    yolo.detect(img, yoloResult);

    int size = yoloResult.size();
    string name;
    for(int i=0; i<size; i++){
        rectangle(img, yoloResult[i].box, Scalar(200,0,200), 2, 1);
        name = yolo.getName(yoloResult[i].label);
        putText(img, name, Point(yoloResult[i].box.x, yoloResult[i].box.y), 0, 1, Scalar(200,200,0), 1);
//        putText(img, to_string(yoloResult[i].accuracy), Point(yoloResult[i].box.x, yoloResult[i].box.y+40), 0, 1.5, Scalar(200,0,200), 2);
    }

    while(1){
        imshow("test image", img);
        if(waitKey(10) == 27){
            break;
        }
    }
    return 0;
}

int trackingGpu(void)
{

    int camNum = 5;
    VideoCapture vcap[camNum];
    string camAddress[camNum];
    string window_name[camNum];
    Mat knnImg;
    cuda::GpuMat knnGpuImg;
    vector<Mat> frame(camNum);
    vector<Mat> srcImg(camNum);
    Mat panoImg;
    cuda::GpuMat panoGpuImg;
    cuda::GpuMat prevGpuImg;
    cuda::GpuMat currGpuImg;
    Panorama pano;//    pano.setGpu(true);
//    Ptr<cuda::BackgroundSubtractorFGD> pBGS = cuda::createBackgroundSubtractorFGD();
//    Ptr<cuda::BackgroundSubtractorGMG> pBGS = cuda::createBackgroundSubtractorGMG();
//    Ptr<cuda::BackgroundSubtractorMOG> pBGS = cuda::createBackgroundSubtractorMOG();
    Ptr<cuda::BackgroundSubtractorMOG2> pBGS = cuda::createBackgroundSubtractorMOG2();
    cuda::GpuMat d_prevPts;
    cuda::GpuMat d_nextPts;
    cuda::GpuMat d_status;
    Ptr< cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(CV_8U, 4000, 0.01, 0);
    Ptr< cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);

    TickMeter meter;
    TickMeter meter2;
    vector<Rect> trackingRect(0);

//    string namefile = "../Tracking/yolo/coco.names";
//    string cfgfile = "../Tracking/yolo/yolo.cfg";
//    string weightsfile = "../Tracking/yolo/yolo.weights";
//    Yolo2 yolo(namefile, cfgfile ,weightsfile);
    Rect roiRect;

    if(g_isBebug){
        for(int i=0; i<camNum; i++){
//            camAddress[i] = strsprintf("./Videos/s_cam%d.avi", i+1);
            camAddress[i] = strsprintf("./Videos/Fingermark-video/19-4-16/Camera%d.mp4", i+1);
        }
    }else{
        camAddress[0] = "rtsp://192.168.5.103:554/s1";
        camAddress[1] = "rtsp://192.168.5.7:554/s1";
        camAddress[2] = "rtsp://192.168.5.118:554/s1";
        camAddress[3] = "rtsp://192.168.5.8:554/s1";
    }


    //camera check
    for(int i=0; i<camNum; i++){
        if (!vcap[i].open(camAddress[i])) {
            cout << "error opening camera stream ...." << i << endl;
            return -1;
        }
        window_name[i] = strsprintf("camNo%d", i+1);
    }

    bool is_first = true;
    bool isCapError = false;
    while (1) {
        for(int i=0; i<camNum; i++){
            if(!vcap[i].read(frame[i])){
                cout << "capture error" << endl;
                isCapError = true;
                break;
            }
            resize(frame[i], srcImg[i], Size(), 0.4, 0.4);
//            cvtColor(srcImg[i], srcImg[i], COLOR_BGR2GRAY);
//                imshow(window_name[i], frame[i]);
        }
        if(isCapError) break;


        if(is_first){
            pano.estimateAndCompose(srcImg, panoImg);
            is_first = false;
//            panoGpuImg.upload(panoImg);
//            cuda::cvtColor(panoGpuImg, prevGpuImg, COLOR_RGB2GRAY);
            roiRect.x = 0;
            roiRect.y = 0;
            roiRect.width = panoImg.rows;
            roiRect.height = panoImg.rows;
        }else{
//            meter.reset();
//            meter.start();

            //make pano
            pano.composePanorama(srcImg, panoImg);
//            panoGpuImg.upload(panoImg);
//            cuda::cvtColor(panoGpuImg, currGpuImg, COLOR_RGB2GRAY);

/*            //feature
            detector->detect(prevGpuImg, d_prevPts);
            d_pyrLK->calc(prevGpuImg, currGpuImg, d_prevPts, d_nextPts, d_status);
            //copy to prev
            currGpuImg.copyTo(prevGpuImg);

            // Draw arrows
            vector< Point2f> prevPts(d_prevPts.cols);
            download(d_prevPts, prevPts);

            vector< Point2f> nextPts(d_nextPts.cols);
            download(d_nextPts, nextPts);

            vector< uchar> status(d_status.cols);
            download(d_status, status);

            drawArrows(panoImg, prevPts, nextPts, status, Scalar(155, 0, 0));

            //estimate tracking rect movement
            estimateTrackTransform(panoImg, trackingRect, prevPts, nextPts, status);
            for(size_t i=0; i<trackingRect.size(); i++){
                rectangle(panoImg, trackingRect[i], Scalar(200,0,200), 2);
            }

            meter.stop();
            std::cout << "OptFlow:"<< meter.getTimeMilli() << "ms" << std::endl;
*/
        }

//        meter2.reset();
//        meter2.start();

//        Mat roiImg = panoImg(roiRect);
//        rectangle(panoImg, roiRect, Scalar(255, 0, 255));

//        vector<tagDetected> yoloResult;
//        yolo.detect(roiImg, yoloResult);
/*
        pBGS->apply(panoGpuImg, knnGpuImg);
        if(!is_first){
            //Labelling
            knnGpuImg.download(knnImg);
            if( searchMotion(knnImg,trackingRect, 800) ){
//                int size = trackingRect.size();
//                for(int i=0; i<size; i++){
//                    rectangle(panoImg, trackingRect[i], Scalar(200,0,200), 2);
//                }
            }
        }
*/
//        meter2.stop();
//        std::cout << "BackGround:" << meter2.getTimeMilli() << "ms" << std::endl;

        imshow("panorama image", panoImg);

        if(waitKey(10) == 27){
            break;
        }
    }

    destroyAllWindows();

    return 0;
}


// Cafe has memory leak bug !!!!!!!!!!!!!!!!
// if you use Ptr (OpenCV)
/*
int caffeTest()
{
    ::google::InitGoogleLogging("./npd");

//    string model_file = "../Tracking/caffe_models/cifar10_full.prototxt";
//    string trained_file = "../Tracking/caffe_models/cifar10_full_iter_70000.caffemodel.h5";
//    string mean_file = "../Tracking/caffe_models/mean.binaryproto";
//    string label_file = "../Tracking/caffe_models/batches.meta.txt";

    string model_file = "../Tracking/caffe_models2/deploy.prototxt";
    string trained_file = "../Tracking/caffe_models2/bvlc_reference_caffenet.caffemodel";
    string mean_file = "../Tracking/caffe_models2/imagenet_mean.binaryproto";
    string label_file = "../Tracking/caffe_models2/synset_words.txt";


    bool isGpu = true;

    Classifier classifier(model_file, trained_file, mean_file, label_file, isGpu);

    string img_file = "../Tracking/caffe_models/cat.jpg";
    Mat img = imread(img_file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << img_file;
    vector<Prediction> predictions = classifier.Classify(img, 3);

    // Print the top N predictions.
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
//        cout << p.second << " - \""<< p.first << "\"" << endl;
        cout << fixed << setprecision(4) << p.second << " - \""
              << p.first << "\"" << endl;
    }

    return 0;
}
*/

int cudaTest()
{
    Mat src = imread("./images/1.jpg", 0);
    if (!src.data) exit(1);
    cuda::GpuMat d_src(src);
    cuda::GpuMat d_dst;
    cuda::bilateralFilter(d_src, d_dst, -1, 50, 7);
    Mat dst;
    d_dst.download(dst);
    cv::Canny(dst, dst, 35, 200, 3);
//    resize(dst, dst, Size(dst.cols/3, dst.rows/3));
    imshow("test", dst); waitKey(0);
    return 0;
}


bool findNewVehicle(vector<Rect> &detected, vector<tagDetected> &yoloResult, int minWidth, int minHeight)
{
    // find good one
    int size = yoloResult.size();
    int xpos = 0;
    Rect candidate;
    for(int i=0; i<size; i++){
        if(yoloResult[i].label == 2){   // 2 == car (from coco model)
            yoloResult[i].box.x += g_Boxes[0].x;
            yoloResult[i].box.y += g_Boxes[0].y;
            if(xpos < yoloResult[i].box.x){
                candidate = yoloResult[i].box;
            }
        }
    }


//    if(x<=padding || x+width >= binImg.cols-padding )   return false;
    int currNum = detected.size();

    if( candidate.width > minWidth && candidate.height > minHeight ){
        // if no detected bounding box, just add candidate as new
        if(currNum == 0){
            detected.push_back(candidate);
            return true;
        }
        //check candidate position
        else{
            for(int j=0; j<currNum; j++){
                double length = calcRectDiff(candidate, detected[j]);
                if(length < 150){
                    return false;
                }
            }
            //add new bounding box
            detected.push_back(candidate);
            return false;
        }
    }

    return false;
}

bool findNewVehicleEx(Rect &findRect, vector<Rect2d> &detected, vector<tagDetected> &yoloResult, int minWidth, int minHeight)
{
    // find good one
    int size = yoloResult.size();
    int xpos = 0;
    Rect2d candidate;
    for(int i=0; i<size; i++){
        if(yoloResult[i].label == 2){   // 2 == car (from coco model)
            yoloResult[i].box.x += g_Boxes[0].x;
            yoloResult[i].box.y += g_Boxes[0].y;
            if(xpos < yoloResult[i].box.x){
                candidate = yoloResult[i].box;
            }
        }
    }


//    if(x<=padding || x+width >= binImg.cols-padding )   return false;
    int currNum = detected.size();

    if( candidate.width > minWidth && candidate.height > minHeight ){
        // if no detected bounding box, just add candidate as new
        if(currNum == 0){
            findRect = candidate;
            return true;
        }
        //check candidate position
        else{
            for(int j=0; j<currNum; j++){
                double length = calcRectDiff(candidate, detected[j]);
                if(length < 150){
                    return false;
                }
            }
            //add new bounding box
            findRect = candidate;
            return false;
        }
    }

    return false;
}

int trackVehicle3(){

    VideoCapture vcap;

    string camAddress;
    if (g_isBebug) {
//        camAddress = "../Tracking/data/DJI_0204.MOV";
//        camAddress = "../Tracking/data/sample.avi";
        camAddress = "../Tracking/data/DJI_0204_B_STABLE.mp4";
    }else {
        camAddress = "rtsp://192.168.5.7:554/s1";
    }
    if (!vcap.open(camAddress)) {
        cout << "error opening camera stream ...." << endl;
        return -1;
    }

    Mat frame;
    cuda::GpuMat gpuImg, prevGpuImg, nextGpuImg;
    cuda::GpuMat prevGpuPts;
    cuda::GpuMat nextGpuPts;
    cuda::GpuMat ptsGpuStatus;
    Ptr< cuda::CornersDetector> pDetector = cuda::createGoodFeaturesToTrackDetector(CV_8U, 4000, 0.01, 0);
    Ptr< cuda::SparsePyrLKOpticalFlow> pPyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);

//    string namefile = "../Tracking/yolo/coco.names";
//    string cfgfile = "../Tracking/yolo/yolo.cfg";
//    string weightsfile = "../Tracking/yolo/yolo.weights";
//    Yolo2 yolo(namefile, cfgfile ,weightsfile);


    if(!vcap.read(frame)){
        cout << "capture error" << endl;
        return -1;
    }
    Mat dstImg = Mat::ones(frame.rows * 0.4, frame.cols * 0.4, CV_8UC3);
    resize(frame, dstImg, dstImg.size(), INTER_AREA);
    string setWinName = "Points Setting...";
    setPoints(setWinName, dstImg);
    vector<tagDetected> yoloResult;
    vector<Rect2d> detectedRect;
    bool isInit = false;

//    int fourcc   = VideoWriter::fourcc('X', 'V', 'I', 'D');
//    double fps   = 30.0;
//    bool isColor = true;
//    VideoWriter writer("sample.avi", fourcc, fps, dstImg.size(), isColor);


    if( g_Points.size() < 6) return -1;
/*
    int widthMax = 0;
    bool isDetected = false;
    Rect findRect;
    findRect.x = 0;
    findRect.y = 0;
    findRect.width = 0;
    findRect.height = 0;
    int serachCount = 0;

    while (1) {

        if(!vcap.read(frame)){
            cout << "capture error" << endl;
            break;
        }
        resize(frame, dstImg, dstImg.size(), INTER_AREA);

        rectangle(dstImg, g_Boxes[0], Scalar(100,100,100), 1, 1);

        if(isInit){
            gpuImg.upload(dstImg);
            cuda::cvtColor(gpuImg, nextGpuImg, COLOR_RGB2GRAY);

//            int detectedSize = detectedRect.size();
//            for(int i=0; i<detectedSize; i++){
//                cuda::GpuMat prevRoi = prevGpuImg(detectedRect[i]);
//                cuda::GpuMat nextRoi = nextGpuImg(detectedRect[i]);
//            }

            //feature
            pDetector->detect(prevGpuImg, prevGpuPts);
            pPyrLK->calc(prevGpuImg, nextGpuImg, prevGpuPts, nextGpuPts, ptsGpuStatus);
            //copy to prev
            nextGpuImg.copyTo(prevGpuImg);

            // Draw arrows
            vector< Point2f> prevPts(prevGpuPts.cols);
            download(prevGpuPts, prevPts);
            vector< Point2f> nextPts(nextGpuPts.cols);
            download(nextGpuPts, nextPts);
            vector< uchar> status(ptsGpuStatus.cols);
            download(ptsGpuStatus, status);

            drawArrows(dstImg, prevPts, nextPts, status, Scalar(155, 0, 0));

            estimateTrackTransform(dstImg, detectedRect, prevPts, nextPts, status);
            int boxSize = detectedRect.size();
            for(int i=0; i<boxSize; i++){
                rectangle(dstImg, detectedRect[i], Scalar(255, 0, 255));
            }
        }


        // yolo
        yoloResult.clear();
        Mat roiImg = dstImg(g_Boxes[0]);
        yolo.detect(roiImg, yoloResult);
//        findNewVehicle(detectedRect, yoloResult, 300, 100);

        //Find most longest vehicle from candidates
        if(findNewVehicleEx(findRect,detectedRect, yoloResult, 300, 100)){
            if(widthMax < findRect.width ){
                widthMax = findRect.width;
                isDetected = true;
                serachCount = 0;
            }else{
                if(isDetected){
                    serachCount++;
                    if(serachCount > 1){
                        detectedRect.push_back(findRect);
                        isDetected = false;
                    }
                }
            }
        }


        if(detectedRect.size() > 0 && !isInit){
            gpuImg.upload(dstImg);
            cuda::cvtColor(gpuImg, prevGpuImg, COLOR_RGB2GRAY);
            isInit = true;
        }


//        writer << dstImg;
        imshow("Tracking", dstImg);


        //get the input from the keyboard
        if(waitKey(10) == 27){
            break;
        }
    }
    vcap.release();
    destroyAllWindows();
*/
    return 0;
}

int trackVehicle(){

    VideoCapture vcap;

    string camAddress;
    if (g_isBebug) {
//        camAddress = "../Tracking/data/DJI_0204.MOV";
//        camAddress = "../Tracking/data/sample.avi";
        camAddress = "../Tracking/data/DJI_0204_B_STABLE.mp4";
    }else {
        camAddress = "rtsp://192.168.5.7:554/s1";
    }
    if (!vcap.open(camAddress)) {
        cout << "error opening camera stream ...." << endl;
        return -1;
    }

    Mat frame;
    cuda::GpuMat gpuImg, prevGpuImg, nextGpuImg;
    cuda::GpuMat prevGpuPts;
    cuda::GpuMat nextGpuPts;
    cuda::GpuMat ptsGpuStatus;
    Ptr< cuda::CornersDetector> pDetector = cuda::createGoodFeaturesToTrackDetector(CV_8U, 4000, 0.01, 0);
    Ptr< cuda::SparsePyrLKOpticalFlow> pPyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);

    string namefile = "../Tracking/yolo/coco.names";
    string cfgfile = "../Tracking/yolo/yolo.cfg";
    string weightsfile = "../Tracking/yolo/yolo.weights";
    Yolo2 yolo(namefile, cfgfile ,weightsfile);


    if(!vcap.read(frame)){
        cout << "capture error" << endl;
        return -1;
    }
    Mat dstImg = Mat::ones(frame.rows * 0.4, frame.cols * 0.4, CV_8UC3);
    resize(frame, dstImg, dstImg.size(), INTER_AREA);
    string setWinName = "Area Setting...";
    setBoundBox(setWinName, dstImg);
    vector<tagDetected> yoloResult;
    vector<Rect2d> detectedRect;
    bool isInit = false;

//    int fourcc   = VideoWriter::fourcc('X', 'V', 'I', 'D');
//    double fps   = 30.0;
//    bool isColor = true;
//    VideoWriter writer("sample.avi", fourcc, fps, dstImg.size(), isColor);


    if( g_Boxes.size() < 1) return -1;

    int widthMax = 0;
    bool isDetected = false;
    Rect findRect;
    findRect.x = 0;
    findRect.y = 0;
    findRect.width = 0;
    findRect.height = 0;
    int serachCount = 0;

    while (1) {

        if(!vcap.read(frame)){
            cout << "capture error" << endl;
            break;
        }
        resize(frame, dstImg, dstImg.size(), INTER_AREA);

        rectangle(dstImg, g_Boxes[0], Scalar(100,100,100), 1, 1);

        if(isInit){
            gpuImg.upload(dstImg);
            cuda::cvtColor(gpuImg, nextGpuImg, COLOR_RGB2GRAY);

//            int detectedSize = detectedRect.size();
//            for(int i=0; i<detectedSize; i++){
//                cuda::GpuMat prevRoi = prevGpuImg(detectedRect[i]);
//                cuda::GpuMat nextRoi = nextGpuImg(detectedRect[i]);
//            }

            //feature
            pDetector->detect(prevGpuImg, prevGpuPts);
            pPyrLK->calc(prevGpuImg, nextGpuImg, prevGpuPts, nextGpuPts, ptsGpuStatus);
            //copy to prev
            nextGpuImg.copyTo(prevGpuImg);

            // Draw arrows
            vector< Point2f> prevPts(prevGpuPts.cols);
            download(prevGpuPts, prevPts);
            vector< Point2f> nextPts(nextGpuPts.cols);
            download(nextGpuPts, nextPts);
            vector< uchar> status(ptsGpuStatus.cols);
            download(ptsGpuStatus, status);

            drawArrows(dstImg, prevPts, nextPts, status, Scalar(155, 0, 0));

            estimateTrackTransform(dstImg, detectedRect, prevPts, nextPts, status);
            int boxSize = detectedRect.size();
            for(int i=0; i<boxSize; i++){
                rectangle(dstImg, detectedRect[i], Scalar(255, 0, 255));
            }
        }


        // yolo
        yoloResult.clear();
        Mat roiImg = dstImg(g_Boxes[0]);
        yolo.detect(roiImg, yoloResult);
//        findNewVehicle(detectedRect, yoloResult, 300, 100);

        //Find most longest vehicle from candidates
        if(findNewVehicleEx(findRect,detectedRect, yoloResult, 300, 100)){
            if(widthMax < findRect.width ){
                widthMax = findRect.width;
                isDetected = true;
                serachCount = 0;
            }else{
                if(isDetected){
                    serachCount++;
                    if(serachCount > 1){
                        detectedRect.push_back(findRect);
                        isDetected = false;
                    }
                }
            }
        }


        if(detectedRect.size() > 0 && !isInit){
            gpuImg.upload(dstImg);
            cuda::cvtColor(gpuImg, prevGpuImg, COLOR_RGB2GRAY);
            isInit = true;
        }


//        writer << dstImg;
        imshow("Tracking", dstImg);


        //get the input from the keyboard
        if(waitKey(10) == 27){
            break;
        }
    }
    vcap.release();
    destroyAllWindows();
    return 0;
}


int trackVehicle2(){

    //Video Caputure
    VideoCapture vcap;
    string camAddress;
    if (g_isBebug) {
//        camAddress = "../Tracking/data/sample.avi";
//        camAddress = "../Tracking/data/DJI_0001.MOV";
//        camAddress = "../Tracking/data/0001.avi";
//        camAddress = "../Tracking/data/0207.avi";
        camAddress = "../Tracking/data/0204.avi";
    }else {
        camAddress = "rtsp://192.168.5.7:554/s1";
    }
    if (!vcap.open(camAddress)) {
        cout << "error opening camera stream ...." << endl;
        return -1;
    }
    // 1st frame Capture
    Mat frame;
    if(!vcap.read(frame)){
        cout << "capture error" << endl;
        return -1;
    }
    double scale = 1.0;
    Mat dstImg = Mat::ones(frame.rows * scale, frame.cols * scale, CV_8UC3);
//    Mat rotateImg = Mat::ones(frame.rows * scale, frame.cols * scale, CV_8UC3);
//    Mat dstImg = Mat::ones(frame.cols * scale, frame.rows * scale, CV_8UC3);
    resize(frame, dstImg, dstImg.size(), INTER_AREA);

    //Yolo
    string namefile = "../Tracking/yolo/coco.names";
    string cfgfile = "../Tracking/yolo/yolo.cfg";
    string weightsfile = "../Tracking/yolo/yolo.weights";
    Yolo2 yolo(namefile, cfgfile ,weightsfile);

    //Area setting ....
    string name = "Area Setting...";
    setBoundBox(name, dstImg);
    vector<tagDetected> yoloResult;
    Rect detectedRect;
    if( g_Boxes.size() < 3) return -1;

    //video save
    int fourcc   = VideoWriter::fourcc('X', 'V', 'I', 'D');
    double fps   = 30.0;
    bool isColor = true;
    VideoWriter writer("sample.avi", fourcc, fps, dstImg.size(), isColor);

    ulong frame_count = 0;

    bool isDetected =false;
    int missCount = 0;

    //Roop
    while (1) {
        if(!vcap.read(frame)){
            cout << "capture error" << endl;
            break;
        }
        resize(frame, dstImg, dstImg.size(), INTER_AREA);
//        resize(frame, rotateImg, rotateImg.size(), INTER_AREA);
//        flip(rotateImg.t(), dstImg, 1);

        //Draw line
        rectangle(dstImg, g_Boxes[1], Scalar(0,200,0), -1, 1);
        putText(dstImg, "Entry", Point(g_Boxes[1].x-20, g_Boxes[1].y-10), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0,200,0), 1, CV_AA);
        rectangle(dstImg, g_Boxes[2], Scalar(0,0,200), -1, 1);
        putText(dstImg, "Exit", Point(g_Boxes[2].x-20, g_Boxes[2].y-10), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(0,0,200), 1, CV_AA);


        //Yolo
        yoloResult.clear();
        yolo.detect(dstImg, yoloResult);
        int size = yoloResult.size();
        bool isFind = false;

        int minWidth = 0;
        for(int i=0; i<size; i++){
            if(yoloResult[i].label == 2){

                //Enrty & Exit
                int cx = yoloResult[i].box.x + yoloResult[i].box.width/2;
                int cy = yoloResult[i].box.y + yoloResult[i].box.height/2;

                // Mask inside check
                if( !(g_Boxes[0].x < cx && g_Boxes[0].x+g_Boxes[0].width > cx
                        && g_Boxes[0].y < cy && g_Boxes[0].y+g_Boxes[0].height > cy)){
                    if(minWidth < yoloResult[i].box.width){
                        detectedRect = yoloResult[i].box;
                        minWidth = yoloResult[i].box.width;
                        isFind = true;
                    }
//                    rectangle(dstImg, detectedRect, Scalar(200,200,0), 2, 1);
                }
            }
        }

        if(isFind){
            isDetected = true;
            missCount = 0;
        }else{
            missCount++;
            if(isDetected && missCount > 10){
                isDetected = false;
            }
        }

        //detected
        if(isDetected){
            //Enrty & Exit
            int cx2 = detectedRect.x + detectedRect.width/2;
            int cy2 = detectedRect.y + detectedRect.height/2;

            if(g_Boxes[1].x >= cx2 && g_Boxes[2].y <= cy2){   //0204
//            if(g_Boxes[1].x <= cx2 && g_Boxes[2].x >= cx2){   //0207
//            if(g_Boxes[1].x <= cx2 && g_Boxes[2].y >= cy2){ //0001
                frame_count++;
                double spend_time = (double)frame_count*0.033;  //0204
//                double spend_time = (double)frame_count*0.036;  //0207
//                double spend_time = (double)frame_count*0.033;  //0001
                string strTime = strsprintf("%.1lfsec", spend_time);
                putText(dstImg, strTime, Point(detectedRect.x, detectedRect.y-10), FONT_HERSHEY_TRIPLEX, 0.6, Scalar(200,0,200), 1, CV_AA);
                rectangle(dstImg, detectedRect, Scalar(200,0,200), 2, 1);
            }else{
                frame_count = 0;
            }
        }


        writer << dstImg;
        imshow("Tracking", dstImg);
//        imshow("KNN", knnImg);
//        imshow("MOG2", mogImg);


        //get the input from the keyboard
        if(waitKey(10) == 27){
            break;
        }
    }
    vcap.release();
    destroyAllWindows();
    return 0;
}


int pano_test(){

    int camNum = 4;
    VideoCapture vcap[camNum];
    string camAddress[camNum];

    if(g_isBebug){
        for(int i=0; i<camNum; i++)
//            camAddress[i] = strsprintf("./Videos/Fingermark-video/19-4-16/Camera%d.mp4", i+1);
            camAddress[i] = strsprintf("./Videos/cam%d.avi", i+1);
    }else{
        camAddress[0] = "rtsp://192.168.5.103:554/s1";
        camAddress[1] = "rtsp://192.168.5.7:554/s1";
        camAddress[2] = "rtsp://192.168.5.118:554/s1";
        camAddress[3] = "rtsp://192.168.5.8:554/s1";
    }

    bool try_use_gpu = false;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);

    stitcher.setFeaturesFinder(new detail::OrbFeaturesFinder());

//    stitcher.setWarper(new CylindricalWarperGpu());
//    stitcher.setFeaturesFinder(new detail::OrbFeaturesFinder(Size(3,1),500));
//    stitcher.setWaveCorrection(false);
//    stitcher.setSeamEstimationResol(0.001);
//    stitcher.setPanoConfidenceThresh(0.1);

////    stitcher.setSeamFinder(new cv::detail::GraphCutSeamFinder(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD));
//    stitcher.setSeamFinder(new detail::NoSeamFinder());
//    stitcher.setBlender(detail::Blender::createDefault(detail::Blender::NO, true));
//    //stitcher.setExposureCompensator(cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::NO));
//    stitcher.setExposureCompensator(new detail::NoExposureCompensator());



    Mat panoImg;
//    UMat panoImgGpu;
    cuda::GpuMat panoImgGpu;
    vector<Mat> imgs(camNum);
//    vector<cuda::GpuMat> imgs(camNum);
//    vector<UMat> imgs(4);
    Mat frame[camNum];
    string window_name[camNum];

    for(int i=0; i<camNum; i++){
        if (!vcap[i].open(camAddress[i])) {
            cout << "error opening camera stream ...." << i << endl;
            return -1;
        }
        window_name[i] = strsprintf("camNo%d", i+1);
    }


    bool is_first = true;
    while (1) {
        for(int i=0; i<camNum; i++){
            if(!vcap[i].read(frame[i])){
                cout << "capture error" << endl;
                break;
            }
//            imgs[i].upload(frame[i]);
            imgs[i] = frame[i];
            imshow(window_name[i], frame[i]);
        }

//        if(is_first){
//            Stitcher::Status status0 = stitcher.estimateTransform(imgs);
//            is_first = false;
//        }else{
//            Stitcher::Status status = stitcher.composePanorama(imgs, panoImg);
//            if (status != Stitcher::OK)
//            {
//                cout << "Can't stitch images, error code = " << int(status) << endl;
//                return -1;
//            }
//            imshow("pano", panoImg);
//        }


        if(waitKey(10) == 27){
            break;
        }
    }
    destroyAllWindows();
    for(int i=0; i<camNum; i++){
        string filename = strsprintf("cam%d.jpg", i+1);
        imwrite(filename, imgs[i]);
        vcap[i].release();
    }


    Stitcher::Status status = stitcher.stitch(imgs, panoImg);
//    Stitcher::Status status0 = stitcher.estimateTransform(imgs);
//    Stitcher::Status status = stitcher.composePanorama(imgs, panoImgGpu);
    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }
//    panoImgGpu.download(panoImg);
    imwrite("pano.jpg", panoImg);
//    imwrite("pano.jpg", panoImgGpu);

    while(1){
        imshow("pano", panoImg);
        if(waitKey(10) == 27){
            break;
        }
    }

    return 0;
}


int main(int argc, char *argv[])
{
    g_isBebug = true;
//    trackVehicle3();
//    trackVehicle2();
    //    caffeTest();
    //    cudaTest();
    //    yoloTest();
    trackingGpu();
//    pano_test();
    //    pano_test2();
    //    stich5();
    //    optFlowTracking();
    return 0;
}



/*
int optFlowTracking(void)
{
    VideoCapture vcap;
    string camAddress = "rtsp://192.168.5.7:554/s1";
    if(!vcap.open(camAddress)){
        cout << "error opening camera stream ...." << endl;
        return -1;
    }

    Mat frame;
    if(!vcap.read(frame)){
        cout << "capture error" << endl;
        return -1;
    }

    int fourcc   = VideoWriter::fourcc('X', 'V', 'I', 'D');
    double fps   = 30.0;
    bool isColor = true;
    VideoWriter writer("test.avi", fourcc, fps, Size(640, 360), isColor);

    Mat prevImg, currImg;
    vector<Point2f> prevCorners, currCorners;
    vector<uchar> featuresFound;
    vector<float> featuresErrors;
    cvtColor(frame, prevImg, COLOR_BGR2GRAY);
    goodFeaturesToTrack(prevImg,prevCorners,100,0.3,7);
    cornerSubPix(prevImg, prevCorners, Size(21, 21), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));

    while(1){
        if(!vcap.read(frame)){
            cout << "capture error" << endl;
            return -1;
        }
        cvtColor(frame, currImg, COLOR_BGR2GRAY);
        goodFeaturesToTrack(prevImg,currCorners,100,0.3,7);
        cornerSubPix(currImg, currCorners, Size(21, 21), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));

        calcOpticalFlowPyrLK(
                    prevImg,
                    currImg,
                    prevCorners,
                    currCorners,
                    featuresFound,
                    featuresErrors);

        for (int i = 0; i < featuresFound.size(); i++) {
            double length = calcLength(prevCorners[i], currCorners[i]);
            if(length > 5){
                Point p1 = Point((int) prevCorners[i].x, (int) prevCorners[i].y);
                Point p2 = Point((int) currCorners[i].x, (int) currCorners[i].y);
                line(frame, p1, p2, Scalar(0, 0, 255), 2);
            }
        }

        currImg.copyTo(prevImg);
        prevCorners.resize(currCorners.size());
        copy(currCorners.begin(), currCorners.end(), prevCorners.begin());

        imshow("preview", frame);
        writer << frame;
        if(waitKey(10) == 27){
            destroyAllWindows();
            break;
        }
    }


}



*/
