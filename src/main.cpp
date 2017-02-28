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
#include "opencv2/core/cuda.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/tracking/tracker.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "cuda_runtime.h"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "../include/json11.hpp"
#include "../include/util.h"
#include "../include/fixedqueue.h"


using namespace std;
using namespace cv;
using namespace Beanstalk;
using namespace json11;

// Global
bool g_isBebug = false;
vector<Rect> g_detectedArea;

bool checkTrackerAlgType(String& trackerAlgo);
bool labelling(Mat &grayImg, Rect2d &dtcRect, int minArea);
bool searchMotion(Mat &grayImg, Mat &dstImg, vector<Rect> dstRect, const vector<Rect> &currRect, int minArea, int maxArea);
int trackVehicle();
double clacLength(Point2f p1, Point2f p2);
int optFlowTracking(void);
int pano_test();
int pano_test2();
bool hFromRansac( Mat &image1, Mat &image2, Mat &homography);
int stich3(Mat (&srcImg)[3]);
Size getDistSize(Mat &H, Mat &srcImg);



bool checkTrackerAlgType(cv::String& trackerAlgo)
{
    if (trackerAlgo == "BOOSTING" ||
        trackerAlgo == "MIL" ||
        trackerAlgo == "TLD" ||
        trackerAlgo == "MEDIANFLOW" ||
        trackerAlgo == "KCF" ||
        trackerAlgo == "GOTURN")
    {
        cout << "Tracker Algorithm Type: " << trackerAlgo << endl;
    }
    else{
        CV_Error(Error::StsError, "Unsupported algorithm type " + trackerAlgo + " is specified.");
    }
    return true;
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

        if( area > minArea){
            dtcRect.x = (double)x;
            dtcRect.y = (double)y;
            dtcRect.width = (double)width;
            dtcRect.height = (double)height;
            isDetect = true;
        }
    }
    return isDetect;
}


bool searchMotion(Mat &grayImg, Mat &dstImg, vector<Rect> dstRect, const vector<Rect> &currRect, int minArea=100, int maxArea=200000)
{
    Mat binImg;
    threshold(grayImg,binImg,0,255,THRESH_BINARY | THRESH_OTSU);

    Mat labelImage(binImg.size(), CV_32S);
    Mat stats;
    Mat centroids;
    int nLabels = connectedComponentsWithStats(binImg, labelImage, stats, centroids, 8);

    vector<Rect> goodRect;
    bool isDetect = false;
    for (int i = 1; i < nLabels; ++i) {
        int *param = stats.ptr<int>(i);
        int area = param[ConnectedComponentsTypes::CC_STAT_AREA];
        int x = param[ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[ConnectedComponentsTypes::CC_STAT_TOP];
        int width = param[ConnectedComponentsTypes::CC_STAT_WIDTH];
        int height = param[ConnectedComponentsTypes::CC_STAT_HEIGHT];

        if( area > minArea && area < maxArea ){
            Rect boundBox;
            boundBox.x = x;
            boundBox.y = y;
            boundBox.width = width;
            boundBox.height = height;
            goodRect.push_back(boundBox);
//            rectangle(dstImg, boundBox, Scalar(200,0,200), 2);
        }
    }


//    int size = goodRect.size();
//    for(int i=0; i<size ; i++){
//        for(int j=size-1; j=0; j++){
//            if(i!=j){

//            }
//        }
//    }



    return true;
}



int trackVehicle(){

    VideoCapture vcap;

    string camAddress;
    if (g_isBebug) {
        camAddress = "../track-vehicle/data/test2.avi";
    }else {
        camAddress = "rtsp://192.168.5.7:554/s1";
    }
    if (!vcap.open(camAddress)) {
        cout << "error opening camera stream ...." << endl;
        return -1;
    }

    Ptr<BackgroundSubtractorKNN> pKNN = createBackgroundSubtractorKNN();
//    Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

    Ptr<Tracker> tracker = Tracker::create("MIL");


    Mat frame, dstImg, grayImg;
    Mat knnImg, mogImg;
    Rect candidateBox;
    bool isDetected = false;
    bool isTrackInt = false;

    int fourcc   = VideoWriter::fourcc('X', 'V', 'I', 'D');
    double fps   = 30.0;
    bool isColor = true;
    VideoWriter writer("test.avi", fourcc, fps, Size(640, 360), isColor);


    while (1) {
        if(!vcap.read(frame)){
            cout << "capture error" << endl;
            break;
        }

        frame.copyTo(dstImg);


        cvtColor(frame, grayImg, COLOR_BGR2GRAY);
        //motion detection
        pKNN->apply(grayImg, knnImg);
//        pMOG2->apply(grayImg, mogImg);
//        dilate(knnImg, knnImg, Mat());
//        dilate(mogImg, mogImg, Mat());

        //detect white area
//        if(!isDetected){
//            if(labelling(knnImg, boundBox, 800)){
//                isDetected = true;
//            }
//        }
//        if(searchMotion(knnImg, dstImg, 800)){
//            isDetected = true;
//        }


        writer << dstImg;
        imshow("Tracking", dstImg);
        imshow("KNN", knnImg);
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

double clacLength(Point2f p1, Point2f p2)
{
    double x1, y1, x2, y2;
    double length = pow( (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y), 0.5 );
    return length;
}

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
            double length = clacLength(prevCorners[i], currCorners[i]);
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


int pano_test(){

    int camNum = 4;
    VideoCapture vcap[4];
    string camAddress[4];
    camAddress[0] = "rtsp://192.168.5.103:554/s1";
    camAddress[1] = "rtsp://192.168.5.7:554/s1";
    camAddress[2] = "rtsp://192.168.5.118:554/s1";
    camAddress[3] = "rtsp://192.168.5.8:554/s1";

    bool try_use_gpu = true;
    Stitcher::Mode mode = Stitcher::PANORAMA;
    Stitcher stitcher = Stitcher::createDefault(try_use_gpu);

    stitcher.setWarper(new CylindricalWarperGpu());
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
//    vector<Mat> imgs(4);
    vector<cuda::GpuMat> imgs(4);
//    vector<UMat> imgs(4);
    Mat frame[4];
    string window_name[4];

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
            imgs[i].upload(frame[i]);
//            imgs[i] = frame[i];
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
    for(int i=0; i<4; i++){
        string filename = strsprintf("cam%d.jpg", i+1);
        imwrite(filename, imgs[i]);
        vcap[i].release();
    }


//    Stitcher::Status status = stitcher.stitch(imgs, panoImg);
//    Stitcher::Status status0 = stitcher.estimateTransform(imgs);
//    Stitcher::Status status = stitcher.composePanorama(imgs, panoImgGpu);
//    if (status != Stitcher::OK)
//    {
//        cout << "Can't stitch images, error code = " << int(status) << endl;
//        return -1;
//    }
//    panoImgGpu.download(panoImg);
//    imwrite("pano.jpg", panoImg);
//    imwrite("pano.jpg", panoImgGpu);

    return 0;
}

int pano_test2(){

    int camNum = 3;
    VideoCapture vcap[3];
    string camAddress[3];
    camAddress[0] = "rtsp://192.168.5.103:554/s1";
    camAddress[1] = "rtsp://192.168.5.7:554/s1";
    camAddress[2] = "rtsp://192.168.5.118:554/s1";
//    camAddress[3] = "rtsp://192.168.5.8:554/s1";


    Mat frame[3];
    string window_name[3];
    bool isVideo = false;

    if(g_isBebug){
        for(int i=0; i<camNum; i++){
            window_name[i] = strsprintf("camNo%d", i+1);
            if(!isVideo){
                camAddress[i] = strsprintf("./images/%d.jpg", i+1);
                frame[i] = imread(camAddress[i]);
            }else{
                for(int i=0; i<camNum; i++){
                    camAddress[i] = strsprintf("./Videos/s_cam%d.avi", i+1);
                    if (!vcap[i].open(camAddress[i])) {
                        cout << "error opening camera stream ...." << i << endl;
                        return -1;
                    }else{
                        if(!vcap[i].read(frame[i])){
                            return -1;
                        }
                    }
                }
            }
        }
        while (1) {
            for(int i=0; i<camNum; i++){
                imshow(window_name[i], frame[i]);
            }

            if(waitKey(10) == 27){
                break;
            }
        }

    }else{

        for(int i=0; i<camNum; i++){
            if (!vcap[i].open(camAddress[i])) {
                cout << "error opening camera stream ...." << i << endl;
                return -1;
            }
            window_name[i] = strsprintf("camNo%d", i+1);
        }


        bool is_first = false;
        while (1) {
            for(int i=0; i<camNum; i++){
                if(!vcap[i].read(frame[i])){
                    cout << "capture error" << endl;
                    break;
                }
                imshow(window_name[i], frame[i]);
            }

            if(waitKey(10) == 27){
                break;
            }
        }
        for(int i=0; i<3; i++){
            string filename = strsprintf("ssscam%d.jpg", i+1);
            imwrite(filename, frame[i]);
            vcap[i].release();
        }
    }

    destroyAllWindows();
    stich3(frame);
    return 0;
}


bool hFromRansac( Mat &image1, Mat &image2, Mat &homography)
{
    int minHessian = 400;

    Ptr<Feature2D> detector;
    int detectorType = 1;
    switch (detectorType) {
        case 0:
        default:
            detector = xfeatures2d::SURF::create();
            break;
        case 1:
            detector = xfeatures2d::SIFT::create();
            break;
        case 2:
            detector = ORB::create();
            break;
        case 3:
            detector = AKAZE::create();
            break;
    }


    //Detect and compute
    vector<cv::KeyPoint> keys1, keys2;
    Mat desc1, desc2;
    detector->detectAndCompute(image1, noArray(), keys1, desc1);
    detector->detectAndCompute(image2, noArray(), keys2, desc2);

    auto matchtype = detector->defaultNorm(); // SIFT, SURF: NORM_L2
                                                // BRISK, ORB, KAZE, A-KAZE: NORM_HAMMING
    BFMatcher matcher(matchtype);
    vector<vector<DMatch >> knn_matches;

    // 上位2点
    matcher.knnMatch(desc1, desc2, knn_matches, 2);


    // 対応点を絞る
//    const auto match_par = .6f; //対応点のしきい値
    const auto match_par = .6f; //対応点のしきい値
    std::vector<cv::DMatch> good_matches;

    std::vector<cv::Point2f> match_point1;
    std::vector<cv::Point2f> match_point2;

    for (size_t i = 0; i < knn_matches.size(); ++i) {
        auto dist1 = knn_matches[i][0].distance;
        auto dist2 = knn_matches[i][1].distance;

        //良い点を残す（最も類似する点と次に類似する点の類似度から）
        if (dist1 <= dist2 * match_par) {
            good_matches.push_back(knn_matches[i][0]);
            match_point1.push_back(keys1[knn_matches[i][0].queryIdx].pt);
            match_point2.push_back(keys2[knn_matches[i][0].trainIdx].pt);
        }
    }
    if(match_point1.size() >= 4){
//        homography = findHomography( match_point1, match_point2, CV_RANSAC );
        Mat masks;
        homography = findHomography( match_point1, match_point2, masks, RANSAC, 3.f);

        //RANSACで使われた対応点のみ抽出
        vector<DMatch> inlierMatches;
        for (auto i = 0; i < masks.rows; ++i) {
            uchar *inlier = masks.ptr<uchar>(i);
            if (inlier[0] == 1) {
                inlierMatches.push_back(good_matches[i]);
            }
        }
        //特徴点の表示
        Mat dst1, dst2;
        drawMatches(image1, keys1, image2, keys2, good_matches, dst1);
        drawMatches(image1, keys1, image2, keys2, inlierMatches, dst2);
        imshow( "matches1", dst1 );
        imshow( "matches2", dst2 );

        while(1){
            if(waitKey(10) == 27){
                break;
            }
        }
        destroyAllWindows();
        return true;
    }

    // 特徴点の対応付け
//    vector< DMatch > matches;
//    FlannBasedMatcher matcher;
//    matcher.match( desp1, desp2, matches );

//    double max_dist = 0; double min_dist = 100;
//    for( int i = 0; i < desp1.rows; i++ ){
//        double dist = matches[i].distance;
//        if( dist < min_dist ) min_dist = dist;
//        if( dist > max_dist ) max_dist = dist;
//    }

//    std::vector< DMatch > good_matches;

//    for( int i = 0; i < desp1.rows; i++ ){
//    if( matches[i].distance < 4*min_dist ){
//        good_matches.push_back( matches[i]); }
//    }

//    vector<Point2f> obj;
//    vector<Point2f> scene;
//    for( int i = 0; i < good_matches.size(); i++ ){
//        obj.push_back( keys1[ good_matches[i].queryIdx ].pt );
//        scene.push_back( keys2[ good_matches[i].trainIdx ].pt );
//    }

//    if(obj.size() >= 4){
//        homography = findHomography( obj, scene, CV_RANSAC );
//        return true;
//    }

    return false;
}



int stich3(Mat (&srcImg)[3]){
    // Gray_scale
    Mat grayImg[3];
    for(int i=0; i<3; i++){
        cvtColor(srcImg[i], grayImg[i], COLOR_BGR2GRAY);
    }

    //1と2
    Mat H12;
    if(!hFromRansac( grayImg[1], grayImg[0], H12)){
        cout << "homograpy error H12" << endl;
        return -1;
    }

    Mat result1;
    Size newSize = getDistSize(H12, grayImg[0]);
    newSize.width = newSize.width*0.98;
    warpPerspective( grayImg[1], result1, H12, newSize);
//    warpPerspective( grayImg[1], result1, H12, Size( grayImg[0].cols * 1.5 , grayImg[0].rows ) );
    for( int y = 0 ; y < grayImg[0].rows ; ++y ){
        for( int x = 0 ; x < grayImg[0].cols ; ++x ){
            result1.at<uchar>( y, x ) = grayImg[0].at<uchar>( y, x );
        }
    }

    //2と3
    Mat H23;
//    if(!hFromRansac( grayImg[2], grayImg[1], H23 )){
    if(!hFromRansac( grayImg[2], result1, H23 )){
        cout << "homograpy error H23" << endl;
        return -1;
    }
    Mat result2;
    Size ssSize = getDistSize(H23, result1);
    warpPerspective( grayImg[2], result2, H23, ssSize );

//    Mat H123 = H23*H12;
//    Size ssSize = getDistSize(H123, result1);
//////    Size sSize = getDistSize(H23, grayImg[1]);
//////    Size ssSize;
//////    ssSize.width = sSize.width+result1.cols;
//////    ssSize.height = sSize.height;
//////    ssSize.width = ssSize.width*0.98;
//    warpPerspective( grayImg[2], result2, H123, ssSize, INTER_CUBIC );
////    warpPerspective( grayImg[2], result2, H23*H12, Size( result1.cols * 1.5 , result1.rows ) );
    for( int y = 0 ; y < result1.rows ; y++ ){
        for( int x = 0 ; x < result1.cols ; x++ ){
            result2.at<uchar>( y, x ) = result1.at<uchar>( y, x );
        }
    }

//    //画像を反転して元に戻す
//    flip( result2, result2, 1 );


//    //画像を左右反転する
//    flip( grayImg[1], grayImg[1], 1 );
//    flip( grayImg[2], grayImg[2], 1 );
//    flip( result1, result1, 1 );

//    //2と3
//    Mat H23 = hFromRansac( grayImg[2], grayImg[1] );
//    Mat result2;
//    warpPerspective( grayImg[2], result2, H23, Size( result1.cols * 1.5 , result1.rows ) );

//    //右に移動させる
//    for ( int y = result2.rows -1; y >= 0; y-- ){
//        for ( int x = result2.cols -1; x >= 0; x-- ){
//            int dx = 1.25*grayImg[2].cols;
//            if ( grayImg[2].cols <= x +dx ){
//                continue;
//            }
//            result2.at<uchar>( y, x + dx ) = result2.at<uchar>( y , x);
//            result2.at<uchar>( y , x) = 0;
//        }
//    }

//    for( int y = 0 ; y < result1.rows ; y++ ){
//        for( int x = 0 ; x < result1.cols ; x++ ){
//            result2.at<uchar>( y, x ) = result1.at<uchar>( y, x );
//        }
//    }

//    //画像を反転して元に戻す
//    flip( result2, result2, 1 );

    //画像を表示する
    imshow( "Mosaicing", result2 );
    waitKey();
    imwrite("./Result/Mosaicing.jpg", result2);


    return 0;
}

Size getDistSize(Mat &H, Mat &srcImg)
{
    vector<cv::Point2f> obj_corner;
    obj_corner.push_back(cv::Point2f(0, 0));
    obj_corner.push_back(cv::Point2f(srcImg.cols, 0));
    obj_corner.push_back(cv::Point2f(srcImg.cols, srcImg.rows));
    obj_corner.push_back(cv::Point2f(0, srcImg.rows));

    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corner,scene_corners,H);
    int maxCols(0),maxRows(0);

    for(int i=0;i<scene_corners.size();i++){
      if(maxRows < scene_corners.at(i).y)
           maxRows = scene_corners.at(i).y;
      if(maxCols < scene_corners.at(i).x)
           maxCols = scene_corners.at(i).x;
    }
    maxRows = srcImg.rows;
//    if(maxRows > srcImg.rows*2){
//        maxRows = srcImg.rows*1.5;
//    }

    if(maxCols > srcImg.cols*2){
        maxCols = srcImg.cols*2;
    }
    if(maxCols < srcImg.cols){
        maxCols = srcImg.cols*2;
    }
    return Size(maxCols, maxRows);
}

int stich5(){
    // 画像読み込み（グレースケールで読み込み）
    Mat image1 = imread( "./images/S6.jpg",  CV_LOAD_IMAGE_GRAYSCALE );
    Mat image2 = imread( "./images/S5.jpg",  CV_LOAD_IMAGE_GRAYSCALE );
    Mat image3 = imread( "./images/S3.jpg",  CV_LOAD_IMAGE_GRAYSCALE );
    Mat image4 = imread( "./images/S2.jpg",  CV_LOAD_IMAGE_GRAYSCALE );
    Mat image5 = imread( "./images/S1.jpg",  CV_LOAD_IMAGE_GRAYSCALE );

    double stichR = 1.5;

    //２と３
    Mat H23;
    hFromRansac( image2, image3, H23);
    Mat result1;
    Size newSize = getDistSize(H23, image3);

    warpPerspective( image2, result1, H23, Size( image3.cols * stichR , image3.rows ) );
//    warpPerspective( image2, result1, H23, newSize);
    for( int y = 0 ; y < image3.rows ; y++ ){
        for( int x = 0 ; x < image3.cols ; x++ ){
            result1.at<uchar>( y, x ) = image3.at<uchar>( y, x );
        }
      }

    //１と２
    Mat H12;
    hFromRansac( image1, image2, H12);
    Mat result2;
    Mat H123 = H12 * H23;
    newSize = getDistSize(H123, result1);
    warpPerspective( image1, result2, H12 * H23, Size( result1.cols * 1.5 , result1.rows ) );
//    warpPerspective( image1, result2, H12 * H23, newSize );
    for( int y = 0 ; y < result1.rows ; ++y ){
        for( int x = 0 ; x < result1.cols ; ++x ){
            result2.at<uchar>( y, x ) = result1.at<uchar>( y, x );
        }
    }

    //画像を左右反転する
    flip( image3, image3, 1 );
    flip( image4, image4, 1 );
    flip( result2, result2, 1 );

    //３と４
    Mat H34;
    hFromRansac( image4, image3, H34 );
    Mat result3;
//    Mat H1234 = H123 * H34;
//    newSize = getDistSize(H34, result2);
    warpPerspective( image4, result3, H34, Size( result2.cols * stichR , result2.rows ) );
//    warpPerspective( image4, result3, H34, newSize );

    //右に移動させる
    for ( int y = result3.rows -1; y >= 0; y-- ){
        for ( int x = result3.cols -1; x >= 0; x-- ){
            int dx = 1.25*image4.cols;
            if ( result3.cols <= x +dx ){
                continue;
            }
            result3.at<uchar>( y, x + dx ) = result3.at<uchar>( y , x);
            result3.at<uchar>( y , x) = 0;
        }
    }

    for( int y = 0 ; y < result2.rows ; y++ ){
        for( int x = 0 ; x < result2.cols ; x++ ){
            result3.at<uchar>( y, x ) = result2.at<uchar>( y, x );
        }
    }

    //画像を左右反転する
    flip( image5, image5, 1 );
    Mat H45;
    hFromRansac( image5, image4, H45 );

    //４と５
    Mat result4;
//    Mat H12345 = H1234 * H45;
//    newSize = getDistSize(H45, result3);
    warpPerspective( image5, result4, H34*H45, Size( result3.cols * stichR , result3.rows ) );
//    warpPerspective( image5, result4, H34*H45, newSize );

    //右に移動させる
    for ( int y = result4.rows -1; y > 0; y-- ){
        for ( int x = result4.cols -1; x > 0; x-- ){
            int dx = (1+0.5*0.5)*image5.cols;
            if ( result4.cols <= x +dx ){
                continue;
            }
            result4.at<uchar>( y, x + dx ) = result4.at<uchar>( y , x);
            result4.at<uchar>( y , x) = 0;
        }
    }

    for( int y = 0 ; y < result3.rows ; y++ ){
        for( int x = 0 ; x < result3.cols ; x++ ){
            if ( (int)result3.at<uchar>( y, x ) != 0 ){
                result4.at<uchar>( y, x ) = result3.at<uchar>( y, x );
            }
        }
      }

    //画像を反転して元に戻す
    flip( result4, result4, 1 );

    //画像を表示する
    imshow( "Mosaicing", result4 );
    waitKey();

    //画像を保存する
    imwrite( "pano1.jpg", result1 );
    imwrite( "pano2.jpg", result2 );
    imwrite( "pano3.jpg", result3 );
    imwrite( "pano4.jpg", result4 );

    return 0;
}


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



int main(int argc, char *argv[])
{
//    cudaTest();
    g_isBebug = true;
//    pano_test();
    pano_test2();
//    stich5();
//    trackVehicle();
//    optFlowTracking();
    return 0;
}
