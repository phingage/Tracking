#include "panorama.h"

Panorama::Panorama()
{
    m_isGpu = false;
    m_siftWidth = 0;
}




bool Panorama::hFromRansac( cv::Mat &image1, cv::Mat &image2, cv::Mat &homography)
{
//    int minHessian = 400;

    cv::Ptr<cv::Feature2D> detector;
    int detectorType = 1;
    switch (detectorType) {
        case 0:
        default:
            detector = cv::xfeatures2d::SURF::create();
            break;
        case 1:
            detector = cv::xfeatures2d::SIFT::create();
            break;
        case 2:
            detector = cv::ORB::create();
            break;
        case 3:
            detector = cv::AKAZE::create();
            break;
    }


    //Detect and compute
    std::vector<cv::KeyPoint> keys1, keys2;
    cv::Mat desc1, desc2;
    detector->detectAndCompute(image1, cv::noArray(), keys1, desc1);
    detector->detectAndCompute(image2, cv::noArray(), keys2, desc2);

    auto matchtype = detector->defaultNorm(); // SIFT, SURF: NORM_L2
                                                // BRISK, ORB, KAZE, A-KAZE: NORM_HAMMING
    cv::BFMatcher matcher(matchtype);
    std::vector<std::vector<cv::DMatch>> knn_matches;

    // 上位2点
    matcher.knnMatch(desc1, desc2, knn_matches, 2);


    // 対応点を絞る
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
        cv::Mat masks;
        homography = cv::findHomography( match_point1, match_point2, masks, cv::RANSAC, 3.f);

        //RANSACで使われた対応点のみ抽出
        std::vector<cv::DMatch> inlierMatches;
        for (auto i = 0; i < masks.rows; ++i) {
            uchar *inlier = masks.ptr<uchar>(i);
            if (inlier[0] == 1) {
                inlierMatches.push_back(good_matches[i]);
            }
        }
        //特徴点の表示
        cv::Mat dst1, dst2;
        cv::drawMatches(image1, keys1, image2, keys2, good_matches, dst1);
        cv::drawMatches(image1, keys1, image2, keys2, inlierMatches, dst2);
        cv::imshow( "matches1", dst1 );
        cv::imshow( "matches2", dst2 );

        while(1){
            if(cv::waitKey(10) == 27){
                break;
            }
        }
        cv::destroyAllWindows();
        return true;
    }

    return false;
}

cv::Size Panorama::getDistSize(cv::Mat &H, cv::Mat &srcImg)
{
    std::vector<cv::Point2f> obj_corner;
    obj_corner.push_back(cv::Point2f(0, 0));
    obj_corner.push_back(cv::Point2f(srcImg.cols, 0));
    obj_corner.push_back(cv::Point2f(srcImg.cols, srcImg.rows));
    obj_corner.push_back(cv::Point2f(0, srcImg.rows));

    std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform(obj_corner,scene_corners,H);
    int maxCols(0),maxRows(0);

    for(size_t i=0;i<scene_corners.size();i++){
      if(maxRows < scene_corners[i].y)
           maxRows = scene_corners[i].y;
      if(maxCols < scene_corners[i].x)
           maxCols = scene_corners[i].x;
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
    return cv::Size(maxCols, maxRows);
}

void Panorama::setGpu(bool enable)
{
    m_isGpu = enable;
}

int Panorama::estimateAndCompose(std::vector<cv::Mat> &imgs, cv::Mat &result)
{
    int size = imgs.size();
    m_nRight = size/2;
    // Even or Odd
    if( size % 2 == 0 ){
        m_nLeft = m_nRight -1;
    } else {
        m_nLeft = m_nRight;
    }


    //Right Images Stitch
    std::vector<cv::Mat> rightStitch(m_nRight+1);
    m_rightHomo.resize(m_nRight);
    m_rightSize.resize(m_nRight);
    if(m_isGpu){
        m_rightHomoGPU.resize(m_nRight);
    }
    cv::Mat H_right;
    for(int i=0; i<m_nRight; i++){
        cv::Mat H;
        if(!hFromRansac( imgs[m_nLeft+1+i], imgs[m_nLeft+i], H)){
            std::cout << "Homograpy error" << ":right=" << i << std::endl;
            return -1;
        }
        if(i==0){
            rightStitch[i] = imgs[m_nLeft];
            H_right = H;
        }else{
            H_right *= H;
        }
        //Set Homography and dstSize
        cv::Size size = getDistSize(H_right, rightStitch[i]);
//        size.width = size.width*0.99;
        m_rightSize[i] = size;
        H_right.copyTo(m_rightHomo[i]); //have to use copyTo

        cv::warpPerspective( imgs[m_nRight+i], rightStitch[i+1], H_right, size );
        // ROI copy
        cv::Mat img_roi(rightStitch[i+1], cv::Rect(0, 0, rightStitch[i].cols, rightStitch[i].rows));
        rightStitch[i].copyTo(img_roi);

        if(m_isGpu){
            m_rightHomoGPU[i].upload(m_rightHomo[i]);
        }
    }


    //Left Images Homography & Size
    //Flip
    for(int i=0; i<m_nLeft+1; i++){
        cv::flip( imgs[i], imgs[i], 1 );
    }
    std::vector<cv::Mat> leftStitch(m_nLeft+1);
    m_leftHomo.resize(m_nLeft);
    m_leftSize.resize(m_nLeft);
    if(m_isGpu){
        m_leftHomoGPU.resize(m_nLeft);
    }
    cv::Mat H_left;
    for(int i=0; i<m_nLeft; i++){
        cv::Mat H;
        if (!hFromRansac( imgs[i], imgs[i+1], H )){
            std::cout << "Homograpy error" << ":left=" << i << std::endl;
            return -1;
        }

        if(i==0){
            leftStitch[i] = imgs[m_nLeft];
            H_left = H;
        }else{
            H_left *= H;
        }

        cv::Size size = getDistSize(H_left, leftStitch[i]);
//        size.width = size.width*0.99;
        m_leftSize[i] = size;
        H_left.copyTo(m_leftHomo[i]); //have to use copyTo

        cv::warpPerspective( imgs[i], leftStitch[i+1], H_left, size );
        if(i!=0){
            // ROI copy
            cv::Mat img_roi(leftStitch[i+1], cv::Rect(0, 0, leftStitch[i].cols, leftStitch[i].rows));
            leftStitch[i].copyTo(img_roi);
        }
        if(m_isGpu){
            m_leftHomoGPU[i].upload(m_leftHomo[i]);
        }

    }
    cv::flip(leftStitch[m_nLeft], leftStitch[m_nLeft], 1);


    // calc Dist size
    cv::Size dstSize;
    dstSize.height = m_leftSize[0].height;
    dstSize.width = leftStitch[m_nLeft].cols + rightStitch[m_nRight].cols - imgs[m_nLeft].cols;
    m_dstImg = cv::Mat::zeros(dstSize, imgs[0].type() ); //CV_8U/ CV_8UC3
    if(m_isGpu){
//        m_dstImgGPU = cv::cuda::GpuMat(dstSize, CV_32FC3);
        m_dstImgGPU = cv::cuda::GpuMat(dstSize, imgs[0].type());
    }
    m_siftWidth = leftStitch[m_nLeft].cols - imgs[m_nLeft].cols;


    // ROI
    cv::Mat imgLeft(m_dstImg, cv::Rect(0, 0, leftStitch[m_nLeft].cols, leftStitch[m_nLeft].rows));
    leftStitch[m_nLeft].copyTo(imgLeft);

    cv::Mat imgRight(m_dstImg, cv::Rect(m_siftWidth, 0, rightStitch[m_nRight].cols, rightStitch[m_nRight].rows));   // 元画像のROIを生成
    rightStitch[m_nRight].copyTo(imgRight);   // 貼り付ける画像をROIにコピー


    //画像を表示する
    m_dstImg.copyTo(result);
//    cv::imshow( "Mosaicing", m_dstImg );
//    cv::imwrite( "Mosa.jpg", m_dstImg );
//    cv::imshow( "left", leftStitch[m_nLeft] );
//    cv::imshow( "right", rightStitch[m_nRight] );
//    cv::waitKey();


    return 0;
}


int Panorama::composePanorama(std::vector<cv::Mat> &imgs, cv::Mat &result)
{
    int im_size = imgs.size();
    if( im_size < 2){
        return -1;
    }

    if(m_isGpu){
//        cv::cuda::GpuMat gpu_imgs(im_size);
        std::vector<cv::cuda::GpuMat> gpu_imgs(im_size);
        for(int i=0; i<im_size; i++){
            gpu_imgs[i].upload(imgs[i]);
        }

        //Right Images Stitch
        std::vector<cv::cuda::GpuMat> rightStitch(m_nRight+1);
        for(int i=0; i<m_nRight; i++){
            if(i==0){
                rightStitch[i] = gpu_imgs[m_nLeft];
            }
            cv::cuda::warpPerspective( gpu_imgs[m_nRight+i], rightStitch[i+1], m_rightHomoGPU[i], m_rightSize[i] );
            // ROI copy
            cv::cuda::GpuMat img_roi(rightStitch[i+1], cv::Rect(0, 0, rightStitch[i].cols, rightStitch[i].rows));
            rightStitch[i].copyTo(img_roi);
        }

        //Flip
        for(int i=0; i<m_nLeft+1; i++){
            cv::cuda::flip(gpu_imgs[i], gpu_imgs[i], 1);
        }
        std::vector<cv::cuda::GpuMat> leftStitch(m_nLeft+1);
        for(int i=0; i<m_nLeft; i++){
            if(i==0){
                leftStitch[i] = gpu_imgs[m_nLeft];
            }
            cv::cuda::warpPerspective( gpu_imgs[i], leftStitch[i+1], m_leftHomo[i], m_leftSize[i] );
            if(i!=0){
                // ROI copy
                cv::cuda::GpuMat img_roi(leftStitch[i+1], cv::Rect(0, 0, leftStitch[i].cols, leftStitch[i].rows));
                leftStitch[i].copyTo(img_roi);
            }
        }
        cv::cuda::flip(leftStitch[m_nLeft], leftStitch[m_nLeft], 1);


        // Make Dist Img
        cv::cuda::GpuMat  imgLeft(m_dstImgGPU, cv::Rect(0, 0, leftStitch[m_nLeft].cols, leftStitch[m_nLeft].rows));
        leftStitch[m_nLeft].copyTo(imgLeft);

        cv::cuda::GpuMat imgRight(m_dstImgGPU, cv::Rect(m_siftWidth, 0, rightStitch[m_nRight].cols, rightStitch[m_nRight].rows));
        rightStitch[m_nRight].copyTo(imgRight);

        //Copy To
        m_dstImgGPU.download(result);


    }else{
        //Right Images Stitch
        std::vector<cv::Mat> rightStitch(m_nRight+1);
        for(int i=0; i<m_nRight; i++){
            if(i==0){
                rightStitch[i] = imgs[m_nLeft];
            }
            cv::warpPerspective( imgs[m_nRight+i], rightStitch[i+1], m_rightHomo[i], m_rightSize[i] );
            // ROI copy
            cv::Mat img_roi(rightStitch[i+1], cv::Rect(0, 0, rightStitch[i].cols, rightStitch[i].rows));
            rightStitch[i].copyTo(img_roi);
        }

        //Flip
        for(int i=0; i<m_nLeft+1; i++){
            cv::flip( imgs[i], imgs[i], 1 );
        }
        std::vector<cv::Mat> leftStitch(m_nLeft+1);
        for(int i=0; i<m_nLeft; i++){
            if(i==0){
                leftStitch[i] = imgs[m_nLeft];
            }
            cv::warpPerspective( imgs[i], leftStitch[i+1], m_leftHomo[i], m_leftSize[i] );
            if(i!=0){
                // ROI copy
                cv::Mat img_roi(leftStitch[i+1], cv::Rect(0, 0, leftStitch[i].cols, leftStitch[i].rows));
                leftStitch[i].copyTo(img_roi);
            }
        }
        cv::flip(leftStitch[m_nLeft], leftStitch[m_nLeft], 1);


        // Make Dist Img
        cv::Mat imgLeft(m_dstImg, cv::Rect(0, 0, leftStitch[m_nLeft].cols, leftStitch[m_nLeft].rows));
        leftStitch[m_nLeft].copyTo(imgLeft);

        cv::Mat imgRight(m_dstImg, cv::Rect(m_siftWidth, 0, rightStitch[m_nRight].cols, rightStitch[m_nRight].rows));
        rightStitch[m_nRight].copyTo(imgRight);

        //Copy To
        m_dstImg.copyTo(result);


    }


    return 0;
}

