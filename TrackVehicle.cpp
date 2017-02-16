// TrackVehicle.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/tracking/tracker.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/optflow.hpp"
//#include "opencv2/cudaimgproc.hpp"
//#include "opencv2/cudaobjdetect.hpp"
//#include "../include/json11.hpp"
//#include "../include/util.h"
//#include "../include/fixedqueue.h"


bool g_isBebug = false;


using namespace std;
using namespace cv;


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
	else {
		CV_Error(Error::StsError, "Unsupported algorithm type " + trackerAlgo + " is specified.");
	}
	return true;
}

bool labelling(Mat &grayImg, Rect2d &dtcRect, int minArea = 100)
{
	Mat binImg;
	threshold(grayImg, binImg, 0, 255, THRESH_BINARY | THRESH_OTSU);

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

		if (x <= padding || y <= padding || x + width >= binImg.cols - padding || y + height >= binImg.rows - padding) {
			continue;
		}

		if (area > minArea) {
			dtcRect.x = (double)x;
			dtcRect.y = (double)y;
			dtcRect.width = (double)width;
			dtcRect.height = (double)height;
			isDetect = true;
		}
	}
	return isDetect;
}


bool labelling2(Mat &grayImg, Mat &dstImg, int minArea = 100)
{
	Mat binImg;
	threshold(grayImg, binImg, 0, 255, THRESH_BINARY | THRESH_OTSU);

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

		//        if(x<=padding || y <= padding || x+width >= binImg.cols-padding || y+height >= binImg.rows-padding){
		//            continue;
		//        }

		if (area > minArea) {
			Rect boundBox;
			boundBox.x = x;
			boundBox.y = y;
			boundBox.width = width;
			boundBox.height = height;
			rectangle(dstImg, boundBox, Scalar(200, 0, 200), 2);
			isDetect = true;
		}
	}
	return isDetect;
}



int trackVehicle() {

	VideoCapture vcap;
	string camAddress;
	if (g_isBebug) {
		camAddress = "../data/test1.avi";
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
	Rect2d boundBox;
	bool isDetected = false;
	bool isTrackInt = false;

	//int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
	//double fps = 30.0;
	//bool isColor = true;
	//VideoWriter writer("test.avi", fourcc, fps, Size(640, 360), isColor);


	while (1) {
		if (!vcap.read(frame)) {
			cout << "capture error" << endl;
			break;
		}

		frame.copyTo(dstImg);
		//        if(isDetected){
		//            if(!isTrackInt){
		//                tracker->init(frame, boundBox);
		//                isTrackInt = true;
		//            }else{
		//                if(tracker->update(frame, boundBox)){
		//                    rectangle(dstImg, boundBox, Scalar(200,0,200), 2);
		//                }else{
		//                    isDetected = false;
		//                    isTrackInt = false;
		//                }
		//            }
		//        }


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
		if(labelling2(knnImg, dstImg, 800)){
		    isDetected = true;
		}


		//writer << dstImg;
		imshow("Tracking", dstImg);
		imshow("KNN", knnImg);
		//        imshow("MOG2", mogImg);
		//get the input from the keyboard
		if (waitKey(10) == 27) {
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
	double length = pow((p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y), 0.5);
	return length;
}

int optFlowTracking(void)
{
	VideoCapture vcap;
	string camAddress = "rtsp://192.168.5.7:554/s1";
	if (!vcap.open(camAddress)) {
		cout << "error opening camera stream ...." << endl;
		return -1;
	}

	Mat frame;
	if (!vcap.read(frame)) {
		cout << "capture error" << endl;
		return -1;
	}

	int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');
	double fps = 30.0;
	bool isColor = true;
	VideoWriter writer("test.avi", fourcc, fps, Size(640, 360), isColor);

	Mat prevImg, currImg;
	vector<Point2f> prevCorners, currCorners;
	vector<uchar> featuresFound;
	vector<float> featuresErrors;
	cvtColor(frame, prevImg, COLOR_BGR2GRAY);
	goodFeaturesToTrack(prevImg, prevCorners, 100, 0.3, 7);
	cornerSubPix(prevImg, prevCorners, Size(21, 21), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.01));

	while (1) {
		if (!vcap.read(frame)) {
			cout << "capture error" << endl;
			return -1;
		}
		cvtColor(frame, currImg, COLOR_BGR2GRAY);
		goodFeaturesToTrack(prevImg, currCorners, 100, 0.3, 7);
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
			if (length > 5) {
				Point p1 = Point((int)prevCorners[i].x, (int)prevCorners[i].y);
				Point p2 = Point((int)currCorners[i].x, (int)currCorners[i].y);
				line(frame, p1, p2, Scalar(0, 0, 255), 2);
			}
		}

		currImg.copyTo(prevImg);
		prevCorners.resize(currCorners.size());
		copy(currCorners.begin(), currCorners.end(), prevCorners.begin());

		imshow("preview", frame);
		writer << frame;
		if (waitKey(10) == 27) {
			destroyAllWindows();
			break;
		}
	}
}

int main()
{
	g_isBebug = true;
	trackVehicle();
	//    optFlowTracking();
	return 0;
}

