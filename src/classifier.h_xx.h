#ifndef CLASSIFIER_H
#define CLASSIFIER_H
//#define CPU_ONLY
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
using namespace std;
using namespace caffe;
using namespace cv;

typedef std::pair<string, float> Prediction;

class Classifier
{
public:
      Classifier(const string& model_file,
                 const string& trained_file,
                 const string& mean_file,
                 const string& label_file);
      std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
private:
      void SetMean(const string& mean_file);
      std::vector<float> Predict(const cv::Mat& img);
      void WrapInputLayer(std::vector<cv::Mat>* input_channels);
      void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
private:
      caffe::shared_ptr<Net<float> > net_;
      cv::Size input_geometry_;
      int num_channels_;
      cv::Mat mean_;
      std::vector<string> labels_;

};

#endif // CLASSIFIER_H
