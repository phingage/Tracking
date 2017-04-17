#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;


//class Classifier:public std::enable_shared_from_this<caffe::Net<float>>
class Classifier
{
public:
//    Classifier();
    Classifier(const std::string& model_file,
               const std::string& trained_file,
               const std::string& mean_file,
               const std::string& label_file,
               const bool& use_gpu );
    ~Classifier();

    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
    void SetMean(const std::string& mean_file);
    std::vector<float> Predict(const cv::Mat& img);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs);
    std::vector<int> Argmax(const std::vector<float>& v, int N);

private:
//    std::unique_ptr<caffe::Net<float> > net_;
//    std::shared_ptr<caffe::Net<float> > net_;
//    std::weak_ptr<caffe::Net<float> > net_;
    caffe::Net<float>* net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<std::string> labels_;
};

#endif // CLASSIFIER_H
