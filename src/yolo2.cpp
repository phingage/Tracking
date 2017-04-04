#include "yolo2.hpp"


//Constractor
Yolo2::Yolo2(){

}


Yolo2::Yolo2(std::string &namefile, std::string &cfgfile, std::string &weightsfile)
{
    readFile(namefile);
    char* c_cfgfile = new char[cfgfile.size() + 1];
    strcpy(c_cfgfile, cfgfile.c_str());
    char* c_weightsfile = new char[weightsfile.size() + 1];
    strcpy(c_weightsfile, weightsfile.c_str());
    //call
    initParam(c_cfgfile, c_weightsfile);
    //delete
    delete [] c_cfgfile;
    delete [] c_weightsfile;
}

//Destructor
Yolo2::~Yolo2()
{
    releaseYololib();
}


bool Yolo2::readFile(std::string &filePath)
{
    std::ifstream ifs(filePath);
    std::string str;
    if (ifs.fail())
    {
        std::cerr << "name file read error" << std::endl;
        return false;
    }
    while (getline(ifs, str))
    {
        std::cout << "[" << str << "]" << std::endl;
        m_names.push_back(str);
    }
    return true;
}


bool Yolo2::detect(cv::Mat &srcImg, std::vector<tagDetected> &detected, float thresh, float nms)
{
    image img = mat2YoloImage(srcImg);

    int maxNum = 100;
    box *boxes = new box[maxNum];
    int *label = new int[maxNum];
    float *accuracy = new float[maxNum];
    thresh = 0.25;
    nms = 0.4;

    cv::TickMeter tick;
    tick.start();
    int dtcNum = detectObject(&img, boxes, label, accuracy, maxNum, thresh, nms);
    tick.stop();
    std::cout << "fuction time=" << tick.getTimeMilli() << std::endl;


    if(dtcNum==0) return false;

    tagDetected tempTag;
    for(int i=0; i<dtcNum; i++){
        tempTag.box.x = boxes[i].x;
        tempTag.box.y = boxes[i].y;
        tempTag.box.width = boxes[i].w;
        tempTag.box.height = boxes[i].h;
        tempTag.label = label[i];
        tempTag.accuracy = accuracy[i];
        detected.push_back(tempTag);
//        cv::rectangle(srcImg, tempTag.box, cv::Scalar(200,0,200), 3, 1);
    }

    delete[] boxes;
    delete[] label;
    delete[] accuracy;

    return true;
}

image Yolo2::mat2YoloImage(cv::Mat &src)
{
  unsigned char *data = (unsigned char *)src.data;
  int h = src.rows;
  int w = src.cols;
  int c = src.channels();
  int step = src.step1();
  image out = make_image(w, h, c);
  int i, j, k, count = 0;
  for (k = 0; k < c; ++k)
  {
      for (i = 0; i < h; ++i)
      {
          for (j = 0; j < w; ++j)
          {out.data[count++] = data[i * step + j * c + k] / 255.;}
      }
  }
  return out;
}

