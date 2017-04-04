TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += src/main.cpp \
    src/util.cpp \
    src/json11.cpp \
    src/panorama.cpp \
    src/yolo2.cpp

TARGET = npd

INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /home/shiomi/work/beanstalk-client
INCLUDEPATH += /usr/local/cuda-8.0/include
INCLUDEPATH += /home/shiomi/work/caffe/build/install/include
INCLUDEPATH += /home/shiomi/work/GitHub/yolo/my_darknet/src/

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_objdetect -lopencv_tracking -lopencv_stitching -lopencv_calib3d -lopencv_features2d -lopencv_flann
LIBS += -L/usr/local/lib -lopencv_cudaimgproc -lopencv_cudaarithm -lopencv_xfeatures2d -lopencv_cudawarping -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudabgsegm -lopencv_cudaobjdetect
LIBS += -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
LIBS += -L/home/shiomi/work/GitHub/yolo/my_darknet -ldarknet-cpp-shared
#LIBS += -L/home/shiomi/work/beanstalk-client -lbeanstalk
#LIBS += -L/home/shiomi/work/caffe/build/install/lib -lcaffe
#LIBS += -L/usr/lib/x86_64-linux-gnu -lboost_system -lglog
LIBS += -pthread


HEADERS += \
    src/util.h \
    src/fixedqueue.h \
    src/json11.hpp \
    src/panorama.h \
    src/yolo2.hpp
