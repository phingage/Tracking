TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ./src/main.cpp \
    ./src/util.cpp \
    ./src/json11.cpp \
    src/panorama.cpp

TARGET = npd

INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /home/shiomi/work/beanstalk-client
INCLUDEPATH += /usr/local/cuda-8.0/include

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_objdetect -lopencv_tracking -lopencv_stitching -lopencv_calib3d -lopencv_features2d -lopencv_flann
LIBS += -L/usr/local/lib -lopencv_cudaimgproc -lopencv_cudaarithm -lopencv_xfeatures2d -lopencv_cudawarping -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudabgsegm
LIBS += -L/home/shiomi/work/beanstalk-client -lbeanstalk
LIBS += -pthread
HEADERS += \
    ./include/util.h \
    ./include/fixedqueue.h \
    ./include/json11.hpp \
    ./include/panorama.h
