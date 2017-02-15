TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ./src/main.cpp \
    ./src/util.cpp \
    ./src/json11.cpp

TARGET = npd

INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /home/shiomi/work/beanstalk-client

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_objdetect -lopencv_tracking
LIBS += -L/home/shiomi/work/beanstalk-client -lbeanstalk
LIBS += -pthread
HEADERS += \
    ./include/util.h \
    ./include/fixedqueue.h \
    ./include/json11.hpp
