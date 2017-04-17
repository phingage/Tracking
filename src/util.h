#ifndef UTIL_H
#define UTIL_H

struct lblElement
{
    int  area;
    int  x;
    int  y;
    int  width;
    int  height;
    double  centerX;
    double  centerY;
};

extern std::string strsprintf(const char* format,...);
extern std::string randomString( size_t length );
extern int remkDirs(std::string s, mode_t mode);
extern bool mkDirs(std::string dirname);
extern bool chmodDir(int permission, std::string dirname);


#endif // UTIL_H
