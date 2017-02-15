#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdarg.h>
#include <iterator>
#include <algorithm>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include "../include/util.h"

// Functions

//c++ version sprintf
std::string strsprintf(const char* format,...){
    va_list ap;
    va_start(ap, format);
    char* alloc;
    if(vasprintf(&alloc,format,ap) == -1) {
     return std::string("");
    }
    va_end(ap);
    std::string retStr = std::string(alloc);
    free(alloc);
    return retStr;
}

//make random string
std::string randomString( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}



int remkDirs(std::string s, mode_t mode)
{
    size_t pre=0,pos;
    std::string dir;
    int mdret;

    if(s[s.size()-1]!='/'){
        // force trailing / so we can handle everything in loop
        s+='/';
    }

    while((pos=s.find_first_of('/',pre))!=std::string::npos){
        dir=s.substr(0,pos++);
        pre=pos;
        if(dir.size()==0) continue; // if leading / first time is 0 length
        if((mdret=mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}

//make dir from system function with sudo
bool mkDirs(std::string dirname)
{
    // is dir exist?
    struct stat st;
    if(stat(dirname.c_str(),&st) == 0){
        mode_t m = st.st_mode;
        if(S_ISDIR(m)){
            return true;
        }
    }

    //create folder
    char buffer[128];
//    sprintf (buffer, "sudo mkdir -m 777 -p %s", dirname.c_str());
    sprintf (buffer, "sudo install --directory --mode=777 %s", dirname.c_str());
    int dir_err = system(buffer);
    if (-1 == dir_err)        
    {
        std::cout << "Error creating directory!" << std::endl;
        return false;
    }
    return true;
}

//set chmod with sudo
bool chmodDir(int permission, std::string dirname)
{

    //permission for folder
    char buffer[256];
    sprintf (buffer, "sudo chmod -R %d %s", permission, dirname.c_str());
    int chmod_err = system(buffer);
    if (-1 == chmod_err)
    {
        std::cout << "Error chmod" << std::endl;
        return false;
    }
    return true;
}

