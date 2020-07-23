#include <iostream>
#include <fstream>
#include <sstream>
#include "DRViewer.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace visual_utils;

int main(int argc, char** argv){

    if(argc != 2){
        cout<<"usage: ./viewer <path-to-assoc-file>"<<endl;
        return -1;
    }
    string assoc_file_path = argv[1];
    ifstream fin(argv[1]);
    string root = assoc_file_path.substr(0, assoc_file_path.find_last_of("\\/"));
    if(!fin.is_open()){
        printf("%s does not exist\n", argv[1]);
        return -1;
    }    
    DRViewer viewer(0.5,0.5,8,800,600);
    while(!viewer.ShouldExit()){
        std::string line, img_ts, pose_ts, img_rel_path;
        if(getline(fin, line)){
            istringstream iss(line);
            iss>>img_ts>>img_rel_path>>pose_ts;
            cv::Mat img = cv::imread(root + "/"+img_rel_path, cv::IMREAD_COLOR);
            float tx,ty,tz,qx,qy,qz,qw;
            iss>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, DOWN_LEFT1);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, DOWN_LEFT2);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, DOWN_RIGHT1);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, DOWN_RIGHT2);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, TOP_LEFT1);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, TOP_LEFT2);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, TOP_RIGHT1);
            viewer.BindImageData(img.data, img.cols, img.rows, ImageFormat::BGR, TOP_RIGHT2);
            viewer.AddCameraPose(qw,qx,qy,qz,tx,ty,tz);
        }
        viewer.Render();
    }
    fin.close();
    return 0;
}
