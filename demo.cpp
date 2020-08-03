#include <iostream>
#include <fstream>
#include <sstream>
#include "DRViewer.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "data_list_path.h"

using namespace std;
using namespace visual_utils;

//for the sample dataset
constexpr float FX = 525.0;
constexpr float FY = 525.0;
constexpr float CX = 319.5;
constexpr float CY = 239.5;
constexpr float DEPTH_FACTOR = 5000;
constexpr float MAX_DEPTH = 2.0f;

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;

    Vertex(const glm::vec3 pos, const glm::vec3 col):
        position(pos), color(col) {}
};

/*pixel in raw depth map is of uint16, convert to proper type before displaying*/
cv::Mat GetNormalImageFromDepth(const cv::Mat& depth){
    cv::Mat tmp = depth.clone();
    cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
    return tmp;
}

/*pose is assumed from camera frame to world frame*/
void UpdatePointCloud(const cv::Mat& image, const cv::Mat& depth, const glm::mat3& R, const glm::vec3 T,
                      std::vector<Vertex>& pcl, float fx = FX, float fy = FY,float cx = CX, float cy = CY){
    int width = image.cols;
    int height = image.rows;
    for(int y = 0; y < height; y++){
        const uchar*  row_i = image.ptr<uchar>(y);
        const ushort* row_d = depth.ptr<ushort>(y);
        for(int x = 0; x < width; x++){
             float d = (float)row_d[x] / DEPTH_FACTOR;
             if(d < 1e-6 || d > MAX_DEPTH) continue;
             glm::vec3 pos = R * glm::vec3(d * (x - cx) / fx, d * (y - cy) / fy, d) + T;
             glm::vec3 col((float)row_i[x*3 + 2] / 255., (float)row_i[x*3 + 1] / 255., (float)row_i[x*3] / 255.);
             pcl.emplace_back(pos, col);
        }
    }
}

int main(){
    string assoc_file_path = data_list_path;
    ifstream fin(data_list_path);
    string root = assoc_file_path.substr(0, assoc_file_path.find_last_of("\\/"));
    if(!fin.is_open()){
        printf("%s does not exist\n", data_list_path);
        return -1;
    }
    DRViewer viewer(0.5,0.5,8,800,600);
    std::vector<Vertex> pcl;    
    while(!viewer.ShouldExit()){
        std::string line,depth_rel_path, img_rel_path;
        if(getline(fin, line)){
            istringstream iss(line);
            iss>>depth_rel_path>>img_rel_path;
            cv::Mat image = cv::imread(root + "/" + img_rel_path, cv::IMREAD_UNCHANGED);
            cv::Mat depth = cv::imread(root + "/" + depth_rel_path, cv::IMREAD_UNCHANGED);
            float tx,ty,tz,qx,qy,qz,qw;
            iss>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
            glm::mat3 R(glm::quat(qw, qx, qy, qz));
            glm::vec3 T(tx, ty, tz);
            UpdatePointCloud(image, depth, R, T, pcl);

            viewer.BindImageData(image.data, image.cols, image.rows, ImageFormat::BGR, DOWN_LEFT1);
            viewer.BindImageData(GetNormalImageFromDepth(depth).data, depth.cols, depth.rows,
                                 ImageFormat::BGR, DOWN_LEFT2);
            viewer.BindPoinCloudData(pcl.data(), pcl.size());
            viewer.AddCameraPose(qw,qx,qy,qz,tx,ty,tz);
            viewer.Wait(200);
        }
        viewer.Render();
    }
    fin.close();
    return 0;
}
