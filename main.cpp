#include <iostream>
#include <fstream>
#include <sstream>
#include "DRViewer.h"

using namespace std;

int main(int argc, char** argv){

    if(argc != 2){
        cout<<"usage: ./viewer <path-to-assoc-file>"<<endl;
        return -1;
    }
    ifstream fin(argv[1]);
    if(!fin.is_open()){
        printf("%s does not exist\n", argv[1]);
        return -1;
    }
    DRViewer viewer(0.5,0.5,8,800,600);
    while(!viewer.ShouldExit()){
        std::string line, tmp;
        if(getline(fin, line)){
            istringstream iss(line);
            iss>>tmp>>tmp>>tmp;
            float tx,ty,tz,qx,qy,qz,qw;
            iss>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
            viewer.AddCameraPose(qw,qx,qy,qz,tx,ty,tz);
        }
        viewer.Render();
//        viewer.Wait(30);
    }
    fin.close();
    return 0;
}
