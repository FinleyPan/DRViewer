#include <iostream>
#include "DRViewer.h"

using namespace std;

int main()
{

    DRViewer viewer(Eigen::Vector3f(0,0,10),800,600);
    while(!viewer.ShouldExit()){
        viewer.Render();
    }
    return 0;
}
