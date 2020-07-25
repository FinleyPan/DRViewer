# Introduction
DRViewer means **Dense Reconstruction Viewer**, it's basically a lightweight 3D visualization tool based on modern graphic pipeline and C++11, designed for Simultaneous localization and mapping([SLAM](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping) in brief) and realtime MVS, capable of viewing structured **3D point cloud**, **camera trajectory** and **multiple input sources**(up to eight).
<p align = "center">
<img src="https://github.com/FinleyPan/DRViewer/blob/master/recon.gif" alt="recon_gif" height="300">
<img src="https://github.com/FinleyPan/DRViewer/blob/master/track.gif" alt="track_gif" height="300">
</p>

# How to Build && Usage
after cloning this project, install following prerequisites firstly:
- [GLFW](https://www.glfw.org/):
```
$ sudo apt-get install libglfw-dev
```
- [GLM](https://glm.g-truc.net/0.9.9/index.html)
```
sudo apt-get install libglm-dev
```
- OpenCV 3(optionally for building the demo)

Then follow cmake routine to build the whole project,the library file and header file can be found in `/path/to/install_directory`:
```
$ cd /path/to/project_directory
$ mkdir build && cd build
$ cmake -DCMAK_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install_directory ..
$ make && make install
```
Finally, you can add these files to your projects.

## Interactive Operations on DRViewer
1 *move mouse under left mouse button pressed*: move the whole 3D scene.  
2 *move mouse under right mouse button pressed*: rotate the whole 3D scene around x/y axis.  
3 *scroll mouse wheel*: zoom in and out the whole 3D scene.  
4 *scroll mouse wheel under ctrl pressed*: enlarge and shrink ths points.  
5 *press left/right arrow button*: rotate the whole 3D scene around z axis.
