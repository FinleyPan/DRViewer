#ifndef DRVIEWER_H
#define DRVIEWER_H

#include <vector>
#include "glad.h"

class GLFWwindow;
class Shader;

extern char const* const VERTEX_SHADER_DEFAULT;
extern char const* const FRAGMENT_SHADER_DEFAULT;

class DRViewer{
public:
    DRViewer(float cam_x,float cam_y,float cam_z, int width, int height,
             const char* vert_shader_src = VERTEX_SHADER_DEFAULT,
             const char* frag_shader_src = FRAGMENT_SHADER_DEFAULT);
    ~DRViewer();

    void BindPoinCloudData(const void* data, size_t num_vertices);
    bool ShouldExit() const;
    void Wait(unsigned int milliseconds);
    void Render();

    void AddCameraPose(float qw, float qx, float qy, float qz,
                       float x, float y, float z);

private:
    Shader* shader_;
    GLFWwindow* window_;
    std::vector<float> traj_;
};

#endif // DRVIEWER_H
