#ifndef DRVIEWER_H
#define DRVIEWER_H

#include <Eigen/Core>
#include <vector>
#include "glad.h"

class GLFWwindow;
class Shader;

extern char const* const VERTEX_SHADER_DEFAULT;
extern char const* const FRAGMENT_SHADER_DEFAULT;

class DRViewer{
public:
    DRViewer(const Eigen::Vector3f& cam_pos, int width, int height,
             const char* vert_shader_src = VERTEX_SHADER_DEFAULT,
             const char* frag_shader_src = FRAGMENT_SHADER_DEFAULT);
    ~DRViewer();

    bool ShouldExit() const;
    void Render();

private:
    Shader* shader_;
    GLFWwindow* window_;
};

#endif // DRVIEWER_H
