#ifndef DRVIEWER_H
#define DRVIEWER_H

#include <memory>

namespace visual_utils{

extern char const* const VERTEX_SHADER_DEFAULT;
extern char const* const FRAGMENT_SHADER_DEFAULT;

enum class GraphicAPI{
    OPENGL,
    VULKAN, ///TODO
    METAL   ///TODO
};

class DRViewer{
public:
    DRViewer(float cam_x = 0,float cam_y = 0,float cam_z = 0,
             int width = 800, int height = 600,
             const char* vert_shader_src = VERTEX_SHADER_DEFAULT,
             const char* frag_shader_src = FRAGMENT_SHADER_DEFAULT,
             GraphicAPI api = GraphicAPI::OPENGL);
    ~DRViewer();
    DRViewer(const DRViewer&);
    DRViewer(DRViewer&&) noexcept;
    DRViewer& operator=(const DRViewer&);
    DRViewer& operator=(DRViewer&&);


    void BindPoinCloudData(const void* data, size_t num_vertices);
    bool ShouldExit() const;
    void Wait(unsigned int milliseconds);
    void Render();

    void AddCameraPose(float qw, float qx, float qy, float qz,
                       float x,  float y,  float z);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}


#endif // DRVIEWER_H
