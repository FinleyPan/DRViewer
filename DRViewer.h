#ifndef DRVIEWER_H
#define DRVIEWER_H

#include <memory>

namespace visual_utils{

extern char const* const VERTEX_SHADER_DEFAULT;
extern char const* const FRAGMENT_SHADER_DEFAULT;
extern char const* const DEFAULT_WINDOW_NAME;

using byte = unsigned char;

enum GraphicAPI{
    OPENGL,
    VULKAN, ///TODO
    METAL   ///TODO
};

enum ImageFormat{
    RGB = 0, BGR = 1, RGBA = 3, BGRA = 4
};

enum SubWindowPos{  //---------------------
    TOP_LEFT1,      //|1|2|           |1|2|
    TOP_LEFT2,      //|----           ----|
    TOP_RIGHT1,     //|                   |
    TOP_RIGHT2,     //|                   |
    DOWN_LEFT1,     //|                   |
    DOWN_LEFT2,     //|----           ----|
    DOWN_RIGHT1,    //|1|2|           |1|2|
    DOWN_RIGHT2     //---------------------
};

class DRViewer{
public:
    DRViewer(float cam_x = 0,float cam_y = 0, float cam_z = 0,
             int width = 800, int height = 600,
             const char* window_name = DEFAULT_WINDOW_NAME,
             const char* vert_shader_src = VERTEX_SHADER_DEFAULT,
             const char* frag_shader_src = FRAGMENT_SHADER_DEFAULT,
             GraphicAPI api = GraphicAPI::OPENGL);
    ~DRViewer();
    DRViewer(const DRViewer&);
    DRViewer(DRViewer&&) noexcept;
    DRViewer& operator=(const DRViewer&);
    DRViewer& operator=(DRViewer&&);

    //return true once escape pressed
    bool ShouldExit() const;
    void Wait(unsigned int milliseconds);
    void Render();

    //each least indivisible element in the data array must be of float type
    //default layout of data array is like: ...|x y z r g b|x y z r g b|...
    void BindPoinCloudData(const void* data, size_t num_vertices, int stride = 6*sizeof(float),
                           int position_offset = 0, int color_offset = 3 * sizeof(float));
    void BindImageData(const byte* data, int width, int height, ImageFormat format, SubWindowPos win = DOWN_LEFT1);
    void AddCameraPose(float qw, float qx, float qy, float qz, float x, float y, float z);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}


#endif // DRVIEWER_H
