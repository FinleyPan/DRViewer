#include "DRViewer.h"
#include "glad.h"
#include "shader_m.h"
#include "camera.h"
#include "widgets.h"
#define STB_IMAGE_IMPLEMENTATION

#include <unordered_map>
#include <mutex>
#include <string.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#if defined(__linux) || defined(__posix)
  #include <unistd.h>
  #define SLEEP(MILLI_SECONDS) usleep(MILLI_SECONDS * 1000)
#elif defined(_WIN64) || defined(_WIN32)
///TODO for windows
#elif defined(__APPLE__)
///TODO macos
#else
  #error "unidentified operation system!"
#endif

namespace std{
//specialize hash for using SubWindowPos as key in unordered_map
template <>
struct hash<visual_utils::SubWindowPos>{
    using result_type = size_t;
    using argument_type = visual_utils::SubWindowPos;
    result_type operator()(const argument_type& s) const{
        return static_cast<result_type>(s);
    }
};

}

namespace visual_utils{

constexpr char const* VERTEX_SHADER_DEFAULT=
        "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in vec3 aColor;\n"
        "out vec3 Color;\n"
        "uniform mat4 model;\n"
        "uniform mat4 view;"
        "uniform mat4 projection;\n"
        "void main()\n"
        "{\n"
        "gl_Position = projection * view * model * vec4(aPos, 1.0f);\n"
        "Color = aColor;\n"
        "}\n";

constexpr char const* FRAGMENT_SHADER_DEFAULT=
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "in vec3 Color;\n"
        "void main()\n"
        "{\n"
        "FragColor = vec4(Color,1.0);\n"
        "}\n";

constexpr char const* TEXTURE_VERTEX_SHADER =
        "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in vec2 aTexCoord;\n"
        "out vec2 TexCoord;\n"
        "void main()\n"
        "{\n"
            "gl_Position = vec4(aPos, 1.0f);\n"
            "TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);\n"
        "}\n";

constexpr char const* TEXTURE_FRAGMENT_SHADER =
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "in vec2 TexCoord;\n"
        "uniform sampler2D texture1;\n"
        "void main()\n"
        "{\n"
            "FragColor = texture(texture1, TexCoord);\n"
        "}\n";

constexpr char const* DEFAULT_WINDOW_NAME = "DRViewer";

/*--------------helper functions definitions----------*/
namespace  {

constexpr float kSubWindowScale = 0.166666f; // 1/6
constexpr float kAngleSpeedZAxis = 3.0f;
constexpr float kSceneMoveSpeedFactor = 0.00125f;
constexpr float kPointSizeExpandSpeed = 1.0f;
constexpr float kMaxPointSize = 10.0f;
constexpr float kMinPointSize = 1.0f;

GLenum GLFormat(ImageFormat format){
    switch(format) {
        case RGB:  return GL_RGB;
        case BGR:  return GL_BGR;
        case RGBA: return GL_RGBA;
        case BGRA: return GL_BGRA;
    }
}

struct SubWindow{
    //(x,y) represents the downleft corner of the sub-window
    int x, y;
    int w, h;
    int img_w, img_h;
    ImageFormat format;
    float aspect_ratio;
    std::unique_ptr<byte> data;

    SubWindow(SubWindowPos pos, int parent_width, int parent_height, int _img_w,
              int _img_h, ImageFormat f): img_w(_img_w), img_h(_img_h), format(f){
        aspect_ratio = (float)img_w / img_h;
        size_t area = img_h * img_w;
        if(area > 0){
            data.reset(new byte[area * (f > 1? 4 : 3)]);
        }
        Resize(pos, parent_width, parent_height);
    }

    SubWindow(const SubWindow& rhs): x(rhs.x), y(rhs.y), img_w(rhs.img_w),
        img_h(rhs.img_h),format(rhs.format), aspect_ratio(rhs.aspect_ratio){
        size_t num_bytes = img_h * img_w * (format > 1? 4 : 3);
        data.reset(new byte[num_bytes]);
        memcpy(data.get(), rhs.data.get(), num_bytes);
    }

    SubWindow& operator=(const SubWindow& rhs){
        if(this != &rhs){
            size_t num_bytes = img_h * img_w * (format > 1? 4 : 3);
            memcpy(data.get(), rhs.data.get(), num_bytes);
        }
        return *this;
    }

    SubWindow(SubWindow&& rhs) = default;
    SubWindow& operator=(SubWindow&& rhs) = default;


    void Resize(SubWindowPos pos, int parent_width, int parent_height){
        h = kSubWindowScale * parent_height;
        w = h * aspect_ratio;
        switch(pos) {
            case TOP_LEFT1: x = 0; y = parent_height - h; break;
            case TOP_LEFT2: x = w; y = parent_height - h; break;
            case TOP_RIGHT1:x = parent_width -  2 * w; y = parent_height - h; break;
            case TOP_RIGHT2:x = parent_width - w; y = parent_height - h; break;
            case DOWN_LEFT1: x = 0; y = 0; break;
            case DOWN_LEFT2: x = w; y = 0; break;
            case DOWN_RIGHT1:x = parent_width -  2 * w; y = 0; break;
            case DOWN_RIGHT2:x = parent_width - w; y=0; break;
        }
    }
};

}

/*--------------class definitions---------------------*/
class ImplDRViewerBase{
public:
    ImplDRViewerBase(float x,float y,float z, int width, int height, GraphicAPI api):
        pos_cam_(x, y, z), width_(width), height_(height), traj_(0), api_(api) {}

    ImplDRViewerBase(const ImplDRViewerBase& rhs): traj_(rhs.traj_), api_(rhs.api_),
        pos_cam_(rhs.pos_cam_), width_(rhs.width_), height_(rhs.height_),
        sub_windows_(rhs.sub_windows_){
        array_pcl_ = rhs.array_pcl_;
        size_pcl_ = rhs.size_pcl_;
        pos_off_pcl_ = rhs.pos_off_pcl_;
        col_off_pcl_ = rhs.col_off_pcl_;
    }

    ImplDRViewerBase(ImplDRViewerBase&& rhs) noexcept: traj_(std::move(rhs.traj_)),
        sub_windows_(std::move(rhs.sub_windows_)),api_(rhs.api_), pos_cam_(rhs.pos_cam_),
        width_(rhs.width_),height_(rhs.height_){
        array_pcl_ = rhs.array_pcl_;
        size_pcl_ = rhs.size_pcl_;
        pos_off_pcl_ = rhs.pos_off_pcl_;
        col_off_pcl_ = rhs.col_off_pcl_;
    }

    ImplDRViewerBase& operator=(const ImplDRViewerBase& rhs) {
        if(this != &rhs){            
            traj_ = rhs.traj_;
            sub_windows_ = rhs.sub_windows_;
            pos_cam_ = rhs.pos_cam_;
            width_ = rhs.width_;
            height_ = rhs.height_;
            api_ = rhs.api_;
            array_pcl_ = rhs.array_pcl_;
            size_pcl_ = rhs.size_pcl_;
            pos_off_pcl_ = rhs.pos_off_pcl_;
            col_off_pcl_ = rhs.col_off_pcl_;
            model_ = rhs.model_;
            view_ = rhs.view_;
            projection_ = rhs.projection_;
            frustum_pose_ = rhs.frustum_pose_;
        }
        return *this;
    }

    ImplDRViewerBase& operator=(ImplDRViewerBase&& rhs) noexcept{
        if(this != &rhs){
            traj_ = std::move(rhs.traj_);
            sub_windows_ = std::move(rhs.sub_windows_);
            pos_cam_ = rhs.pos_cam_;
            width_ = rhs.width_;
            height_ = rhs.height_;
            api_ = rhs.api_;
            array_pcl_ = rhs.array_pcl_;
            size_pcl_ = rhs.size_pcl_;
            pos_off_pcl_ = rhs.pos_off_pcl_;
            col_off_pcl_ = rhs.col_off_pcl_;
            model_ = rhs.model_;
            view_ = rhs.view_;
            projection_ = rhs.projection_;
            frustum_pose_ = rhs.frustum_pose_;
        }
        return *this;
    }

    virtual ~ImplDRViewerBase(){};
    virtual bool ShouldExit() const = 0;
    virtual void Render() = 0;    

    virtual void Wait(unsigned int milliseconds){        
        SLEEP(milliseconds);
    }

    GraphicAPI APIType() const {return api_;}

    void BindPointCloudData(const void* data, size_t num_vertices,
                            int stride, int pos_off, int col_off){
        std::lock_guard<std::mutex> lck(mtx_);
        array_pcl_ = data;
        size_pcl_ = num_vertices;
        stride_pcl_ = stride;
        pos_off_pcl_ = pos_off;
        col_off_pcl_ = col_off;
    }

    void BindImageData(const byte* data, int w, int h, ImageFormat f, SubWindowPos sub_win){
        std::lock_guard<std::mutex> lck(mtx_);
        auto iter = sub_windows_.find(sub_win);
        if(iter != sub_windows_.end()){
            if(w != iter->second.img_w || h != iter->second.img_h){
                iter->second = std::move(SubWindow(iter->first, width_, height_, w, h, f));
            }
            memcpy(iter->second.data.get(), data, w * h * (f > 1? 4 : 3));
        }else{
            auto ret = sub_windows_.insert(std::make_pair(sub_win, SubWindow(
                                           sub_win, width_, height_, w, h, f)));
            CreateTexture(sub_win);
            memcpy(ret.first->second.data.get(), data, w * h * (f > 1? 4 : 3));
        }
    }

    void AddCameraPose(glm::quat& rotation, const glm::vec3& position,
                       const glm::vec3& color){
        std::lock_guard<std::mutex> lck(mtx_);
        frustum_pose_ = glm::mat4(rotation);
        frustum_pose_[3] = glm::vec4(position, 1.0f);
        traj_.emplace_back(position);
        traj_.emplace_back(color);
    }

protected:
    std::vector<glm::vec3> traj_;
    glm::mat4 model_ = glm::mat4(1.0f);
    glm::mat4 view_ = glm::mat4(1.0f);
    glm::mat4 projection_ = glm::mat4(1.0f);
    glm::mat4 frustum_pose_ = glm::mat4(1.0f);
    const void* array_pcl_ = nullptr;
    size_t size_pcl_ = 0;
    int stride_pcl_ = 0;
    int pos_off_pcl_ = 0;
    int col_off_pcl_ = 0;
    GraphicAPI api_;
    glm::vec3 pos_cam_;
    int width_, height_;    
    std::mutex mtx_;
    std::unordered_map<SubWindowPos, SubWindow> sub_windows_;

    virtual void CreateTexture(SubWindowPos pos) = 0;
};

class ImplDRViewerOGL : public ImplDRViewerBase{
public:
    ImplDRViewerOGL(float x,float y,float z, int width, int height, const char* vert_shader_src,
                    const char* frag_shader_src, const char* win_name, GraphicAPI api) :
        ImplDRViewerBase(x, y, z, width, height, api), camera_(pos_cam_), ref_count_(new size_t(1)) {

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif
        window_ = glfwCreateWindow(width_, height_, win_name, nullptr, nullptr);
        if (window_ == nullptr) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwDestroyWindow(window_);
            exit(-1);
        }
        callback_helper_ = new CallbackHelper(this);
        glfwSetWindowUserPointer(window_, callback_helper_);

        auto framebuffer_size_callback = [](GLFWwindow* win, int w, int h){
            auto helper = static_cast<CallbackHelper*>(glfwGetWindowUserPointer(win));
            helper->WindowSizeCallback(w,h);
        };

        auto scroll_callback = [](GLFWwindow* win, double xoff, double yoff){
            auto helper = static_cast<CallbackHelper*>(glfwGetWindowUserPointer(win));
            helper->ScrollCallback(win, xoff, yoff);
        };

        auto mouse_move_callback = [](GLFWwindow* win, double xpos, double ypos){
            auto helper = static_cast<CallbackHelper*>(glfwGetWindowUserPointer(win));
            helper->MouseMoveCallback(win, xpos, ypos);
        };

        auto keyboard_callback = [](GLFWwindow* win, int key, int scancode, int action, int mod){
            auto helper = static_cast<CallbackHelper*>(glfwGetWindowUserPointer(win));
            helper->KeyboardCallback(win, key, scancode, action, mod);
        };

        glfwMakeContextCurrent(window_);
        glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
        glfwSetKeyCallback(window_, keyboard_callback);
        glfwSetCursorPosCallback(window_, mouse_move_callback);
        glfwSetScrollCallback(window_, scroll_callback);
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
            std::cerr << "Failed to initialize GLAD" << std::endl;
            exit(-1);
        }
        plain_shader_ = new Shader(std::string(vert_shader_src), frag_shader_src);
        texture_shader_ = new Shader(std::string(TEXTURE_VERTEX_SHADER), std::string(TEXTURE_FRAGMENT_SHADER));
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glGenBuffers(1, &ebo_);
    }

    ImplDRViewerOGL(const ImplDRViewerOGL& rhs): ImplDRViewerBase(rhs),
        plain_shader_(rhs.plain_shader_), texture_shader_(rhs.texture_shader_),
        window_(rhs.window_),textures_(rhs.textures_){
        TrivialAssign(rhs);
        ++(*ref_count_);
    }

    ImplDRViewerOGL(ImplDRViewerOGL&& rhs) noexcept: ImplDRViewerBase(std::move(rhs)),
        plain_shader_(rhs.plain_shader_), texture_shader_(rhs.texture_shader_),
        window_(rhs.window_), textures_(std::move(rhs.textures_)){
        TrivialAssign(rhs);

        rhs.plain_shader_ = nullptr;
        rhs.texture_shader_ = nullptr;
        rhs.window_ = nullptr;
        rhs.ref_count_ = nullptr;
        rhs.callback_helper_ = nullptr;
    }

    ImplDRViewerOGL& operator=(const ImplDRViewerOGL& rhs){
        if(this != &rhs){
            Destruct();
            TrivialAssign(rhs);

            textures_ = rhs.textures_;
            ImplDRViewerBase::operator=(rhs);
            plain_shader_ = rhs.plain_shader_;
            texture_shader_ = rhs.texture_shader_;
            window_ = rhs.window_;
            ++(*ref_count_);
        }
        return *this;
    }

    ImplDRViewerOGL& operator=(ImplDRViewerOGL&& rhs){
        if(this != &rhs){
            Destruct();
            TrivialAssign(rhs);

            textures_ = std::move(rhs.textures_);
            ImplDRViewerBase::operator=(std::move(rhs));
            plain_shader_ = rhs.plain_shader_;
            texture_shader_ = rhs.texture_shader_;
            window_ = rhs.window_;

            rhs.plain_shader_ = nullptr;
            rhs.texture_shader_ = nullptr;
            rhs.window_ = nullptr;
            rhs.ref_count_ = nullptr;
            rhs.callback_helper_ = nullptr;
        }
        return *this;
    }

    virtual ~ImplDRViewerOGL(){
        Destruct();
    }

    virtual bool ShouldExit() const override{
        if(glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window_, true);
        return glfwWindowShouldClose(window_);
    }

    void Render() override{        
        std::lock_guard<std::mutex> lck(mtx_);

        float current_time = glfwGetTime();
        delta_time_ = current_time - last_time_;
        last_time_ = current_time;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_MULTISAMPLE);
        glViewport(0, 0, width_, height_);

        plain_shader_->use();
        view_ = camera_.GetViewMatrix();
        projection_ = glm::perspective(glm::radians(camera_.Zoom),
                           (float)width_ / height_, 0.1f, 100.0f);
        plain_shader_->setMat4("view", view_);
        plain_shader_->setMat4("projection", projection_);

//        DrawCube();
        DrawCoordinateSystem(4.0f);
        DrawGrids();
        DrawFrustum();
        DrawTrajectory();
        DrawPointCloud(point_size_);
        DrawTexture();

        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            printf("OpenGL error: %u", err);
            exit(-1);
        }
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_MULTISAMPLE);

        glfwSwapBuffers(window_);
        glfwPollEvents();
    }

private:
    class CallbackHelper{
    public:
        void WindowSizeCallback(int, int);
        void MouseMoveCallback(GLFWwindow*, double, double);
        void KeyboardCallback(GLFWwindow*, int, int, int, int);
        void ScrollCallback(GLFWwindow*, double, double);

        ImplDRViewerOGL* handle_;

        CallbackHelper(ImplDRViewerOGL* handle) : handle_(handle){}
    };

private:
    Shader* plain_shader_, *texture_shader_;
    GLFWwindow* window_;
    CallbackHelper* callback_helper_;
    float lastX_ = 0.0f, lastY_ = 0.0f;
    float last_time_ = 0.0f, delta_time_ = 0.0f;
    bool clr_left_mouse_ = true, clr_right_mouse_ = true;
    Camera camera_ = Camera(glm::vec3(0.0f, 0.0f, 0.0f));
    size_t* ref_count_ = nullptr;
    GLuint vao_, vbo_, ebo_;
    GLfloat point_size_ = 1.0f;
    std::unordered_map<SubWindowPos, GLuint*> textures_;

private:
    void TrivialAssign(const ImplDRViewerOGL& rhs) noexcept{
        lastX_ = rhs.lastX_;
        lastY_ = rhs.lastY_;
        vao_ = rhs.vao_;
        vbo_ = rhs.vbo_;
        ebo_ = rhs.ebo_;
        last_time_ = rhs.last_time_;
        delta_time_ = rhs.delta_time_;
        clr_left_mouse_ = rhs.clr_left_mouse_;
        clr_right_mouse_ = rhs.clr_right_mouse_;
        camera_ = rhs.camera_;
        ref_count_ = rhs.ref_count_;
        point_size_ = rhs.point_size_;
        callback_helper_ = rhs.callback_helper_;
        callback_helper_->handle_ = this;
    }

    void Destruct(){
        if(ref_count_ != nullptr){
            --(*ref_count_);
            if(*ref_count_ == 0){
                delete plain_shader_;
                delete texture_shader_;
                delete callback_helper_;
                delete ref_count_;
                ref_count_ = nullptr;
                for(auto& e : textures_){
                    glDeleteTextures(1, e.second);
                    delete e.second;
                }

                glDeleteVertexArrays(1, &vao_);
                glDeleteBuffers(1, &vbo_);
                glDeleteBuffers(1, &ebo_);                
                glfwDestroyWindow(window_);
            }
        }
    }

    virtual void CreateTexture(SubWindowPos pos) override{
        textures_[pos] = new GLuint;
        glGenTextures(1, textures_[pos]);
        glBindTexture(GL_TEXTURE_2D, *(textures_[pos]));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void BindRenderBuffer(const GLvoid* vert_buff, GLsizeiptr vert_buff_size, bool use_ebo = false,
                          const GLvoid* indices_buff = nullptr, GLsizeiptr indices_buff_size = 0,
                          GLenum usage = GL_STATIC_DRAW, GLsizei stride = 6 * sizeof(float),
                          size_t pos_offset = 0, size_t color_offset = 3 * sizeof(float), bool bind_tex = false){
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, vert_buff_size, vert_buff, usage);
        if(use_ebo){
            if(indices_buff == nullptr || indices_buff_size == 0){
                std::cerr<<"ERROR: Null indices buffer or buffer size when binding EBO"<<std::endl;
                return ;
            }
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_buff_size, indices_buff, usage);
        }
        if(bind_tex){
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
        }else{
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)pos_offset);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)color_offset);
        }
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }

    void DrawTexture(){
        for(auto it = sub_windows_.begin(); it != sub_windows_.end(); ++it){
            SubWindowPos pos = it->first;
            const SubWindow& sub_win = it->second;
            glBindTexture(GL_TEXTURE_2D, *textures_[pos]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sub_win.img_w, sub_win.img_h, 0,
                        GLFormat(sub_win.format), GL_UNSIGNED_BYTE, sub_win.data.get());
            glGenerateMipmap(GL_TEXTURE_2D);
            glViewport(sub_win.x, sub_win.y, sub_win.w, sub_win.h);

            texture_shader_->use();
            BindRenderBuffer(vertices_texture, sizeof(vertices_texture), true, indices_texture,
                            sizeof(indices_texture), GL_STATIC_DRAW, 0, 0, 0, true);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        }
    }

    void DrawCube(){
        BindRenderBuffer(vertices_cube, sizeof(vertices_cube));
        plain_shader_->setMat4("model", model_);
        glDrawArrays(GL_TRIANGLES, 0, 36);
    }

    void DrawCoordinateSystem(GLfloat line_width = 1.0f){
        BindRenderBuffer(vertices_coordinates, sizeof(vertices_coordinates));
        plain_shader_->setMat4("model", model_);
        glLineWidth(line_width);
        glDrawArrays(GL_LINES, 0, 6);
        glLineWidth(1.0f);
    }

    void DrawFrustum(GLfloat line_width = 1.0f){
        if(traj_.empty()) return;
        BindRenderBuffer(vertices_frustum, sizeof(vertices_frustum), true,
                         indices_frustum , sizeof(indices_frustum));
        plain_shader_->setMat4("model", model_ * frustum_pose_);
        glLineWidth(line_width);
        glDrawElements(GL_LINES, 16, GL_UNSIGNED_SHORT, 0);
        glLineWidth(1.0f);
    }

    void DrawGrids(GLfloat line_width = 1.0f){
        BindRenderBuffer(vertices_grids, sizeof(vertices_grids), true,
                         indices_grids , sizeof(indices_grids));
        plain_shader_->setMat4("model", model_);
        glLineWidth(line_width);
        glDrawElements(GL_LINES, sizeof(indices_grids) / sizeof(unsigned short),
                       GL_UNSIGNED_SHORT, 0);
        glLineWidth(1.0f);
    }

    void DrawTrajectory(GLfloat line_width = 1.0f){
       if(traj_.empty()) return;
       BindRenderBuffer(traj_.data(), traj_.size() * sizeof(glm::vec3));
       plain_shader_->setMat4("model", model_);
       glLineWidth(line_width);
       glDrawArrays(GL_LINE_STRIP, 0, traj_.size() / 2);// traj_.size() * 3 / 6
       glLineWidth(1.0f);
    }

    void DrawPointCloud(GLfloat point_size = 1.0f){        
        BindRenderBuffer(array_pcl_, size_pcl_ * stride_pcl_,
                         false, nullptr, 0, GL_STATIC_DRAW,
                         stride_pcl_, pos_off_pcl_, col_off_pcl_);
        plain_shader_->setMat4("model", model_);
        glPointSize(point_size);
        glDrawArrays(GL_POINTS, 0, size_pcl_);
        glPointSize(1.0f);
    }

};

void ImplDRViewerOGL::CallbackHelper::WindowSizeCallback(int w, int h){    
    handle_->width_ = w;
    handle_->height_ = h;
    for(auto it = handle_->sub_windows_.begin();
        it!= handle_->sub_windows_.end(); ++it){
        it->second.Resize(it->first, w, h);
    }
}

void ImplDRViewerOGL::CallbackHelper::ScrollCallback(GLFWwindow* win, double xoff, double yoff){
    if(glfwGetKey(win,  GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
       glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS){
        handle_->point_size_ += yoff > 0? kPointSizeExpandSpeed : -kPointSizeExpandSpeed;
        if(handle_->point_size_ > kMaxPointSize)
            handle_->point_size_ = kMaxPointSize;
        if(handle_->point_size_ < kMinPointSize)
            handle_->point_size_ = kMinPointSize;
    }else
        handle_->camera_.ProcessMouseScroll(yoff, handle_->delta_time_);
}

void ImplDRViewerOGL::CallbackHelper::MouseMoveCallback(GLFWwindow* window, double xpos, double ypos){
    do{
        if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            handle_->clr_left_mouse_ = true;
            break;
        }
        //move camera when left mouse key pressed
        if(handle_->clr_left_mouse_){
            handle_->lastX_ = xpos;
            handle_->lastY_ = ypos;
            handle_->clr_left_mouse_ = false;
        }

        float xoffset = xpos - handle_->lastX_;
        float yoffset = handle_->lastY_ - ypos;
        handle_->lastX_ = xpos;
        handle_->lastY_ = ypos;

        float speed = std::fabs(handle_->camera_.Position[2]) * kSceneMoveSpeedFactor;
        handle_->model_[3] = handle_->model_[3] + glm::vec4(glm::vec3(speed * xoffset, 0, 0),0)
                                + glm::vec4(glm::vec3(0, speed * yoffset, 0),0);
        return ;
    }while(false);

    do{
        if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
            handle_->clr_right_mouse_ = true;
            break;
        }
        //rotate camera when right mouse key pressed
        if(handle_->clr_right_mouse_){
            handle_->lastX_ = xpos;
            handle_->lastY_ = ypos;
            handle_->clr_right_mouse_ = false;
        }

        float xoffset = xpos - handle_->lastX_;
        float yoffset = handle_->lastY_ - ypos;
        handle_->lastX_ = xpos;
        handle_->lastY_ = ypos;

        glm::mat4 rx = glm::rotate(glm::mat4(1.0f), glm::radians(-yoffset * 0.5f), glm::vec3(1.0f,0.0f,0.0f));
        glm::mat4 ry = glm::rotate(glm::mat4(1.0f), glm::radians( xoffset * 0.5f), glm::vec3(0.0f,1.0f,0.0f));
        glm::mat4 tmp = ry * rx * handle_->model_;
        for(int i=0; i<3; i++)
            handle_->model_[i] = tmp[i];
        return ;
    }while(false);
}

void ImplDRViewerOGL::CallbackHelper::KeyboardCallback(GLFWwindow *win, int key, int scancode, int action, int mod){
    if(glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(1.0f), glm::radians(kAngleSpeedZAxis), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * handle_->model_;
        for(int i=0; i<3; i++)
            handle_->model_[i] = tmp[i];
    }else if(glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(1.0f), glm::radians(kAngleSpeedZAxis), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * handle_->model_;
        for(int i=0; i<3; i++)
            handle_->model_[i] = tmp[i];
    }
}

class ImplDRViewerVLK : public ImplDRViewerBase{
public:
    ImplDRViewerVLK(float x,float y,float z, int width, int height, const char* vert_shader_src,
                    const char* frag_shader_src, const char* win_name, GraphicAPI api) :
        ImplDRViewerBase(x, y, z, width, height, api) {}
    virtual bool ShouldExit() const override { return true;}
    virtual void CreateTexture(SubWindowPos pos) override {};
    void Render() override{}

    virtual ~ImplDRViewerVLK(){}
    ///TODO for Vulkan API
};

class ImplDRViewerMTL : public ImplDRViewerBase{
public:
    ImplDRViewerMTL(float x,float y,float z, int width, int height, const char* vert_shader_src,
                    const char* frag_shader_src, const char* win_name, GraphicAPI api) :
        ImplDRViewerBase(x, y, z, width, height, api) {}
    virtual bool ShouldExit() const override { return true;}
    virtual void CreateTexture(SubWindowPos pos) override {};
    void Render() override{}

    virtual ~ImplDRViewerMTL(){}
    ///TODO for Metal API
};


class DRViewer::Impl{
public:
    Impl(float cam_x,float cam_y,float cam_z, int width, int height,
         const char* vert_shader_src, const char* frag_shader_src,
         const char* win_name, GraphicAPI api) : impl_(nullptr){
        switch (api) {
            case GraphicAPI::OPENGL:
                impl_.reset(new ImplDRViewerOGL(cam_x, cam_y, cam_z, width, height,
                            vert_shader_src, frag_shader_src, win_name, api)); break;
            case GraphicAPI::VULKAN:
                impl_.reset(new ImplDRViewerVLK(cam_x, cam_y, cam_z, width, height,
                            vert_shader_src, frag_shader_src, win_name, api)); break;
            case GraphicAPI::METAL:
                impl_.reset(new ImplDRViewerMTL(cam_x, cam_y, cam_z, width, height,
                            vert_shader_src, frag_shader_src, win_name, api)); break;
        }
    }

    Impl(const Impl& rhs) : impl_(nullptr){
        switch(rhs.impl_->APIType()) {
            case GraphicAPI::OPENGL:
                impl_.reset(new ImplDRViewerOGL(*static_cast<ImplDRViewerOGL*>(
                                                rhs.impl_.get()))); break;
            case GraphicAPI::VULKAN:
                impl_.reset(new ImplDRViewerVLK(*static_cast<ImplDRViewerVLK*>(
                                                rhs.impl_.get()))); break;
            case GraphicAPI::METAL:
                impl_.reset(new ImplDRViewerMTL(*static_cast<ImplDRViewerMTL*>(
                                                rhs.impl_.get()))); break;
        }
    }

    Impl& operator=(const Impl& rhs){
        if(this != &rhs){
            *impl_ = *rhs.impl_;
        }
        return *this;
    }

    Impl(Impl&&) noexcept = default;
    Impl& operator=(Impl&&) = default;
    ~Impl() = default;

    bool ShouldExit() const {return impl_->ShouldExit();}
    void Render() {impl_->Render();}
    void BindPoinCloudData(const void* data, size_t num_vertices,
                           int stride, int pos_off, int col_off){
        impl_->BindPointCloudData(data, num_vertices,stride,
                                  pos_off, col_off);
    }
    void BindImageData(const byte *data, int width, int height,
                      ImageFormat format, SubWindowPos sub_win){
        impl_->BindImageData(data, width, height, format, sub_win);
    }
    void AddCameraPose(glm::quat& rotation, const glm::vec3& position,
                       const glm::vec3& color=glm::vec3(1.0f,1.0f,1.0f)){
        impl_->AddCameraPose(rotation, position, color);
    }
    void Wait(unsigned int milliseconds){
        impl_->Wait(milliseconds);
    }

private:
    std::unique_ptr<ImplDRViewerBase> impl_;
};


DRViewer::DRViewer(float cam_x,float cam_y,float cam_z, int width, int height,
                   const char* window_name,const char* vert_shader_src,
                   const char* frag_shader_src,GraphicAPI api) : impl_(new Impl(
                   cam_x, cam_y, cam_z, width, height, vert_shader_src,
                   frag_shader_src, window_name, api)) {}

DRViewer::DRViewer(const DRViewer& rhs) : impl_(new Impl(*rhs.impl_)){}

DRViewer& DRViewer::operator=(const DRViewer& rhs){
    if(this != &rhs){
        *impl_ = *rhs.impl_;
    }
    return *this;
}

DRViewer::DRViewer(DRViewer&&) noexcept = default;
DRViewer& DRViewer::operator=(DRViewer&&) = default;
DRViewer::~DRViewer() = default;


void DRViewer::BindPoinCloudData(const void *data, size_t num_vertices,
                                 int stride, int pos_off, int col_off){
    impl_->BindPoinCloudData(data, num_vertices, stride, pos_off, col_off);
}

void DRViewer::BindImageData(const byte *data, int width, int height,
                             ImageFormat format, SubWindowPos sub_win){
    impl_->BindImageData(data, width, height, format, sub_win);
}

void DRViewer::AddCameraPose(float qw, float qx, float qy, float qz,
                             float  x, float  y, float z){
    glm::quat r(qw, qx, qy, qz);
    glm::vec3 t(x,y,z);
    impl_->AddCameraPose(r, t);
}

void DRViewer::Render(){
    impl_->Render();
}

bool DRViewer::ShouldExit() const{
    return impl_->ShouldExit();
}

void DRViewer::Wait(unsigned int milliseconds){
    impl_->Wait(milliseconds);
}

}
