#include "DRViewer.h"
#include "glad.h"
#include "shader_m.h"
#include "camera.h"
#include "widgets.h"

#include <mutex>
#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#if defined(__linux) || defined(__posix)
  #include <unistd.h>
  #define SLEEP(MILLI_SECONDS) usleep(MILLI_SECONDS * 1000)
#elif defined(_WIN64) || defined(_WIN32)
///TODO
#elif defined(__APPLE__)
///TODO
#else
  #error "unidentified operation system!"
#endif

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

/*--------------class definitions---------------------*/
class ImplDRViewerBase{
public:
    ImplDRViewerBase(float x,float y,float z, int width, int height, GraphicAPI api):
        pos_cam_(x, y, z), width_(width), height_(height), traj_(0), api_(api) {}

    ImplDRViewerBase(const ImplDRViewerBase& rhs): traj_(rhs.traj_), api_(rhs.api_),
        pos_cam_(rhs.pos_cam_), width_(rhs.width_), height_(rhs.height_),
        array_pc_(rhs.array_pc_), num_pc_(rhs.num_pc_){}

    ImplDRViewerBase(ImplDRViewerBase&& rhs) noexcept: traj_(std::move(rhs.traj_)),
        api_(rhs.api_), pos_cam_(rhs.pos_cam_), width_(rhs.width_),height_(rhs.height_),
        array_pc_(rhs.array_pc_), num_pc_(rhs.num_pc_){}

    ImplDRViewerBase& operator=(const ImplDRViewerBase& rhs) {
        if(this != &rhs){
            traj_ = rhs.traj_;
            pos_cam_ = rhs.pos_cam_;
            width_ = rhs.width_;
            height_ = rhs.height_;
            api_ = rhs.api_;
            array_pc_ = rhs.array_pc_;
            num_pc_ = rhs.num_pc_;
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
            pos_cam_ = rhs.pos_cam_;
            width_ = rhs.width_;
            height_ = rhs.height_;
            api_ = rhs.api_;
            array_pc_ = rhs.array_pc_;
            num_pc_ = rhs.num_pc_;
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
    void BindPointCloudData(const void* data, size_t num_vertices){
        array_pc_ = data;
        num_pc_ = num_vertices;
    }

    void AddCameraPose(glm::quat& rotation, const glm::vec3& position,
                       const glm::vec3& color){
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
    const void* array_pc_ = nullptr;
    size_t num_pc_ = 0;
    GraphicAPI api_;
    glm::vec3 pos_cam_;
    int width_, height_;
};

class ImplDRViewerOGL : public ImplDRViewerBase{
public:
    ImplDRViewerOGL(float x,float y,float z, int width, int height, const char* vert_shader_src,
                    const char* frag_shader_src, GraphicAPI api) : ImplDRViewerBase(x, y, z, width,
        height, api), camera_(pos_cam_), ref_count_(new size_t(1)) {

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif
        window_ = glfwCreateWindow(width_, height_, "DRViewer", nullptr, nullptr);
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
            helper->ScrollCallback(xoff, yoff);
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
        shader_ = new Shader(std::string(vert_shader_src), frag_shader_src);
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glGenBuffers(1, &ebo_);
    }

    ImplDRViewerOGL(const ImplDRViewerOGL& rhs): ImplDRViewerBase(rhs),
        shader_(rhs.shader_), window_(rhs.window_){
        TrivialAssign(rhs);
        ++(*ref_count_);
    }

    ImplDRViewerOGL(ImplDRViewerOGL&& rhs) noexcept: ImplDRViewerBase(std::move(rhs)),
        shader_(rhs.shader_), window_(rhs.window_){
        TrivialAssign(rhs);

        rhs.shader_ = nullptr;
        rhs.window_ = nullptr;
        rhs.ref_count_ = nullptr;
    }

    ImplDRViewerOGL& operator=(const ImplDRViewerOGL& rhs){
        if(this != &rhs){
            Destruct();
            TrivialAssign(rhs);

            ImplDRViewerBase::operator=(rhs);
            shader_ = rhs.shader_;
            window_ = rhs.window_;
            ++(*ref_count_);
        }
        return *this;
    }

    ImplDRViewerOGL& operator=(ImplDRViewerOGL&& rhs){
        if(this != &rhs){
            Destruct();
            TrivialAssign(rhs);

            ImplDRViewerBase::operator=(std::move(rhs));
            shader_ = rhs.shader_;
            window_ = rhs.window_;

            rhs.shader_ = nullptr;
            rhs.window_ = nullptr;
            rhs.ref_count_ = nullptr;
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
        //support multi-threads
        std::lock_guard<std::mutex> lck(mtx_);

        float current_time = glfwGetTime();
        delta_time_ = current_time - last_time_;
        last_time_ = current_time;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_MULTISAMPLE);

        shader_->use();
        view_ = camera_.GetViewMatrix();
        projection_ = glm::perspective(glm::radians(camera_.Zoom),
                           (float)width_ / height_, 0.1f, 100.0f);
        shader_->setMat4("view", view_);
        shader_->setMat4("projection", projection_);

//        DrawCube();
        DrawCoordinateSystem(4.0f);
        DrawGrids();
        DrawFrustum();
        DrawTrajectory();
        DrawPointCloud();

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
        void ScrollCallback(double, double);

        ImplDRViewerOGL* handle_;

        CallbackHelper(ImplDRViewerOGL* handle) : handle_(handle){}
    };

private:
    Shader* shader_;
    GLFWwindow* window_;
    CallbackHelper* callback_helper_;
    std::mutex mtx_;
    float lastX_ = 0.0f, lastY_ = 0.0f;
    float last_time_ = 0.0f, delta_time_ = 0.0f;
    bool clr_left_mouse_ = true, clr_right_mouse_ = true;
    Camera camera_ = Camera(glm::vec3(0.0f, 0.0f, 0.0f));
    size_t* ref_count_ = nullptr;
    GLuint vao_, vbo_, ebo_;

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
    }

    void Destruct(){
        if(ref_count_ != nullptr){
            --(*ref_count_);
            if(*ref_count_ == 0){
                delete shader_;
                delete callback_helper_;
                delete ref_count_;
                ref_count_ = nullptr;

                glDeleteVertexArrays(1, &vao_);
                glDeleteBuffers(1, &vbo_);
                glDeleteBuffers(1, &ebo_);
                glfwDestroyWindow(window_);
            }
        }
    }

    void BindRenderBuffer(const GLvoid* vert_buff, GLsizeiptr vert_buff_size, bool use_ebo = false,
                          const GLvoid* indices_buff = nullptr, GLsizeiptr indices_buff_size = 0,
                          GLenum usage = GL_STATIC_DRAW, GLsizei stride = 6 * sizeof(float),
                          size_t pos_offset = 0, size_t color_offset = 3 * sizeof(float)){
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)pos_offset);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)color_offset);
        glEnableVertexAttribArray(1);
    }

    void DrawCube(){
        BindRenderBuffer(vertices_cube, sizeof(vertices_cube));
        glDrawArrays(GL_TRIANGLES, 0, 36);
    }

    void DrawCoordinateSystem(GLfloat line_width = 1.0f){
        BindRenderBuffer(vertices_coordinates, sizeof(vertices_coordinates));
        shader_->setMat4("model", model_);
        glLineWidth(line_width);
        glDrawArrays(GL_LINES, 0, 6);
        glLineWidth(1.0f);
    }

    void DrawFrustum(GLfloat line_width = 1.0f){
        BindRenderBuffer(vertices_frustum, sizeof(vertices_frustum), true,
                         indices_frustum , sizeof(indices_frustum));
        shader_->setMat4("model", model_ * frustum_pose_);
        glLineWidth(line_width);
        glDrawElements(GL_LINES, 16, GL_UNSIGNED_SHORT, 0);
        glLineWidth(1.0f);
    }

    void DrawGrids(GLfloat line_width = 1.0f){
        BindRenderBuffer(vertices_grids, sizeof(vertices_grids), true,
                         indices_grids , sizeof(indices_grids));
        shader_->setMat4("model", model_);
        glLineWidth(line_width);
        glDrawElements(GL_LINES, sizeof(indices_grids) / sizeof(unsigned short),
                       GL_UNSIGNED_SHORT, 0);
        glLineWidth(1.0f);
    }

    void DrawTrajectory(GLfloat line_width = 1.0f){
       BindRenderBuffer(traj_.data(), traj_.size() * sizeof(glm::vec3));
       shader_->setMat4("model", model_);
       glLineWidth(line_width);
       glDrawArrays(GL_LINE_STRIP, 0, traj_.size() / 2);// traj_.size() * 3 / 6
       glLineWidth(1.0f);
    }

    void DrawPointCloud(GLfloat point_size = 1.0f){
        BindRenderBuffer(array_pc_, num_pc_ * 6 * sizeof(float));
        shader_->setMat4("model", model_);
        glPointSize(point_size);
        glDrawArrays(GL_POINTS, 0, num_pc_);
        glPointSize(1.0f);
    }

};

void ImplDRViewerOGL::CallbackHelper::WindowSizeCallback(int w, int h){
    glViewport(0, 0, w, h);
    handle_->width_ = w;
    handle_->height_ = h;
}

void ImplDRViewerOGL::CallbackHelper::ScrollCallback(double xoff, double yoff){
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

        float speed = std::fabs(handle_->camera_.Position[2]) * 0.00125f;
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

        glm::mat4 r1 = glm::rotate(glm::mat4(1.0f), glm::radians(-yoffset * 0.5f), glm::vec3(1.0f,0.0f,0.0f));
        glm::mat4 r2 = glm::rotate(glm::mat4(1.0f), glm::radians( xoffset * 0.5f), glm::vec3(0.0f,1.0f,0.0f));
//        glm::mat4 r3 = glm::rotate(glm::mat4(), glm::radians(speed), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = /*r3 **/ r2 * r1 * handle_->model_;
        for(int i=0; i<3; i++)
            handle_->model_[i] = tmp[i];
        return ;
    }while(false);
}

void ImplDRViewerOGL::CallbackHelper::KeyboardCallback(GLFWwindow *win, int key, int scancode, int action, int mod){
    if(glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(1.0f), glm::radians(3.0f), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * handle_->model_;
        for(int i=0; i<3; i++)
            handle_->model_[i] = tmp[i];
    }else if(glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(1.0f), glm::radians(-3.0f), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * handle_->model_;
        for(int i=0; i<3; i++)
            handle_->model_[i] = tmp[i];
    }
}

class ImplDRViewerVLK : public ImplDRViewerBase{
public:
    ImplDRViewerVLK(float x,float y,float z, int width, int height, const char* vert_shader_src,
                    const char* frag_shader_src, GraphicAPI api) :
        ImplDRViewerBase(x, y, z, width, height, api) {}
    virtual bool ShouldExit() const override { return true;}
    void Render() override{}

    virtual ~ImplDRViewerVLK(){}
    ///TODO for Vulkan API
};

class ImplDRViewerMTL : public ImplDRViewerBase{
public:
    ImplDRViewerMTL(float x,float y,float z, int width, int height, const char* vert_shader_src,
                    const char* frag_shader_src,  GraphicAPI api) :
        ImplDRViewerBase(x, y, z, width, height, api) {}
    virtual bool ShouldExit() const override { return true;}
    void Render() override{}

    virtual ~ImplDRViewerMTL(){}
    ///TODO for Metal API
};


class DRViewer::Impl{
public:
    Impl(float cam_x,float cam_y,float cam_z, int width, int height,
         const char* vert_shader_src, const char* frag_shader_src,
         GraphicAPI api) : impl_(nullptr){
        switch (api) {
            case GraphicAPI::OPENGL:
                impl_.reset(new ImplDRViewerOGL(cam_x, cam_y, cam_z, width, height,
                            vert_shader_src, frag_shader_src, api)); break;
            case GraphicAPI::VULKAN:
                impl_.reset(new ImplDRViewerVLK(cam_x, cam_y, cam_z, width, height,
                            vert_shader_src, frag_shader_src, api)); break;
            case GraphicAPI::METAL:
                impl_.reset(new ImplDRViewerMTL(cam_x, cam_y, cam_z, width, height,
                            vert_shader_src, frag_shader_src, api)); break;
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
    void BindPoinCloudData(const void* data, size_t num_vertices){
        impl_->BindPointCloudData(data, num_vertices);
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
                   const char* vert_shader_src, const char* frag_shader_src,
                   GraphicAPI api) : impl_(new Impl(cam_x, cam_y, cam_z, width,
            height, vert_shader_src, frag_shader_src, api)) {}

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


void DRViewer::BindPoinCloudData(const void *data, size_t num_vertices){
    impl_->BindPoinCloudData(data, num_vertices);
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
