#include "DRViewer.h"

#include "shader_m.h"
#include <GLFW/glfw3.h>
#include "camera.h"
#include "widgets.h"
#include <glm/gtc/quaternion.hpp>
#include <unistd.h>

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

static float g_lastX = 0.0f;
static float g_lastY = 0.0f;
static float g_delta_time = 0.0f;
static float g_last_time = 0.0f;
static bool g_clr_left_mouse = true;
static bool g_clr_right_mouse = true;
static glm::mat4 g_model = glm::mat4(1.0f);
static glm::mat4 g_view = glm::mat4(1.0f);
static glm::mat4 g_projection = glm::mat4(1.0f);
static glm::mat4 g_frustum_pose = glm::mat4(1.0f);
static Camera g_camera = Camera(glm::vec3(0.0f, 0.0f, 0.0f));
static int g_width = 0, g_height = 0;

static GLuint g_vao_coord, g_vbo_coord;
static GLuint g_vao_traj, g_vbo_traj;
static GLuint g_vao_pc, g_vbo_pc;
static GLuint g_vao_util, g_vbo_util;
static GLuint g_vao_grids, g_vbo_grids, g_ebo_grids;
static GLuint g_vao_frustum, g_vbo_frustum, g_ebo_frustum;

static glm::vec3 g_traj_origin = glm::vec3(0.0f, 0.0f, 0.0f);
static std::vector<glm::vec3> g_traj = std::vector<glm::vec3>(0);
static const void* g_array_pc = nullptr;
static size_t g_num_pc = 0;

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    g_width = width;
    g_height = height;
}

static void mouse_move_callback(GLFWwindow* window, double xpos, double ypos){
    do{
        if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            g_clr_left_mouse = true;
            break;
        }
        //move camera when left mouse key pressed
        if(g_clr_left_mouse){
            g_lastX = xpos;
            g_lastY = ypos;
            g_clr_left_mouse = false;
        }

        float xoffset = xpos - g_lastX;
        float yoffset = g_lastY - ypos;
        g_lastX = xpos;
        g_lastY = ypos;

        float speed = std::fabs(g_camera.Position[2]) * 0.00125f;
        g_model[3] = g_model[3] + glm::vec4(glm::vec3(speed * xoffset, 0, 0),0)
                                + glm::vec4(glm::vec3(0, speed * yoffset, 0),0);
        return ;
    }while(false);

    do{
        if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
            g_clr_right_mouse = true;
            break;
        }
        //rotate camera when right mouse key pressed
        if(g_clr_right_mouse){
            g_lastX = xpos;
            g_lastY = ypos;
            g_clr_right_mouse = false;
        }

        float xoffset = xpos - g_lastX;
        float yoffset = g_lastY - ypos;
        g_lastX = xpos;
        g_lastY = ypos;

        glm::mat4 r1 = glm::rotate(glm::mat4(1.0f), glm::radians(-yoffset * 0.5f), glm::vec3(1.0f,0.0f,0.0f));
        glm::mat4 r2 = glm::rotate(glm::mat4(1.0f), glm::radians( xoffset * 0.5f), glm::vec3(0.0f,1.0f,0.0f));
//        glm::mat4 r3 = glm::rotate(glm::mat4(), glm::radians(speed), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = /*r3 **/ r2 * r1 * g_model;
        for(int i=0; i<3; i++)
            g_model[i] = tmp[i];
        return ;
    }while(false);
}

void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mod){
    if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(1.0f), glm::radians(3.0f), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * g_model;
        for(int i=0; i<3; i++)
            g_model[i] = tmp[i];
    }else if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(1.0f), glm::radians(-3.0f), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * g_model;
        for(int i=0; i<3; i++)
            g_model[i] = tmp[i];
    }
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    g_camera.ProcessMouseScroll(yoffset, g_delta_time);
}

static void BindRenderBuffer(const GLvoid* vert_buff, GLsizeiptr vert_buff_size, GLuint VAO, GLuint VBO,
                             GLuint* p_EBO = nullptr, const GLvoid* indices_buff = nullptr,
                             GLsizeiptr indices_buff_size = 0,GLenum usage = GL_STATIC_DRAW,
                             GLsizei stride = 6 * sizeof(float)){
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vert_buff_size, vert_buff, usage);    
    if(p_EBO != nullptr){
        if(indices_buff == nullptr || indices_buff_size == 0){
            std::cerr<<"ERROR: Null indices buffer or buffer size when binding EBO"<<std::endl;
            return ;
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *p_EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_buff_size, indices_buff, usage);
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
}

static void DrawCube(GLuint VAO, GLuint VBO){
    BindRenderBuffer(vertices_cube, sizeof(vertices_cube), VAO, VBO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

static void DrawCoordinateSystem(GLuint VAO, GLuint VBO, const Shader* shader = nullptr,
                                 GLfloat line_width = 1.0f){
    BindRenderBuffer(vertices_coordinates, sizeof(vertices_coordinates), VAO, VBO);
    if(shader != nullptr){
        shader->setMat4("model", g_model);
    }
    glLineWidth(line_width);
    glDrawArrays(GL_LINES, 0, 6);
    glLineWidth(1.0f);
}

static void DrawFrustum(GLuint VAO, GLuint VBO, GLuint EBO, const Shader* shader = nullptr,
                        GLfloat line_width = 1.0f){
    BindRenderBuffer(vertices_frustum, sizeof(vertices_frustum), VAO, VBO,
                     &EBO, indices_frustum, sizeof(indices_frustum));
    if(shader != nullptr){
        shader->setMat4("model", g_model * g_frustum_pose);
    }
    glLineWidth(line_width);
    glDrawElements(GL_LINES, 16, GL_UNSIGNED_SHORT, 0);
    glLineWidth(1.0f);    
}

static void DrawGrids(GLuint VAO, GLuint VBO, GLuint EBO, const Shader* shader = nullptr,
                      GLfloat line_width = 1.0f){
    BindRenderBuffer(vertices_grids, sizeof(vertices_grids), VAO, VBO,
                     &EBO, indices_grids, sizeof(indices_grids));
    if(shader != nullptr){
        shader->setMat4("model", g_model);
    }
    glLineWidth(line_width);
    glDrawElements(GL_LINES, sizeof(indices_grids) / sizeof(unsigned short), GL_UNSIGNED_SHORT, 0);
    glLineWidth(1.0f);
}

static void DrawTrajectory(GLuint VAO, GLuint VBO, const Shader* shader = nullptr,
                           GLfloat line_width = 1.0f){
    BindRenderBuffer(g_traj.data(), g_traj.size() * sizeof(glm::vec3), VAO, VBO,
                     nullptr, nullptr, 0, GL_STREAM_DRAW);
    if(shader != nullptr){
        shader->setMat4("model", g_model);
    }
    glLineWidth(line_width);
    glDrawArrays(GL_LINE_STRIP, 0, g_traj.size() / 2); // g_traj.size() * 3 / 6
    glLineWidth(1.0f);
}

static void DrawPointCloud(GLuint VAO, GLuint VBO, const Shader* shader = nullptr,
                           GLfloat point_size = 1.0f){
    BindRenderBuffer(g_array_pc, g_num_pc * 6 * sizeof(float), VAO, VBO,
                     nullptr, nullptr, 0, GL_STATIC_DRAW);
    if(shader != nullptr){
        shader->setMat4("model", g_model);
    }
    glPointSize(point_size);
    glDrawArrays(GL_POINTS, 0, g_num_pc);
    glPointSize(1.0f);
}


DRViewer::DRViewer(float cam_x,float cam_y,float cam_z, int width, int height,
                   const char* vert_shader_src,const char* frag_shader_src){    
    g_camera = Camera(glm::vec3(cam_x,cam_y,cam_z));
    g_width = width;
    g_height = height;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif
    window_ = glfwCreateWindow(width, height, "DRViewer", nullptr, nullptr);
    if (window_ == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window_);
    glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
    glfwSetKeyCallback(window_, keyboard_callback);
    glfwSetCursorPosCallback(window_, mouse_move_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }
    shader_ = new Shader(std::string(vert_shader_src), frag_shader_src);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_MULTISAMPLE);

    //create and bind buffer for drawing grids widget
    glGenVertexArrays(1, &g_vao_grids);
    glGenBuffers(1, &g_vbo_grids);
    glGenBuffers(1, &g_ebo_grids);

    //create and bind buffer for drawing frustum widget
    glGenVertexArrays(1, &g_vao_frustum);
    glGenBuffers(1, &g_vbo_frustum);
    glGenBuffers(1, &g_ebo_frustum);

    //create and bind buffer for drawing coordinates system widget
    glGenVertexArrays(1, &g_vao_coord);
    glGenBuffers(1, &g_vbo_coord);

    //create and bind buffer for drawing trajectory widget
    glGenVertexArrays(1, &g_vao_traj);
    glGenBuffers(1, &g_vbo_traj);

    //create and bind buffer for drawing point cloud
    glGenVertexArrays(1, &g_vao_pc);
    glGenBuffers(1, &g_vbo_pc);

    //create and bind buffer for drawing util(e.g. cube) widget
    glGenVertexArrays(1, &g_vao_util);
    glGenBuffers(1, &g_vbo_util);
}

DRViewer::~DRViewer(){
    delete shader_;

    glDeleteVertexArrays(1, &g_vao_grids);
    glDeleteBuffers(1, &g_vbo_grids);
    glDeleteBuffers(1, &g_ebo_grids);

    glDeleteVertexArrays(1, &g_vao_frustum);
    glDeleteBuffers(1, &g_vbo_frustum);
    glDeleteBuffers(1, &g_ebo_frustum);

    glDeleteVertexArrays(1, &g_vao_coord);
    glDeleteBuffers(1, &g_vbo_coord);

    glDeleteVertexArrays(1, &g_vao_traj);
    glDeleteBuffers(1, &g_vbo_traj);

    glDeleteVertexArrays(1, &g_vao_pc);
    glDeleteBuffers(1, &g_vbo_pc);

    glDeleteVertexArrays(1, &g_vao_util);
    glDeleteBuffers(1, &g_vbo_util);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_MULTISAMPLE);
    glfwTerminate();
}

void DRViewer::BindPoinCloudData(const void *data, size_t num_vertices){
    g_array_pc = data;
    g_num_pc = num_vertices;
}

void DRViewer::AddCameraPose(float qw, float qx, float qy, float qz, float x, float y, float z){
    glm::quat q(qw, qx, qy, qz);
    g_frustum_pose = glm::mat4(q);
    g_frustum_pose[3] = glm::vec4(x, y, z, 1.0f);
    g_traj.push_back(glm::vec3(x,y,z));
    g_traj.push_back(glm::vec3(1.0f,1.0f,1.0f));
//    static bool first_traj = true;
//    if(first_traj){
//        g_traj_origin = glm::vec3(x,y,z);
//        first_traj = false;
//    }
//    //set trajectory's start point to the orginal point of world frame
//     g_frustum_pose[3] = glm::vec4(x - g_traj_origin[0], y - g_traj_origin[1],
//                                   z - g_traj_origin[2], 1.0f);
//    g_traj.rbegin()[1] -= g_traj_origin;
}

void DRViewer::Render(){
    float current_time = glfwGetTime();
    g_delta_time = current_time - g_last_time;
    g_last_time = current_time;

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader_->use();

    g_view = g_camera.GetViewMatrix();
    g_projection = glm::perspective(glm::radians(g_camera.Zoom), (float)g_width/g_height, 0.1f, 100.0f);

    shader_->setMat4("view", g_view);
    shader_->setMat4("projection", g_projection);

//    DrawCube(g_vao_util, g_vbo_util);
    DrawCoordinateSystem(g_vao_coord, g_vbo_coord, shader_, 4.0f);
    DrawGrids(g_vao_grids, g_vbo_grids, g_ebo_grids, shader_);
    DrawFrustum(g_vao_frustum, g_vbo_frustum, g_ebo_frustum, shader_);
    DrawTrajectory(g_vao_traj,g_vbo_traj, shader_);
    DrawPointCloud(g_vao_pc, g_vbo_pc, shader_);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        printf("OpenGL error: %u", err);
        exit(-1);
    }

    glfwSwapBuffers(window_);
    glfwPollEvents();
}

bool DRViewer::ShouldExit() const{
    if(glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);       
    return glfwWindowShouldClose(window_);
}

void DRViewer::Wait(unsigned int milliseconds){
    usleep(1000 * milliseconds);
}
