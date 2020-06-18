#include "DRViewer.h"

#include "shader_m.h"
#include <GLFW/glfw3.h>
#include "camera.h"

constexpr char const* VERTEX_SHADER_SRC=
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

constexpr char const* FRAGMENT_SHADER_SRC=
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "in vec3 Color;\n"
        "void main()\n"
        "{\n"
        "FragColor = vec4(Color,1.0);\n"
        "}\n";

static constexpr float vertices[] = {
    //front face
    -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
     0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
     0.5f, 0.5f,  -0.5f, 0.0f, 0.0f, 1.0f,
     0.5f, 0.5f,  -0.5f, 0.0f, 0.0f, 1.0f,
    -0.5f,  0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
    //back face
    -0.5f, -0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
     0.5f, -0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
     0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
     0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
    -0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f, 0.0f, 0.0f, 1.0f,
    //left face
    -0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
    -0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,
    //right face
     0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,
     0.5f,  0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
     0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
     0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
     0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 0.0f,
     0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,
    //bottom face
    -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
     0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
     0.5f, -0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
     0.5f, -0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
    -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
    //top face
    -0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
     0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
     0.5f,  0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
     0.5f,  0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 0.0f
};

//static constexpr unsigned int indices[] = {
//  0, 1, 3, //for the first triangle
//  1, 2, 3  //for the second triangle
//};


static float g_lastX = 0.0f;
static float g_lastY = 0.0f;
static float g_delta_time = 0.0f;
static float g_last_time = 0.0f;
static bool g_clr_left_mouse = true;
static bool g_clr_right_mouse = true;
static glm::mat4 g_model = glm::mat4(1.0f);
static glm::mat4 g_view = glm::mat4(1.0f);
static glm::mat4 g_projection = glm::mat4(1.0f);
static Camera g_camera = Camera(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
static int g_width = 0, g_height = 0;

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

        glm::mat4 r1 = glm::rotate(glm::mat4(), glm::radians(-yoffset * 0.5f), glm::vec3(1.0f,0.0f,0.0f));
        glm::mat4 r2 = glm::rotate(glm::mat4(), glm::radians( xoffset * 0.5f), glm::vec3(0.0f,1.0f,0.0f));
//        glm::mat4 r3 = glm::rotate(glm::mat4(), glm::radians(speed), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = /*r3 **/ r2 * r1 * g_model;
        for(int i=0; i<3; i++)
            g_model[i] = tmp[i];
        return ;
    }while(false);
}

void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mod){
    if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(), glm::radians(3.0f), glm::vec3(0,0,1.0f));
        glm::mat4 tmp = r3 * g_model;
        for(int i=0; i<3; i++)
            g_model[i] = tmp[i];
    }else if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
        glm::mat4 r3 = glm::rotate(glm::mat4(), glm::radians(-3.0f), glm::vec3(0,0,1.0f));
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
                             GLsizeiptr indices_buff_size = 0,GLenum usage = GL_STATIC_DRAW){
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vert_buff_size, vert_buff, usage);
    if(p_EBO != nullptr){
        if(indices_buff == nullptr || indices_buff_size == 0){
            std::cerr<<"ERROR: Null indices buffer or buffer size when binding EBO"<<std::endl;
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *p_EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_buff_size, indices_buff, usage);
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
}

static void UnBindRenderBuffer(GLuint VAO, GLuint VBO, GLuint* p_EBO = nullptr){

}


DRViewer::DRViewer(const Eigen::Vector3f& cam_pos, int width, int height,
                   const char* vert_shader_src,const char* frag_shader_src){
    glm::vec3 c_pos(cam_pos[0],cam_pos[1],cam_pos[2]);
    g_camera = Camera(c_pos);
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

    //create and bind buffer for drawing grids widget
    glGenVertexArrays(1, &grids_VAO_);
    glGenBuffers(1, &grids_VBO_);
    glGenBuffers(1, &grids_EBO_);
}

DRViewer::~DRViewer(){
    delete shader_;

    glDeleteVertexArrays(1, &grids_VAO_);
    glDeleteBuffers(1, &grids_VBO_);
    glDeleteBuffers(1, &grids_EBO_);
    glfwTerminate();
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
//    g_model = glm::translate(g_model, glm::vec3(0.1f,0,0));
//    g_model = glm::rotate(g_model, glm::radians(2.0f), glm::vec3(0,0,1));

    shader_->setMat4("model", g_model);
    shader_->setMat4("view", g_view);
    shader_->setMat4("projection", g_projection);

    BindRenderBuffer(vertices, sizeof(vertices), grids_VAO_, grids_VBO_);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

bool DRViewer::ShouldExit() const{
    if(glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);
    return glfwWindowShouldClose(window_);
}
