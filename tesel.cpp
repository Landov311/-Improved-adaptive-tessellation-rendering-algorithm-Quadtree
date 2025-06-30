//
// Created by lvera on 29/06/2025.
//
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include "okkk.h"
#include <chrono>

// Estructura para un vértice 3D
struct Vertex3D {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;

    Vertex3D(float x, float y, float z, float nx = 0.0f, float ny = 1.0f, float nz = 0.0f, float u = 0.0f, float v = 0.0f)
        : position(x, y, z), normal(nx, ny, nz), texCoord(u, v) {}
};

class TessellatedMesh {
private:
    std::vector<Vertex3D> vertices;
    std::vector<unsigned int> indices;
    std::unique_ptr<QuadtreeNode> quadtree_root;
    GLuint VAO, VBO, EBO;

    // Función de altura para generar terreno
    float heightFunction(float u, float v) {
        // Función simple para generar un terreno ondulado
        return 0.5f * sin(u * 10.0f) * cos(v * 8.0f) +
               0.3f * sin(u * 15.0f + v * 12.0f) +
               0.2f * cos(u * 20.0f - v * 18.0f);
    }

    void generateRegularGrid(int resolution) {
        vertices.clear();
        indices.clear();

        // Generar vértices en una grilla regular
        for (int v = 0; v <= resolution; ++v) {
            for (int u = 0; u <= resolution; ++u) {
                float uv_u = static_cast<float>(u) / resolution;
                float uv_v = static_cast<float>(v) / resolution;
                float x = (uv_u - 0.5f) * 10.0f; // Escalar de -5 a 5
                float z = (uv_v - 0.5f) * 10.0f; // Escalar de -5 a 5
                float y = heightFunction(uv_u, uv_v);

                // Calcular normal aproximada
                float delta = 1.0f / resolution;
                float hL = (u > 0) ? heightFunction(uv_u - delta, uv_v) : y;
                float hR = (u < resolution) ? heightFunction(uv_u + delta, uv_v) : y;
                float hD = (v > 0) ? heightFunction(uv_u, uv_v - delta) : y;
                float hU = (v < resolution) ? heightFunction(uv_u, uv_v + delta) : y;

                glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * delta * 10.0f, hD - hU));

                vertices.emplace_back(x, y, z, normal.x, normal.y, normal.z, uv_u, uv_v);
            }
        }

        // Generar índices para triángulos
        for (int v = 0; v < resolution; ++v) {
            for (int u = 0; u < resolution; ++u) {
                int topLeft = v * (resolution + 1) + u;
                int topRight = topLeft + 1;
                int bottomLeft = (v + 1) * (resolution + 1) + u;
                int bottomRight = bottomLeft + 1;

                indices.push_back(topLeft);
                indices.push_back(bottomLeft);
                indices.push_back(topRight);
                indices.push_back(topRight);
                indices.push_back(bottomLeft);
                indices.push_back(bottomRight);
            }
        }
    }

    void generateVerticesFromQuadtree(QuadtreeNode* node, int max_depth, std::vector<std::pair<float, float>>& uvCoords) {
        if (!node) return;
        if (node->isLeaf() || node->getDepth() >= max_depth) {
            float u = node->getU();
            float v = node->getV();
            uvCoords.push_back({u, v});
            return;
        }
        if (node->getChildrenCount() == 0) {
            node->subdivide();
        }
        for (size_t i = 0; i < node->getChildrenCount(); ++i) {
            generateVerticesFromQuadtree(node->getChild(i), max_depth, uvCoords);
        }
    }

public:
    TessellatedMesh() {
        quadtree_root = std::make_unique<QuadtreeNode>(NodeType::REGULAR, 0, 0.5f, 0.5f);
        generateMesh(6);
        setupOpenGL();
    }
    void generateMesh(int max_depth) {
        auto start_time = std::chrono::high_resolution_clock::now();

        vertices.clear();
        indices.clear();

        int resolution = std::min(64, static_cast<int>(std::pow(2, max_depth)));
        generateRegularGrid(resolution);

        std::vector<std::pair<float, float>> uvCoords;
        generateVerticesFromQuadtree(quadtree_root.get(), max_depth, uvCoords);
        for (const auto& uv : uvCoords) {
            float x = (uv.first - 0.5f) * 10.0f;
            float z = (uv.second - 0.5f) * 10.0f;
            float y = heightFunction(uv.first, uv.second);

            float delta = 0.01f;
            float hL = heightFunction(uv.first - delta, uv.second);
            float hR = heightFunction(uv.first + delta, uv.second);
            float hD = heightFunction(uv.first, uv.second - delta);
            float hU = heightFunction(uv.first, uv.second + delta);

            glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * delta * 10.0f, hD - hU));
            vertices.emplace_back(x, y, z, normal.x, normal.y, normal.z, uv.first, uv.second);
        }


        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Malla generada: " << vertices.size() << " vértices, "
                  << indices.size() / 3 << " triángulos\n";
        std::cout << "Tiempo de generación: " << duration.count() << " ms\n";
    }

    void setupOpenGL() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex3D),
                     vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                     indices.data(), GL_STATIC_DRAW);

        // Posición
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D), (void*)0);
        glEnableVertexAttribArray(0);

        // Normal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex3D),
                             (void*)offsetof(Vertex3D, normal));
        glEnableVertexAttribArray(1);

        // Coordenadas de textura
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex3D),
                             (void*)offsetof(Vertex3D, texCoord));
        glEnableVertexAttribArray(2);

        glBindVertexArray(0);
    }

    void render() {
        if (indices.empty()) {
            std::cout << "No hay índices para renderizar\n";
            return;
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    void updateTessellation(int new_max_depth) {
        generateMesh(new_max_depth);

        // Actualizar buffers OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex3D),
                     vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                     indices.data(), GL_STATIC_DRAW);
    }

    size_t getVertexCount() const { return vertices.size(); }
    size_t getTriangleCount() const { return indices.size() / 3; }

    ~TessellatedMesh() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }
};

std::string loadShaderSource(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint compileShader(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Error compilando shader: " << infoLog << std::endl;
    }

    return shader;
}

GLuint createShaderProgram() {
    std::string vertexSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            TexCoord = aTexCoord;

            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
    )";

    // Fragment Shader
    std::string fragmentSource = R"(
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;

        out vec4 FragColor;

        uniform vec3 lightPos;
        uniform vec3 viewPos;

        void main() {
            // Colores basados en altura y coordenadas UV
            vec3 objectColor = mix(
                vec3(0.2, 0.8, 0.2), // Verde para partes bajas
                vec3(0.8, 0.6, 0.3), // Marrón para partes altas
                (FragPos.y + 2.0) / 4.0
            );

            // Iluminación básica
            vec3 norm = normalize(Normal);
            vec3 lightColor = vec3(1.0, 1.0, 1.0);

            // Luz ambiental
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * lightColor;

            // Luz difusa
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // Luz especular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        }
    )";

    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Error enlazando programa: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

float cameraX = 0.0f, cameraY = 5.0f, cameraZ = 10.0f;
float cameraYaw = -90.0f, cameraPitch = -30.0f;
bool firstMouse = true;
float lastX = 400, lastY = 300;

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    cameraYaw += xoffset;
    cameraPitch += yoffset;
    if (cameraPitch > 89.0f) cameraPitch = 89.0f;
    if (cameraPitch < -89.0f) cameraPitch = -89.0f;
}

int main() {
    // Inicializar GLFW
    if (!glfwInit()) {
        std::cerr << "Error inicializando GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Crear ventana
    GLFWwindow* window = glfwCreateWindow(1200, 800, "Teselación con Quadtree", nullptr, nullptr);
    if (!window) {
        std::cerr << "Error creando ventana GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Inicializar GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Error inicializando GLEW" << std::endl;
        return -1;
    }

    // Configurar OpenGL
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, 1200, 800);

    // Crear programa de shaders
    GLuint shaderProgram = createShaderProgram();
    if (shaderProgram == 0) {
        std::cerr << "Error creando programa de shaders" << std::endl;
        return -1;
    }

    // Crear malla teselada usando tu algoritmo
    TessellatedMesh mesh;

    std::cout << "\n=== Ejemplo 3D con Teselación Quadtree ===\n";
    std::cout << "Controles:\n";
    std::cout << "- Mouse: Rotar cámara\n";
    std::cout << "- WASD: Mover cámara\n";
    std::cout << "- 1-8: Cambiar nivel de teselación\n";
    std::cout << "- ESC: Salir\n\n";

    int current_tessellation_level = 6;

    // Bucle principal
    while (!glfwWindowShouldClose(window)) {
        // Procesar entrada
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // Controles de cámara
        float cameraSpeed = 0.1f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraZ -= cameraSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraZ += cameraSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraX -= cameraSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraX += cameraSpeed;
        }

        // Controles de teselación
        static bool keys_pressed[9] = {false}; // Para evitar múltiples pulsaciones
        for (int i = 1; i <= 8; ++i) {
            bool key_current = glfwGetKey(window, GLFW_KEY_0 + i) == GLFW_PRESS;
            if (key_current && !keys_pressed[i]) {
                current_tessellation_level = i;
                mesh.updateTessellation(i);
                std::cout << "Nivel de teselación: " << i
                          << " (" << mesh.getVertexCount() << " vértices, "
                          << mesh.getTriangleCount() << " triángulos)\n";
            }
            keys_pressed[i] = key_current;
        }

        // Limpiar pantalla
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Usar programa de shaders
        glUseProgram(shaderProgram);

        // Configurar matrices
        glm::mat4 model = glm::mat4(1.0f);

        glm::vec3 cameraFront;
        cameraFront.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraFront.y = sin(glm::radians(cameraPitch));
        cameraFront.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraFront = glm::normalize(cameraFront);

        glm::mat4 view = glm::lookAt(
            glm::vec3(cameraX, cameraY, cameraZ),
            glm::vec3(cameraX, cameraY, cameraZ) + cameraFront,
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f), 1200.0f / 800.0f, 0.1f, 100.0f
        );

        // Enviar matrices a shaders
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Configurar iluminación
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 5.0f, 10.0f, 5.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), cameraX, cameraY, cameraZ);

        // Renderizar malla
        mesh.render();

        glfwSwapBuffers(window);
    }

    glDeleteProgram(shaderProgram);
    glfwTerminate();

    return 0;
}