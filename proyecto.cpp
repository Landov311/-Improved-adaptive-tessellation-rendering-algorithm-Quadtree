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
#include <cmath>
#include "okkk.h"
#include <numeric>

const float PI = 3.14159265358979323846f;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;
};

class CircleTessellator {
private:
    std::unique_ptr<QuadtreeNode> root;
    QuadtreeBuilder builder;
    float radius;
    int tessellation_depth;
    std::vector<QuadtreeNode*> leaf_nodes; // Cache de nodos hoja

public:
    CircleTessellator(float r = 1.0f, int depth = 5)
     : radius(r), tessellation_depth(depth) {
        builder.setMaxDepth(depth);
        root = builder.buildFromMesh(nullptr);

        // Preparar one-ring vertices (simplificado)
        std::vector<int> one_ring_vertices(24);
        std::iota(one_ring_vertices.begin(), one_ring_vertices.end(), 0);

        // Asignar template y subdividir
        root->assignTemplateByType(one_ring_vertices);
        subdivideToDepth(root.get(), depth);

        // Pre-cachear nodos hoja
        collectLeafNodes(root.get(), leaf_nodes);
    }

    void subdivideToDepth(QuadtreeNode* node, int target_depth) {
        if (node->getDepth() >= target_depth) return;

        if (node->isLeaf()) {
            auto children = node->subdivide();

            // Asignar templates a los hijos
            std::vector<int> one_ring_vertices;
            for (int i = 0; i < 24; ++i) {
                one_ring_vertices.push_back(i);
            }

            for (auto* child : children) {
                child->assignTemplateByType(one_ring_vertices);
                subdivideToDepth(child, target_depth);
            }
        }
    }

    std::vector<Vertex> generateVertices() {
        std::vector<Vertex> vertices;
        vertices.reserve(leaf_nodes.size() * 4); // Pre-reservar memoria

        std::map<std::pair<int, int>, int> vertex_map;

        for (auto* node : leaf_nodes) {
            generateVerticesForNode(node, vertices, vertex_map);
        }

        return vertices;
    }

    std::vector<unsigned int> generateIndices() {
        std::vector<unsigned int> indices;
        indices.reserve(leaf_nodes.size() * 6); // 2 triángulos por nodo (6 índices)

        int vertex_offset = 0;
        for (auto* node : leaf_nodes) {
            generateIndicesForNode(node, indices, vertex_offset);
            vertex_offset += 4;
        }

        return indices;
    }

private:
    void collectLeafNodes(QuadtreeNode* node, std::vector<QuadtreeNode*>& leaf_nodes) {
        if (node->isLeaf()) {
            leaf_nodes.push_back(node);
        } else {
            for (size_t i = 0; i < node->getChildrenCount(); ++i) {
                auto* child = node->getChild(i);
                if (child) {
                    collectLeafNodes(child, leaf_nodes);
                }
            }
        }
    }

    void generateVerticesForNode(QuadtreeNode* node, std::vector<Vertex>& vertices,
                             std::map<std::pair<int, int>, int>& vertex_map) {
        auto [u_bounds, v_bounds] = node->getUVBounds();
        float u_min = u_bounds.first, u_max = u_bounds.second;
        float v_min = v_bounds.first, v_max = v_bounds.second;

        // Generar 4 vértices para las esquinas del quad
        std::vector<std::pair<float, float>> corners = {
            {u_min, v_min}, {u_max, v_min}, {u_max, v_max}, {u_min, v_max}
        };

        for (const auto& corner : corners) {
            Vertex vertex;

            // Convertir coordenadas UV a coordenadas esféricas para una esfera completa
            float theta = corner.first * 2.0f * PI;  // Ángulo longitudinal [0, 2π]
            float phi = corner.second * PI;         // Ángulo latitudinal [0, π] (ahora cubre toda la esfera)

            // Calcular posición en la esfera
            vertex.position.x = radius * sin(phi) * cos(theta);
            vertex.position.y = radius * cos(phi);
            vertex.position.z = radius * sin(phi) * sin(theta);

            // La normal es igual a la posición normalizada
            vertex.normal = glm::normalize(vertex.position);

            // Coordenadas de textura
            vertex.texCoords.x = corner.first;
            vertex.texCoords.y = corner.second;

            vertices.push_back(vertex);
        }
    }
    void generateIndicesForNode(QuadtreeNode* node, std::vector<unsigned int>& indices, int vertex_offset) {
        // Generar índices para un quad (2 triángulos)
        indices.push_back(vertex_offset + 0);
        indices.push_back(vertex_offset + 1);
        indices.push_back(vertex_offset + 2);

        indices.push_back(vertex_offset + 0);
        indices.push_back(vertex_offset + 2);
        indices.push_back(vertex_offset + 3);
    }
};
// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

int main() {
    // Inicializar GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Crear ventana
    GLFWwindow* window = glfwCreateWindow(800, 600, "Circle Tessellation with Quadtree", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Inicializar GLEW
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // Crear el teselador de círculo
    CircleTessellator tessellator(1.0f, 4); //ajustar profundidad a gusto c:

    // Generar geometría
    std::vector<Vertex> vertices = tessellator.generateVertices();
    std::vector<unsigned int> indices = tessellator.generateIndices();

    std::cout << "Generated " << vertices.size() << " vertices and "
              << indices.size() << " indices (" << indices.size()/3 << " triangles)" << std::endl;

    // Crear buffers
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Atributos de vértice
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
    glEnableVertexAttribArray(2);

    // Crear shader program
    unsigned int shaderProgram = createShaderProgram();

    // Variables para la cámara
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    // Loop de renderizado
    while (!glfwWindowShouldClose(window)) {
        // Input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Matrices de transformación
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, (float)glfwGetTime() * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));

        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        // Enviar uniformes
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(glm::vec3(2.0f, 2.0f, 2.0f)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(glm::vec3(0.5f, 0.8f, 1.0f)));

        // Renderizar
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Limpiar recursos
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}