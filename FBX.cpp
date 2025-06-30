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
#include "quadtree_fbx.h"//lo más importante
#include <numeric>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>

std::vector<Vertex> loadFBX(const std::string& filepath) {
    std::filesystem::path absolutePath = std::filesystem::absolute(filepath);
    std::cout << "Buscando modelo en: " << absolutePath.string() << std::endl;
    std::cout << "El archivo existe: " << std::filesystem::exists(absolutePath) << std::endl;

    if (!std::filesystem::exists(absolutePath)) {
        std::cerr << "ERROR: No se encontró el archivo en:\n" << absolutePath << std::endl;
        std::cerr << "Directorio actual: " << std::filesystem::current_path() << std::endl;
        throw std::runtime_error("Archivo no encontrado: " + absolutePath.string());
    }

    Assimp::Importer importer;

    importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
    const aiScene* scene = importer.ReadFile(filepath,
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_FlipUVs |
        aiProcess_ValidateDataStructure |
        aiProcess_JoinIdenticalVertices |
        aiProcess_CalcTangentSpace);

    if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
        std::cout << "Error de carga de modelo";
        std::string error = importer.GetErrorString();
        std::cerr << "Error al cargar FBX:\n" << error << std::endl;
        throw std::runtime_error("FBX Error: " + error);
    }

    std::vector<Vertex> vertices;
    std::cout << "Modelo tiene " << scene->mNumMeshes << " mallas\n";

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh* mesh = scene->mMeshes[m];
        if (!mesh->mVertices) continue;

        std::cout << "Malla " << m << " tiene " << mesh->mNumVertices << " vértices\n";

        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            Vertex vertex;
            vertex.position = glm::vec3(
                mesh->mVertices[i].x,
                mesh->mVertices[i].y,
                mesh->mVertices[i].z
            );

            // Normal (con fallback)
            if (mesh->mNormals) {
                vertex.normal = glm::normalize(glm::vec3(
                    mesh->mNormals[i].x,
                    mesh->mNormals[i].y,
                    mesh->mNormals[i].z
                ));
            } else {
                vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            // UVs (con fallback)
            if (mesh->mTextureCoords[0]) {
                vertex.texCoords = glm::vec2(
                    mesh->mTextureCoords[0][i].x,
                    mesh->mTextureCoords[0][i].y
                );
            } else {
                vertex.texCoords = glm::vec2(0.0f);
            }

            vertices.push_back(vertex);
        }
    }

    if (vertices.empty()) {
        std::cerr << "Advertencia: No se cargaron vértices válidos\n";
    }

    return vertices;
}

class ModelTessellator {
public:
    std::unique_ptr<QuadtreeNode> root;
    QuadtreeBuilder builder;
    std::vector<QuadtreeNode*> leaf_nodes;
    std::vector<Vertex> original_vertices;

    ModelTessellator(const std::string& fbxPath, int depth = 5) {
        // Cargar modelo
        original_vertices = loadFBX(fbxPath);

        // Configurar builder
        builder.setMaxDepth(depth);
        root = builder.buildFromFBX(original_vertices);

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

        for (size_t i = 0; i < leaf_nodes.size(); ++i) {
            generateVerticesForNode(leaf_nodes[i], vertices, i * 4);
        }

        return vertices;
    }

    std::vector<unsigned int> generateIndices() {
        std::vector<unsigned int> indices;
        indices.reserve(leaf_nodes.size() * 6); // 2 triángulos por nodo (6 índices)

        for (size_t i = 0; i < leaf_nodes.size(); ++i) {
            unsigned int base_index = i * 4;

            // Primer triángulo
            indices.push_back(base_index + 0);
            indices.push_back(base_index + 1);
            indices.push_back(base_index + 2);

            // Segundo triángulo
            indices.push_back(base_index + 0);
            indices.push_back(base_index + 2);
            indices.push_back(base_index + 3);
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

    Vertex interpolateVertex(float u, float v) {
        if (original_vertices.empty()) return Vertex();

        // Si hay pocos vértices, devuelve el más cercano directamente
        if (original_vertices.size() <= 3) {
            size_t closest = 0;
            float min_dist = std::numeric_limits<float>::max();
            for (size_t i = 0; i < original_vertices.size(); ++i) {
                glm::vec2 uv_diff = original_vertices[i].texCoords - glm::vec2(u, v);
                float dist = glm::dot(uv_diff, uv_diff);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = i;
                }
            }
            return original_vertices[closest];
        }

        // Encuentra los vértices más cercanos en UV space
        std::vector<std::pair<float, size_t>> distances;
        distances.reserve(original_vertices.size());

        for (size_t i = 0; i < original_vertices.size(); ++i) {
            glm::vec2 uv_diff = original_vertices[i].texCoords - glm::vec2(u, v);
            float dist = glm::dot(uv_diff, uv_diff);
            distances.emplace_back(dist, i);
        }

        // Ordena por distancia y toma los 3 más cercanos
        std::sort(distances.begin(), distances.end());

        // Limita a máximo 3 vértices para interpolación
        size_t num_vertices = std::min(size_t(3), distances.size());

        // Interpolación con pesos inversos a la distancia
        Vertex result;
        result.position = glm::vec3(0.0f);
        result.normal = glm::vec3(0.0f);
        result.texCoords = glm::vec2(0.0f);

        float totalWeight = 0.0f;
        const float epsilon = 1e-6f;

        for (size_t i = 0; i < num_vertices; ++i) {
            float dist = distances[i].first;
            float weight = 1.0f / (dist + epsilon);
            totalWeight += weight;

            const auto& vertex = original_vertices[distances[i].second];
            result.position += vertex.position * weight;
            result.normal += vertex.normal * weight;
            result.texCoords += vertex.texCoords * weight;
        }

        // Normaliza
        if (totalWeight > 0.0f) {
            result.position /= totalWeight;
            result.normal /= totalWeight;
            result.texCoords /= totalWeight;

            // Asegúrate de que la normal esté normalizada
            if (glm::length(result.normal) > epsilon) {
                result.normal = glm::normalize(result.normal);
            } else {
                result.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }
        }

        return result;
    }

    void generateVerticesForNode(QuadtreeNode* node, std::vector<Vertex>& vertices, size_t base_index) {
        auto [u_bounds, v_bounds] = node->getUVBounds();

        // Generar 4 vértices para las esquinas del quad
        std::vector<std::pair<float, float>> corners = {
            {u_bounds.first, v_bounds.first},    // Bottom-left
            {u_bounds.second, v_bounds.first},   // Bottom-right
            {u_bounds.second, v_bounds.second},  // Top-right
            {u_bounds.first, v_bounds.second}    // Top-left
        };

        for (const auto& [u, v] : corners) {
            Vertex vertex = interpolateVertex(u, v);
            vertices.push_back(vertex);
        }
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
    std::cout << "Directorio actual: " << std::filesystem::current_path() << std::endl;

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

    // Crear el teselador de modelo
    ModelTessellator tessellator("Dragonite.FBX", 16);
    if (tessellator.original_vertices.empty()) {
        std::cerr << "Error: No se cargaron vértices del modelo" << std::endl;
        return -1;
    }

    // Generar geometría
    std::vector<Vertex> vertices = tessellator.generateVertices();
    std::vector<unsigned int> indices = tessellator.generateIndices();

    std::cout << "Generated " << vertices.size() << " vertices and "
              << indices.size() << " indices (" << indices.size()/3 << " triangles)" << std::endl;

    // Calcular bounding box para mejor visualización
    glm::vec3 minBound(FLT_MAX), maxBound(-FLT_MAX);
    for (const auto& v : vertices) {
        minBound = glm::min(minBound, v.position);
        maxBound = glm::max(maxBound, v.position);
    }
    glm::vec3 center = (minBound + maxBound) * 0.5f;
    glm::vec3 size = maxBound - minBound;
    float maxDimension = std::max({size.x, size.y, size.z});

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

    // Variables para la cámara mejoradas
    float cameraDistance = maxDimension * 2.0f;
    glm::vec3 cameraPos = center + glm::vec3(0.0f, 0.0f, cameraDistance);
    glm::vec3 cameraTarget = center;
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

        // Matrices de transformación mejoradas
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, -center); // Centra el modelo
        model = glm::rotate(model, (float)glfwGetTime() * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));

        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, cameraDistance * 10.0f);

        // Enviar uniformes
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glm::vec3 lightPos = center + glm::vec3(maxDimension, maxDimension, maxDimension);
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(glm::vec3(0.5f, 0.8f, 1.0f)));

        // Renderizar
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}