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
#include <chrono>

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
    // Mapas para búsqueda más eficiente con mejor granularidad
    std::map<std::pair<int, int>, std::vector<size_t>> spatial_hash;
    std::map<std::pair<float, float>, std::vector<size_t>> uv_hash;
    float hash_cell_size = 0.002f; // Celda más pequeña para mejor precisión
    float uv_hash_size = 0.01f; // Hash UV más fino

    // Nuevos campos para mejor interpolación
    std::vector<std::vector<size_t>> triangle_indices; // Triángulos originales
    std::map<std::pair<size_t, size_t>, std::vector<size_t>> edge_map; // Mapeo de aristas

    ModelTessellator(const std::string& fbxPath, int depth = 5) {
        // Cargar modelo
        original_vertices = loadFBX(fbxPath);

        // Construir estructuras de datos mejoradas
        buildTopologyInfo();
        buildSpatialHash();
        buildUVHash();

        builder.setMaxDepth(std::max(depth, 10)); // Incrementado para mayor detalle
        root = builder.buildFromFBX(original_vertices);

        adaptiveSubdivision(root.get());

        leaf_nodes.clear();
        collectLeafNodes(root.get(), leaf_nodes);
    }

    void buildTopologyInfo() {
        triangle_indices.clear();
        edge_map.clear();

        for (size_t i = 0; i + 2 < original_vertices.size(); i += 3) {
            triangle_indices.push_back({i, i+1, i+2});

            // Mapear aristas
            std::vector<std::pair<size_t, size_t>> edges = {
                {i, i+1}, {i+1, i+2}, {i+2, i}
            };

            for (auto& edge : edges) {
                if (edge.first > edge.second) std::swap(edge.first, edge.second);
                edge_map[edge].push_back(i/3); // Índice del triángulo
            }
        }
    }

    void adaptiveSubdivision(QuadtreeNode* node) {
        if (!node || node->getDepth() >= 15) return; // Límite máximo aumentado

        auto [u_bounds, v_bounds] = node->getUVBounds();

        float vertex_density = calculateVertexDensity(u_bounds.first, v_bounds.first,
                                                     u_bounds.second, v_bounds.second);
        float curvature_variation = calculateCurvatureVariation(u_bounds.first, v_bounds.first,
                                                               u_bounds.second, v_bounds.second);
        float geometric_complexity = calculateGeometricComplexity(u_bounds.first, v_bounds.first,
                                                                 u_bounds.second, v_bounds.second);

        bool should_subdivide = vertex_density > 2.0f || // Umbral reducido
                               curvature_variation > 0.15f || // Más sensible a curvatura
                               geometric_complexity > 0.2f || // Nueva métrica
                               node->getDepth() < 8; // Mayor profundidad mínima

        if (should_subdivide && node->isLeaf()) {
            auto children = node->subdivide();

            std::vector<int> one_ring_vertices;
            for (int i = 0; i < 48; ++i) { // Duplicado para mayor densidad
                one_ring_vertices.push_back(i);
            }

            for (auto* child : children) {
                child->assignTemplateByType(one_ring_vertices);
                adaptiveSubdivision(child); // Recursión en hijos
            }
        }
    }

    float calculateVertexDensity(float u_min, float v_min, float u_max, float v_max) {
        int count = 0;
        float area = (u_max - u_min) * (v_max - v_min);

        for (const auto& vertex : original_vertices) {
            if (vertex.texCoords.x >= u_min && vertex.texCoords.x <= u_max &&
                vertex.texCoords.y >= v_min && vertex.texCoords.y <= v_max) {
                count++;
            }
        }

        return area > 0.0f ? count / area : 0.0f;
    }

    float calculateCurvatureVariation(float u_min, float v_min, float u_max, float v_max) {
        std::vector<glm::vec3> normals;
        std::vector<glm::vec3> positions;

        for (const auto& vertex : original_vertices) {
            if (vertex.texCoords.x >= u_min && vertex.texCoords.x <= u_max &&
                vertex.texCoords.y >= v_min && vertex.texCoords.y <= v_max) {
                normals.push_back(vertex.normal);
                positions.push_back(vertex.position);
            }
        }

        if (normals.size() < 3) return 0.0f;

        float normal_variation = 0.0f;
        float position_variation = 0.0f;

        glm::vec3 avg_normal(0.0f);
        glm::vec3 avg_position(0.0f);

        for (size_t i = 0; i < normals.size(); ++i) {
            avg_normal += normals[i];
            avg_position += positions[i];
        }
        avg_normal /= static_cast<float>(normals.size());
        avg_position /= static_cast<float>(positions.size());

        for (size_t i = 0; i < normals.size(); ++i) {
            normal_variation += glm::length(normals[i] - avg_normal);
            position_variation += glm::length(positions[i] - avg_position);
        }

        return (normal_variation + position_variation * 0.1f) / normals.size();
    }

    float calculateGeometricComplexity(float u_min, float v_min, float u_max, float v_max) {
        std::vector<glm::vec3> positions;

        for (const auto& vertex : original_vertices) {
            if (vertex.texCoords.x >= u_min && vertex.texCoords.x <= u_max &&
                vertex.texCoords.y >= v_min && vertex.texCoords.y <= v_max) {
                positions.push_back(vertex.position);
            }
        }

        if (positions.size() < 4) return 0.0f;

        // Calcular varianza direccional para detectar características geométricas
        glm::vec3 center(0.0f);
        for (const auto& pos : positions) {
            center += pos;
        }
        center /= static_cast<float>(positions.size());

        float max_distance = 0.0f;
        float variance = 0.0f;

        for (const auto& pos : positions) {
            float dist = glm::length(pos - center);
            max_distance = std::max(max_distance, dist);
            variance += dist * dist;
        }

        variance /= positions.size();

        return variance / (max_distance * max_distance + 1e-6f);
    }

    std::vector<Vertex> generateVertices() {
        std::vector<Vertex> vertices;
        vertices.reserve(leaf_nodes.size() * 25);

        for (size_t i = 0; i < leaf_nodes.size(); ++i) {
            generateHighResVerticesForNode(leaf_nodes[i], vertices);
        }

        return vertices;
    }

    std::vector<unsigned int> generateIndices() {
        std::vector<unsigned int> indices;
        indices.reserve(leaf_nodes.size() * 72);

        unsigned int current_index = 0;
        for (size_t i = 0; i < leaf_nodes.size(); ++i) {
            generateIndicesForNode(current_index, indices);
            current_index += 25; // 5x5 = 25 vértices por nodo
        }

        return indices;
    }

private:
    void buildSpatialHash() {
        for (size_t i = 0; i < original_vertices.size(); ++i) {
            const auto& pos = original_vertices[i].position;
            int x = static_cast<int>(std::floor(pos.x / hash_cell_size));
            int y = static_cast<int>(std::floor(pos.y / hash_cell_size));
            spatial_hash[{x, y}].push_back(i);
        }
    }

    void buildUVHash() {
        for (size_t i = 0; i < original_vertices.size(); ++i) {
            const auto& uv = original_vertices[i].texCoords;
            float u_key = std::floor(uv.x / uv_hash_size) * uv_hash_size;
            float v_key = std::floor(uv.y / uv_hash_size) * uv_hash_size;
            uv_hash[{u_key, v_key}].push_back(i);
        }
    }

    std::vector<size_t> getNearbyVerticesUV(float u, float v, float radius = 0.03f) {
        std::vector<size_t> nearby;

        // Buscar en celdas del hash UV con mayor alcance
        float u_key = std::floor(u / uv_hash_size) * uv_hash_size;
        float v_key = std::floor(v / uv_hash_size) * uv_hash_size;

        // Expandir búsqueda a más celdas vecinas
        int search_range = 2;
        for (int du = -search_range; du <= search_range; du++) {
            for (int dv = -search_range; dv <= search_range; dv++) {
                auto it = uv_hash.find({u_key + du * uv_hash_size, v_key + dv * uv_hash_size});
                if (it != uv_hash.end()) {
                    for (size_t idx : it->second) {
                        float dist = glm::length(original_vertices[idx].texCoords - glm::vec2(u, v));
                        if (dist <= radius) {
                            nearby.push_back(idx);
                        }
                    }
                }
            }
        }

        return nearby;
    }

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
    Vertex interpolateVertexAdvanced(float u, float v) {
        if (original_vertices.empty()) return Vertex();
        std::vector<size_t> nearby_indices = getNearbyVerticesUV(u, v, 0.05f);

        if (nearby_indices.empty()) {
            nearby_indices = getNearbyVerticesUV(u, v, 0.1f);
        }

        if (nearby_indices.empty()) {
            // Fallback: buscar el más cercano globalmente
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

        // Interpolación mejorada con pesos más estables
        const float epsilon = 1e-10f;

        // Ordenar por distancia y tomar más muestras
        std::vector<std::pair<float, size_t>> distances;
        for (size_t idx : nearby_indices) {
            glm::vec2 uv_diff = original_vertices[idx].texCoords - glm::vec2(u, v);
            float dist = glm::dot(uv_diff, uv_diff);
            distances.emplace_back(dist, idx);
        }

        std::sort(distances.begin(), distances.end());

        // Si está muy cerca de un vértice, usar directamente
        if (distances[0].first < epsilon) {
            return original_vertices[distances[0].second];
        }

        // Usar más vértices para interpolación más suave (hasta 12)
        size_t num_samples = std::min(size_t(12), distances.size());

        Vertex result;
        result.position = glm::vec3(0.0f);
        result.normal = glm::vec3(0.0f);
        result.texCoords = glm::vec2(u, v);

        float total_weight = 0.0f;

        // Usar función de peso más suave (Wendland)
        for (size_t i = 0; i < num_samples; ++i) {
            float dist = std::sqrt(distances[i].first);

            // Función de peso Wendland C2 para interpolación más suave
            float h = 0.1f; // Radio de soporte
            float weight = 0.0f;
            if (dist < h) {
                float q = dist / h;
                weight = std::pow(1.0f - q, 4) * (4.0f * q + 1.0f);
            } else {
                weight = 1.0f / (1.0f + dist * dist * 50.0f); // Fallback para distancias mayores
            }

            total_weight += weight;
            const auto& vertex = original_vertices[distances[i].second];

            result.position += vertex.position * weight;
            result.normal += vertex.normal * weight;
        }

        if (total_weight > epsilon) {
            result.position /= total_weight;
            result.normal /= total_weight;

            // Normalizar normal con mejor manejo de casos extremos
            float normal_length = glm::length(result.normal);
            if (normal_length > epsilon) {
                result.normal = result.normal / normal_length;
            } else {
                if (num_samples >= 3) {
                    glm::vec3 v1 = original_vertices[distances[1].second].position -
                                  original_vertices[distances[0].second].position;
                    glm::vec3 v2 = original_vertices[distances[2].second].position -
                                  original_vertices[distances[0].second].position;
                    glm::vec3 computed_normal = glm::normalize(glm::cross(v1, v2));
                    result.normal = computed_normal;
                } else {
                    result.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                }
            }
        }

        return result;
    }

    void generateHighResVerticesForNode(QuadtreeNode* node, std::vector<Vertex>& vertices) {
        auto [u_bounds, v_bounds] = node->getUVBounds();

        // Crear una grilla 5x5 de vértices por nodo para mayor resolución
        const int resolution = 5; // Incrementado de 3 a 5

        for (int i = 0; i < resolution; ++i) {
            for (int j = 0; j < resolution; ++j) {
                float u = u_bounds.first + (u_bounds.second - u_bounds.first) * i / (resolution - 1);
                float v = v_bounds.first + (v_bounds.second - v_bounds.first) * j / (resolution - 1);

                // Clamp coordenadas UV con margen más pequeño
                u = std::clamp(u, 0.001f, 0.999f);
                v = std::clamp(v, 0.001f, 0.999f);

                Vertex vertex = interpolateVertexAdvanced(u, v);
                vertices.push_back(vertex);
            }
        }
    }

    void generateIndicesForNode(unsigned int base_index, std::vector<unsigned int>& indices) {
        // Para una grilla 5x5, generar triángulos
        const int resolution = 5;

        for (int i = 0; i < resolution - 1; ++i) {
            for (int j = 0; j < resolution - 1; ++j) {
                unsigned int tl = base_index + i * resolution + j;       // top-left
                unsigned int tr = base_index + i * resolution + j + 1;   // top-right
                unsigned int bl = base_index + (i + 1) * resolution + j; // bottom-left
                unsigned int br = base_index + (i + 1) * resolution + j + 1; // bottom-right

                // Primer triángulo (tl, bl, tr)
                indices.push_back(tl);
                indices.push_back(bl);
                indices.push_back(tr);

                // Segundo triángulo (tr, bl, br)
                indices.push_back(tr);
                indices.push_back(bl);
                indices.push_back(br);
            }
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
    auto start_time = std::chrono::high_resolution_clock::now();

    ModelTessellator tessellator("Dragonite.FBX", 16);
    if (tessellator.original_vertices.empty()) {
        std::cerr << "Error: No se cargaron vértices del modelo" << std::endl;
        return -1;
    }

    // Generar geometría
    std::vector<Vertex> vertices = tessellator.generateVertices();
    std::vector<unsigned int> indices = tessellator.generateIndices();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "================================\n";
    std::cout << "Resumen de tiempos:\n";
    std::cout << " - Tiempo total de generación de geometría: " << total_duration.count() << " ms\n";
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