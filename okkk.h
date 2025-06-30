//
// Created by lvera on 29/06/2025.
//

#ifndef OKKK_H
#define OKKK_H

#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

enum class NodeType {
    INTERNAL,
    REGULAR,
    CREASE,
    SPECIAL,
    TERMINAL
};

class Template {
private:
    std::vector<int> vertex_indices;
    std::vector<float> weights;

public:
    Template(const std::vector<int>& indices, const std::vector<float>& w)
        : vertex_indices(indices), weights(w) {
        if (vertex_indices.size() != weights.size()) {
            throw std::invalid_argument("Vertex indices and weights must have the same length");
        }

        if (vertex_indices.empty()) {
            throw std::invalid_argument("Template must have at least one vertex");
        }

        float weight_sum = 0.0f;
        for (float w : weights) {
            weight_sum += w;
        }

        if (weight_sum < 0.9f || weight_sum > 1.1f) {
            std::cout << "Warning: Template weights sum to " << weight_sum << ", expected ~1.0\n";
        }
    }

    float evaluate(const std::vector<float>& one_ring_values) const {
        int max_index = *std::max_element(vertex_indices.begin(), vertex_indices.end());
        if (one_ring_values.size() < static_cast<size_t>(max_index + 1)) {
            throw std::invalid_argument("Need at least " + std::to_string(max_index + 1) +
                                      " one-ring values, got " + std::to_string(one_ring_values.size()));
        }

        float result = 0.0f;
        for (size_t i = 0; i < vertex_indices.size(); ++i) {
            result += weights[i] * one_ring_values[vertex_indices[i]];
        }

        return result;
    }

    size_t getVertexCount() const { return vertex_indices.size(); }
    float getWeightSum() const {
        float sum = 0.0f;
        for (float w : weights) sum += w;
        return sum;
    }
};

std::unique_ptr<Template> generateRegularTemplate(const std::vector<int>& one_ring_vertices) {
    const int num_control_points = 16;
    std::vector<int> vertex_indices;

    for (int i = 0; i < num_control_points; ++i) {
        vertex_indices.push_back(one_ring_vertices[i % one_ring_vertices.size()]);
    }

    float uniform_weight = 1.0f / num_control_points;
    std::vector<float> weights(num_control_points, uniform_weight);

    return std::make_unique<Template>(vertex_indices, weights);
}

std::unique_ptr<Template> generateTerminalTemplate(const std::vector<int>& one_ring_vertices) {
    const int num_control_points = 24;
    std::vector<int> vertex_indices;

    for (int i = 0; i < num_control_points; ++i) {
        vertex_indices.push_back(one_ring_vertices[i % one_ring_vertices.size()]);
    }

    float uniform_weight = 1.0f / num_control_points;
    std::vector<float> weights(num_control_points, uniform_weight);

    return std::make_unique<Template>(vertex_indices, weights);
}

std::unique_ptr<Template> generateSpecialTemplate(const std::vector<int>& one_ring_vertices, int template_id = 0) {
    if (template_id < 0 || template_id > 2) {
        throw std::invalid_argument("Special template_id must be 0, 1, or 2, got " + std::to_string(template_id));
    }

    int num_control_points;
    std::vector<float> weights;

    if (template_id == 0) {
        num_control_points = 8;
        weights = {0.3f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f, 0.05f, 0.05f};
    } else if (template_id == 1) {
        num_control_points = 6;
        weights = {0.25f, 0.2f, 0.2f, 0.15f, 0.1f, 0.1f};
    } else {
        num_control_points = 4;
        weights = {0.4f, 0.3f, 0.2f, 0.1f};
    }

    std::vector<int> vertex_indices;
    for (int i = 0; i < num_control_points; ++i) {
        vertex_indices.push_back(one_ring_vertices[i % one_ring_vertices.size()]);
    }

    return std::make_unique<Template>(vertex_indices, weights);
}

class QuadtreeNode {
private:
    NodeType node_type;
    int depth;
    float u, v;
    QuadtreeNode* parent;
    std::vector<std::unique_ptr<QuadtreeNode>> children;
    int template_id;
    std::unique_ptr<Template> node_template;

public:
    QuadtreeNode(NodeType type, int d, float u_coord, float v_coord, QuadtreeNode* p = nullptr)
        : node_type(type), depth(d), u(u_coord), v(v_coord), parent(p), template_id(-1) {
        if (depth < 0) {
            throw std::invalid_argument("Depth must be non-negative");
        }

        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
            throw std::invalid_argument("UV coordinates must be in range [0, 1]");
        }

        if (node_type == NodeType::INTERNAL && children.size() > 4) {
            throw std::invalid_argument("Internal nodes cannot have more than 4 children");
        }
    }

    std::vector<QuadtreeNode*> subdivide() {
        if (!isLeaf()) {
            throw std::runtime_error("Cannot subdivide non-leaf node at depth " + std::to_string(depth));
        }

        float current_size = (depth > 0) ? 1.0f / static_cast<float>(1 << depth) : 1.0f;
        float child_size = current_size / 2.0f;
        float quarter_offset = child_size / 2.0f;

        auto [u_bounds, v_bounds] = getUVBounds();
        float u_min = u_bounds.first, u_max = u_bounds.second;
        float v_min = v_bounds.first, v_max = v_bounds.second;

        float u_left = u_min + quarter_offset;
        float u_right = u_max - quarter_offset;
        float v_bottom = v_min + quarter_offset;
        float v_top = v_max - quarter_offset;

        std::vector<int> simulated_one_ring;
        for (int i = 0; i < 20; ++i) {
            simulated_one_ring.push_back(i);
        }

        children.reserve(4);

        auto child0 = std::make_unique<QuadtreeNode>(NodeType::REGULAR, depth + 1, u_left, v_bottom, this);
        child0->node_template = generateRegularTemplate(simulated_one_ring);

        auto child1 = std::make_unique<QuadtreeNode>(NodeType::REGULAR, depth + 1, u_right, v_bottom, this);
        child1->node_template = generateRegularTemplate(simulated_one_ring);

        auto child2 = std::make_unique<QuadtreeNode>(NodeType::REGULAR, depth + 1, u_left, v_top, this);
        child2->node_template = generateRegularTemplate(simulated_one_ring);

        auto child3 = std::make_unique<QuadtreeNode>(NodeType::REGULAR, depth + 1, u_right, v_top, this);
        child3->node_template = generateRegularTemplate(simulated_one_ring);

        std::vector<QuadtreeNode*> child_ptrs;
        child_ptrs.push_back(child0.get());
        child_ptrs.push_back(child1.get());
        child_ptrs.push_back(child2.get());
        child_ptrs.push_back(child3.get());

        for (size_t i = 0; i < child_ptrs.size(); ++i) {
            float child_u = child_ptrs[i]->u;
            float child_v = child_ptrs[i]->v;
            if (child_u < 0.0f || child_u > 1.0f || child_v < 0.0f || child_v > 1.0f) {
                throw std::runtime_error("Child " + std::to_string(i) + " UV coordinates (" +
                                       std::to_string(child_u) + ", " + std::to_string(child_v) +
                                       ") are outside valid range [0, 1]");
            }
        }

        children.push_back(std::move(child0));
        children.push_back(std::move(child1));
        children.push_back(std::move(child2));
        children.push_back(std::move(child3));

        node_type = NodeType::INTERNAL;

        return child_ptrs;
    }

    bool isLeaf() const {
        return children.empty();
    }

    std::pair<std::pair<float, float>, std::pair<float, float>> getUVBounds() const {
        float size = (depth > 0) ? 1.0f / static_cast<float>(1 << depth) : 1.0f;
        float half_size = size / 2.0f;

        std::pair<float, float> u_bounds = {u - half_size, u + half_size};
        std::pair<float, float> v_bounds = {v - half_size, v + half_size};

        return {u_bounds, v_bounds};
    }

    QuadtreeNode* query(float query_u, float query_v) {
        if (query_u < 0.0f || query_u > 1.0f || query_v < 0.0f || query_v > 1.0f) {
            throw std::invalid_argument("UV coordinates (" + std::to_string(query_u) + ", " +
                                      std::to_string(query_v) + ") are outside valid range [0, 1]");
        }

        auto [u_bounds, v_bounds] = getUVBounds();
        float u_min = u_bounds.first, u_max = u_bounds.second;
        float v_min = v_bounds.first, v_max = v_bounds.second;

        if (!(u_min <= query_u && query_u <= u_max && v_min <= query_v && query_v <= v_max)) {
            return nullptr;
        }

        if (isLeaf()) {
            return this;
        }

        if (children.size() != 4) {
            return nullptr;
        }

        int child_index;
        if (query_u < u && query_v < v) {
            child_index = 0;
        } else if (query_u >= u && query_v < v) {
            child_index = 1;
        } else if (query_u < u && query_v >= v) {
            child_index = 2;
        } else {
            child_index = 3;
        }

        return children[child_index]->query(query_u, query_v);
    }

    float evaluateControlPoint(const std::vector<float>& one_ring_values) {
        if (!node_template) {
            throw std::invalid_argument("No template assigned to node at depth " + std::to_string(depth));
        }

        return node_template->evaluate(one_ring_values);
    }

    void assignTemplateByType(const std::vector<int>& one_ring_vertices) {
        switch (node_type) {
            case NodeType::REGULAR:
                node_template = generateRegularTemplate(one_ring_vertices);
                break;
            case NodeType::TERMINAL:
                node_template = generateTerminalTemplate(one_ring_vertices);
                break;
            case NodeType::SPECIAL: {
                int tid = (template_id != -1) ? template_id : 0;
                node_template = generateSpecialTemplate(one_ring_vertices, tid % 3);
                break;
            }
            case NodeType::CREASE:
                node_template = generateRegularTemplate(one_ring_vertices);
                break;
            case NodeType::INTERNAL:
                break;
        }
    }

    NodeType getNodeType() const { return node_type; }
    int getDepth() const { return depth; }
    float getU() const { return u; }
    float getV() const { return v; }
    size_t getChildrenCount() const { return children.size(); }
    QuadtreeNode* getChild(size_t index) const {
        return (index < children.size()) ? children[index].get() : nullptr;
    }

    void setTemplateId(int id) { template_id = id; }
    int getTemplateId() const { return template_id; }

    bool hasTemplate() const { return node_template != nullptr; }
};

class QuadtreeBuilder {
private:
    int max_depth;

public:
    QuadtreeBuilder(int max_d = 8) : max_depth(max_d) {}

    std::unique_ptr<QuadtreeNode> buildFromMesh(void* mesh_data) {
        return std::make_unique<QuadtreeNode>(NodeType::REGULAR, 0, 0.5f, 0.5f);
    }

    int getMaxDepth() const { return max_depth; }
    void setMaxDepth(int depth) { max_depth = depth; }
};



#endif //OKKK_H
