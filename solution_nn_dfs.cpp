#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>
#include <cmath>

//#include "model_weights.h"

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

// Shape: [20, 7]
const float layer1_weight[] = {
    -0.66389084, -0.17938717, 0.09002069, 0.76644737, -0.024630778, -0.3636426, -1.5655707, 0.23154745, 0.0845111, -0.27982166, -0.18891105, -0.33367375, -0.29384333, 0.15032718, -0.45735013, 0.08760719, -0.50571173, 0.998237, -9.897486e-05, 0.6116006, -0.54501855, 0.4018678, -0.62752956, 0.16056755, -0.7815277, -0.33605784, 0.10492528, 1.0146252, -0.3059654, -0.047742583, -0.14473857, -0.47677264, -0.95850194, 0.18334137, 1.2217054, -0.10742512, 0.13153204, -0.2025283, -0.3342942, -0.5156914, 1.83515, -0.94211996, -0.04258448, -0.33503473, -0.08130658, -0.023038656, 0.31911096, -0.23100606, -0.17127529, 0.14077254, -0.11862691, 0.043759007, -1.1804479, 1.2290167, -0.6920874, 0.73754674, 1.0975868, 0.050941616, -0.038605325, -0.2889235, -0.44720381, -0.8981907, 0.37686312, 0.6856472, -0.30222732, 0.047070652, 0.60381687, -0.2560018, 0.45799425, -0.59022266, 0.6493156, -2.1119416, -0.4401565, 0.8076474, 0.33635664, 0.13996302, -0.016112203, -0.2718842, -0.097953446, 0.07028379, 1.1716713, 0.47826442, -0.244837, -1.1262448, -0.09658347, -1.1261193, 0.1271544, -0.85636234, -0.13307495, 0.031028602, 1.1622044, -1.4738469, 1.6467141, -1.3429933, -0.054677308, -0.52558947, 0.20596054, -0.38711682, 0.23449227, -0.0859696, -0.33206633, -0.16517806, 0.27295277, 0.33253166, 0.053032607, 0.3617698, 0.11958264, -0.28167537, 0.08866644, 0.5264054, 0.78630143, -2.398054, 0.14523253, -0.21907374, -0.37377334, -0.08185929, 0.1612635, 0.21558496, -0.2933995, -0.6209251, -0.26736504, 0.0746293, 0.055090386, -0.13491102, 1.1023101, 0.8046462, -0.092035815, 0.032918293, -0.2144794, 1.7795572, -0.77015245, 0.1383797, -1.0429072, -0.09745458, 0.044614054, -0.04600077, -0.564047, 1.3765807, -0.08248072, 0.699098
};

const int layer1_weight_out = 20;
const int layer1_weight_in = 7;

// Shape: [20]
const float layer1_bias[] = {
    0.11113388, -0.07836115, 0.13031198, 0.09604987, 0.45508438, 0.20462324, -0.13913663, -0.028082361, 0.23992328, 0.29581934, -0.40872347, -0.097502895, 0.26650548, 0.5177326, -0.33425543, 0.2340111, -0.1313708, -0.2619314, 0.042632982, -0.43813613
};

// Shape: [1, 20]
const float layer2_weight[] = {
    -0.43981454, -0.18145227, 0.6031261, -0.23865733, -0.8201003, 1.7810662, -0.11982499, -2.7080305, -0.20887758, 0.7266803, 0.13191357, 1.1558307, -0.22190206, -1.1272582, -0.059956744, 1.3535683, -0.18224664, 0.75175625, -2.5192735, -1.057876
};

const int layer2_weight_out = 1;
const int layer2_weight_in = 20;

// Shape: [1]
const float layer2_bias[] = {
    0.024606882
};

#endif // MODEL_WEIGHTS_H


/*
MAXPATHS_GLOBAL_POOL: double the higher the more accurate and slower

UPDATE_NOW_AFTER_TICKS: int the higher the slower the alg gets. 
        It value states after how many calls of 
        timegate.updateNow() will now actually be updated 
        [now() call has relatively big overhead] 

TIME_LIMIT_SECONDS: double
*/

#define ENABLE_MAX_PATHS true
#define MAXPATHS_GLOBAL_POOL 500'000'000.0

#define ENABLE_TIME_LIMIT true
#define TIME_LIMIT_SECONDS 19.5 
#define UPDATE_NOW_AFTER_TICKS 300

#define SENTINEL_CHANGES_SEPARATOR -1

struct TimeGate {
#if ENABLE_TIME_LIMIT
    std::chrono::steady_clock::time_point startTime;
    bool timeExpired = false;
    int nextUpdate = UPDATE_NOW_AFTER_TICKS;

public:
    void init() {
        startTime = std::chrono::steady_clock::now();
        timeExpired = false;
        nextUpdate = UPDATE_NOW_AFTER_TICKS;
    }

    void updateNow() {
        if (nextUpdate <= 0) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - startTime;
            if (elapsed.count() > TIME_LIMIT_SECONDS) {
                timeExpired = true;
            }
            nextUpdate = UPDATE_NOW_AFTER_TICKS;
        } else {
            nextUpdate--;
        }
    }

    bool expired() const {
        return timeExpired;
    }
#else
    void init() {}
    void updateNow() {}
    bool expired() const { return false; }
#endif
};


// -- NN stuff --

// 1. Activation Function (ReLU)
void relu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

// 2. Dense (Linear) Layer Implementation
// Input: pointer to input array
// Weights: pointer to flattened weight array (Out x In)
// Bias: pointer to bias array (Out)
// Output: pointer to output array
// in_size: number of input features
// out_size: number of output features
void denseLayer(const float* input, const float* weights, const float* bias, float* output, int in_size, int out_size) {
    
    for (int i = 0; i < out_size; i++) {
        // Start with the bias for this output neuron
        float sum = bias[i];
        
        // Perform Dot Product (Matrix Multiplication)
        for (int j = 0; j < in_size; j++) {
            // Access weight: Row-Major order. 
            // PyTorch weights are [out_features][in_features].
            // Flat index = i * in_size + j
            sum += input[j] * weights[i * in_size + j];
        }
        
        output[i] = sum;
    }
}

float forwardPropagation(const std::vector<float>& input_data) {
    float hidden_layer[layer1_weight_out]; // Buffer for layer 1 output
    float final_output[layer2_weight_out]; // Buffer for layer 2 output

    // --- Layer 1 Forward ---
    // Use the variable names generated in model_weights.h (e.g., fc1_weight, fc1_bias)
    denseLayer(input_data.data(), layer1_weight, layer1_bias, hidden_layer, layer1_weight_in, layer1_weight_out);
    
    // --- Activation (ReLU) ---
    relu(hidden_layer, layer1_weight_out);
    
    // --- Layer 2 Forward ---
    denseLayer(hidden_layer, layer2_weight, layer2_bias, final_output, layer2_weight_in, layer2_weight_out);

    return final_output[0]; // i know... hardcoded magic number but idc and the NN only has 1 output
}

// number of features cannot be larger than 20 for small problem (ideally smaller than 10)
// N - number of nodes
// F - number of features [input layer]
// H - number of hidden nodes [middle/hidden layer] (preferably larger than F)
// (output layer's size is just 1 so i dont include it)
// Forward propagation of the network: O(NFH) + we also need to run DFS/some other pathfinding alg later and we only have 20s
struct NodeFeaturesNormalized {
    float degree;
    float min_neighbour_degree;
    float max_neighbour_degree;
    float arithmetic_mean_neighbour_degree;
    float harmonic_mean_neighbour_degree;
    float arithmetic_mean_global_degree;
    float harmonic_mean_global_degree;

    std::vector<float> to_vector() {
        std::vector<float> data = { 
            degree,
            min_neighbour_degree,
            max_neighbour_degree,
            arithmetic_mean_neighbour_degree,
            harmonic_mean_neighbour_degree,
            arithmetic_mean_global_degree,
            harmonic_mean_global_degree
        };
        return data;
    }
};

float normalize(float x, float min_val, float max_val) {
    float diff = std::abs(max_val - min_val);
    if (diff < 1e-9) {
        return 0.0f;
    }
    return (x - min_val) / (max_val - min_val);
}


std::vector<NodeFeaturesNormalized> calculateFeatures(const std::vector<std::vector<int>>& g) {
    int n = g.size();

    std::vector<float> degrees_log(n, 0);
    
    // 1. Pre-calculate degrees for all nodes
    for(int i = 0; i < n; ++i) {
        // we do log here to give higher impact to lower weights 
        // (distinction between 2 and 4 degree should be bigger than 1001 degree and 1030 degree)
        // we need to use log1p to avoid division by 0 later while calculating harmonic mean
        degrees_log[i] = log1p(g[i].size()); 
    }

    // 2. Calculate Global Statistics
    double global_sum = 0;
    double global_reciprocal_sum = 0;

    float global_min_d = std::numeric_limits<float>::max();
    float global_max_d = 0.0f;

    for(float d : degrees_log) {
        global_min_d = std::min(global_min_d, d);
        global_max_d = std::max(global_max_d, d);

        global_sum += d;
        if (d > 0) {
            global_reciprocal_sum += 1.0 / d;
        }
    }

    float global_arithmetic_mean = (n > 0) ? (float)(global_sum / n) : 0.0f;
    
    // Note: Technically Harmonic mean of a set containing 0 is 0. 
    // Since we treated isolated nodes as 0, checking if global_reciprocal_sum > 0 is a proxy.
    float global_harmonic_mean = 0.0f;
    if (global_reciprocal_sum > 1e-9) {
        global_harmonic_mean = (float)(n / global_reciprocal_sum);
    }

    float norm_global_arith = normalize(global_arithmetic_mean, global_min_d, global_max_d);
    float norm_global_harm = normalize(global_harmonic_mean, global_min_d, global_max_d);

    // 3. Calculate Node Features
    std::vector<NodeFeaturesNormalized> features(n);

    for (int i = 0; i < n; ++i) {
        NodeFeaturesNormalized& f = features[i];
        
        // A. Basic Degree
        f.degree = normalize(degrees_log[i], global_min_d, global_max_d);
        
        // B. Global Stats (Same for all nodes)
        f.arithmetic_mean_global_degree = norm_global_arith;
        f.harmonic_mean_global_degree = norm_global_harm;

        // C. Neighbor Stats
        if (degrees_log[i] == 0) {
            // Handle isolated nodes
            // NOTE: there will be no isolated nodes
            f.min_neighbour_degree = 0.0f;
            f.max_neighbour_degree = 0.0f;
            f.arithmetic_mean_neighbour_degree = 0.0f;
            f.harmonic_mean_neighbour_degree = 0.0f;
        } else {
            float min_d = std::numeric_limits<float>::max();
            float max_d = 0.0f;
            double sum_d = 0;
            double sum_reciprocal_d = 0;

            for (int neighbor : g[i]) {
                float d_neighbor = degrees_log[neighbor];
                
                if (d_neighbor < min_d) min_d = d_neighbor;
                if (d_neighbor > max_d) max_d = d_neighbor;
                
                sum_d += d_neighbor;
                sum_reciprocal_d += 1.0 / d_neighbor;
            }

            f.min_neighbour_degree = normalize(min_d, global_min_d, global_max_d);
            f.max_neighbour_degree = normalize(max_d, global_min_d, global_max_d);
            f.arithmetic_mean_neighbour_degree = normalize((float)(sum_d / g[i].size()), global_min_d, global_max_d);
            f.harmonic_mean_neighbour_degree = normalize((float)(g[i].size() / sum_reciprocal_d), global_min_d, global_max_d);
        }
    }

    return features;
}

// Represents an unweighted undirected graph
class Graph {
public:
    std::vector<std::vector<int>> adj;
    std::vector<float> scores;
    float totalScore;

    Graph(int V) : totalScore(0.0) {
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int numOfVertices() const {
        return adj.size();
    }

    void afterInputSort() {

        auto features = calculateFeatures(adj);

        scores = std::vector<float>(adj.size());
        for (int i = 0; i < adj.size(); i++) {
            scores[i] = forwardPropagation(features[i].to_vector());
            totalScore += scores[i];
        }

        for (int i = 0; i < adj.size(); i++) {
            sort(adj[i].begin(), adj[i].end(), [&](int a, int b) {
                return scores[a] > scores[b];
            });
        }
    }
};

class LIPPSolver {
private:
    // Corresponds to 'P_max' in Algorithm 3/4
    std::vector<int> m_path_max;
    
    // Corresponds to 'P_temp' in Algorithm 3/4
    std::vector<int> m_path_temp;

    TimeGate m_timegate;

    std::vector<int> m_changes;

    const Graph m_graph;

    std::vector<bool> m_valid_vertices;

    std::vector<bool> m_valid_neighbour_store;

    int m_current_max_paths;

    int m_num_paths;
    int m_last_improv;
    bool m_truncated;

    // Helper to check if the solution is better and update
    void updateIfBetter() {
        if (m_path_temp.size() > m_path_max.size()) {
            m_path_max = m_path_temp;
            m_last_improv = m_num_paths; // Update last improvement timestamp
        }
    }


    // Algorithm 4: FIRST-INDUCED-PATHS
    // Cites: [cite: 188, 201]
    void dfs(int s) {
        
        m_timegate.updateNow();
        if (m_timegate.expired()) return;

        // Line 1: Append s to P_temp
        m_path_temp.push_back(s);

        bool no_valid_neighbors = true;

        m_changes.push_back(SENTINEL_CHANGES_SEPARATOR);
        
        for (int t : m_graph.adj[s]) {
            if (m_valid_vertices[t]) {
                m_changes.push_back(t);
                m_valid_vertices[t] = false;
            }
        }

        // all neighbours are removed

        for (int idx = m_changes.size() - 1; m_changes[idx] != SENTINEL_CHANGES_SEPARATOR; idx-- ) {


            if (m_timegate.expired()) break;

            no_valid_neighbors = false;

            int t = m_changes[idx];

            // Line 8: Recursive Call
            dfs(t);

            // Line 9-11: Propagate truncation
            if (m_truncated) {
                break;
            }
            if (m_timegate.expired()) return;
        }

        // cleanup
        while(m_changes.back() != SENTINEL_CHANGES_SEPARATOR) {
            m_valid_vertices[m_changes.back()] = true;
            m_changes.pop_back();
        }
        m_changes.pop_back(); // pop sentinel

        if (no_valid_neighbors) {
            // Line 13: Else (Leaf node logic)
            // Line 14: Increment #paths
            m_num_paths++;

            // Line 15-17: Check for improvement
            updateIfBetter();

            // Line 19: Stopping criterion
            if ((m_num_paths - m_last_improv) > m_current_max_paths && ENABLE_MAX_PATHS) {
                // Line 20-21: Truncate
                // Note: We don't clear P_temp here completely because C++ recursion 
                // needs to unwind, but we set the flag to stop exploration.
                m_truncated = true;
                m_path_temp.pop_back(); // Clean up current node
                return;
            }
        }

        // Line 25: Remove s from P_temp (Backtracking)
        m_path_temp.pop_back();
    }

public:
    
    LIPPSolver(TimeGate&& timegate, Graph&& g) : m_timegate(std::move(timegate)), m_graph(std::move(g)) {}

    // Algorithm 3: HLIPP
    // Cites: [cite: 181, 187]
    const std::vector<int>& hlipp() {
        // Line 1: P_max <- empty
        m_path_max.clear();
        
        // Line 2: P_temp <- empty
        m_path_temp.clear();

        // Initial valid vertices set (all vertices are valid at start)
        m_valid_vertices = std::vector<bool>(m_graph.numOfVertices(), true);

        // Line 3: Loop through all vertices
        // Note: The paper implies an ordering or random access. 
        // Iterating 0 to V-1 is standard.
        for (int s = 0; s < m_graph.numOfVertices(); ++s) {
            
            // Line 4: #paths <- 0
            m_num_paths = 0;
            
            // Line 5: last_improv <- 0
            m_last_improv = 0;
            
            // Line 6: truncated <- False
            m_truncated = false;

            if (m_timegate.expired()) break;

            
            m_current_max_paths =  (int)(((double)m_graph.scores[s] / (double)m_graph.totalScore) * MAXPATHS_GLOBAL_POOL);

            m_valid_vertices[s] = false;


            // Line 7: Call recursive procedure
            dfs(s);

            m_valid_vertices[s] = true;

        }

        // Line 9: Return P_max
        return m_path_max;
    }
};

// --- Helper to print vector ---
void printPath(const std::vector<int>& path) {
    std::cout << path.size() << "\n";
    for (int v : path) {
        std::cout << v << " ";
    }
    std::cout << "\n";
}

// --- Main Driver for Testing ---
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);

    TimeGate timegate;
    timegate.init();

    int n, m;
    std::cin >> n >> m;

    Graph g(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        std::cin >> u >> v;
        g.addEdge(u, v);

    }

    g.afterInputSort();

    LIPPSolver solver(std::move(timegate), std::move(g));
    
    // Run HLIPP with maxpaths = 100 (sufficient for this small graph)
    // According to Table 3[cite: 378], maxpaths allows the heuristic to explore 
    // deeper before giving up on a specific start node.
    const std::vector<int>& result = solver.hlipp();

    printPath(result);

    return 0;
}