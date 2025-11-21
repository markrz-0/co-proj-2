#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>
#include <cmath>
#include <tuple>

#include "model_weights.h"

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

struct NNOutput {
    float chanceOfNodeBeingInPath;
    float chanceOfNodeBeingStarterNode;
};

NNOutput forwardPropagation(const std::vector<float>& input_data) {
    float hidden_layer[layer1_weight_out]; // Buffer for layer 1 output
    float final_output[layer2_weight_out]; // Buffer for layer 2 output

    // --- Layer 1 Forward ---
    // Use the variable names generated in model_weights.h (e.g., fc1_weight, fc1_bias)
    denseLayer(input_data.data(), layer1_weight, layer1_bias, hidden_layer, layer1_weight_in, layer1_weight_out);
    
    // --- Activation (ReLU) ---
    relu(hidden_layer, layer1_weight_out);
    
    // --- Layer 2 Forward ---
    denseLayer(hidden_layer, layer2_weight, layer2_bias, final_output, layer2_weight_in, layer2_weight_out);

    NNOutput out;

    out.chanceOfNodeBeingInPath = final_output[0];
    out.chanceOfNodeBeingStarterNode = final_output[1];

    return out;// i know... hardcoded magic number but idc and the NN only has 1 output
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
    std::vector<NNOutput> scores;
    float totalScoreStarterNode;

    Graph(int V) : totalScoreStarterNode(0.0) {
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

        scores = std::vector<NNOutput>(adj.size());
        for (int i = 0; i < adj.size(); i++) {
            scores[i] = forwardPropagation(features[i].to_vector());
            totalScoreStarterNode += scores[i].chanceOfNodeBeingStarterNode;
        }

        for (int i = 0; i < adj.size(); i++) {
            sort(adj[i].begin(), adj[i].end(), [&](int a, int b) {
                return scores[a].chanceOfNodeBeingInPath > scores[b].chanceOfNodeBeingInPath;
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

        std::vector<std::tuple<float, float, int>> nodes(m_graph.numOfVertices());

        for (int i = 0; i < m_graph.numOfVertices(); i++) {
            nodes[i] = std::make_tuple(-m_graph.scores[i].chanceOfNodeBeingStarterNode, -m_graph.scores[i].chanceOfNodeBeingInPath, i);
        }

        std::sort(nodes.begin(), nodes.end());

        for (auto node : nodes) {

            int s = std::get<2>(node);
            
            // Line 4: #paths <- 0
            m_num_paths = 0;
            
            // Line 5: last_improv <- 0
            m_last_improv = 0;
            
            // Line 6: truncated <- False
            m_truncated = false;

            if (m_timegate.expired()) break;

            
            m_current_max_paths =  (int)(((double)-std::get<0>(node) / (double)m_graph.totalScoreStarterNode) * MAXPATHS_GLOBAL_POOL);

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