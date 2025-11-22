#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>
#include <cmath>
#include <tuple>

#include "model_weights.h"

/*
UPDATE_NOW_AFTER_TICKS: int the higher the slower the alg gets. 
        It value states after how many calls of 
        timegate.updateNow() will now actually be updated 
        [now() call has relatively big overhead] 

TIME_LIMIT_SECONDS: double
*/


#define ENABLE_TIME_LIMIT true
#define TIME_LIMIT_SECONDS 19.5 
#define UPDATE_NOW_AFTER_TICKS 10

struct TimeGate {
#if ENABLE_TIME_LIMIT
    std::chrono::steady_clock::time_point m_start_time;
    bool m_time_expired = false;
    int m_next_update = UPDATE_NOW_AFTER_TICKS;

public:
    void init() {
        m_start_time = std::chrono::steady_clock::now();
        m_time_expired = false;
        m_next_update = UPDATE_NOW_AFTER_TICKS;
    }

    void updateNow() {
        if (m_next_update <= 0) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - m_start_time;
            if (elapsed.count() > TIME_LIMIT_SECONDS) {
                m_time_expired = true;
            }
            m_next_update = UPDATE_NOW_AFTER_TICKS;
        } else {
            m_next_update--;
        }
    }

    bool expired() const {
        return m_time_expired;
    }
#else
    void init() {}
    void updateNow() {}
    bool expired() const { return false; }
#endif
};


// -- NN stuff --

// 1. Activation Function (ReLU)
void leakyRelu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] < 0) {
            input[i] = input[i] * 0.01;
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
    float hidden_layer1[layer1_weight_out]; // Buffer for layer 1 output
    float hidden_layer2[layer2_weight_out]; // Buffer for layer 2 output
    float final_output[layer3_weight_out]; // Buffer for layer 3 output

    // --- Layer 1 Forward ---
    // Use the variable names generated in model_weights.h (e.g., fc1_weight, fc1_bias)
    denseLayer(input_data.data(), layer1_weight, layer1_bias, hidden_layer1, layer1_weight_in, layer1_weight_out);
    
    // --- Activation (ReLU) ---
    leakyRelu(hidden_layer1, layer1_weight_out);
    
    // --- Layer 2 Forward ---
    denseLayer(hidden_layer1, layer2_weight, layer2_bias, hidden_layer2, layer2_weight_in, layer2_weight_out);
    
    leakyRelu(hidden_layer2, layer2_weight_out);

    denseLayer(hidden_layer2, layer3_weight, layer3_bias, final_output, layer3_weight_in, layer3_weight_out);

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
    float clustering_coeff;
    float mean_neighbour_clustering_coeff;
    float mean_global_clustering_coeff;

    std::vector<float> to_vector() {
        std::vector<float> data = { 
            degree,
            min_neighbour_degree,
            max_neighbour_degree,
            arithmetic_mean_neighbour_degree,
            harmonic_mean_neighbour_degree,
            arithmetic_mean_global_degree,
            harmonic_mean_global_degree,
            clustering_coeff,
            mean_neighbour_clustering_coeff,
            mean_global_clustering_coeff
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


// Represents the graph
std::vector<float> getClusteringCoeffs(std::vector<std::vector<int>>& g) {
    int n = g.size();
    std::vector<int> degree(n, 0);

    // calc degree
    for (int i = 0; i < n; i++) {
        degree[i] = g[i].size();
    }

    // 2. Sort adjacency lists for faster intersection (optional but good)
    for(int i=0; i<n; i++) {
        std::sort(g[i].begin(), g[i].end());
    }

    // 3. Build Directed Graph (Low -> High)
    std::vector<std::vector<int>> dag_adj(n);
    for (int u = 0; u < n; u++) {
        for (int v : g[u]) {
            if (degree[u] < degree[v] || (degree[u] == degree[v] && u < v)) {
                dag_adj[u].push_back(v);
            }
        }
    }

    // 4. Count Triangles
    std::vector<int> triangles(n, 0);
    for (int u = 0; u < n; u++) {
        for (int v : dag_adj[u]) {
            for (int w : dag_adj[v]) {
                // Check if edge (u, w) exists using binary search (std::binary_search)
                // We check original adj because direction might be u->w or w->u
                if (std::binary_search(g[u].begin(), g[u].end(), w)) {
                    triangles[u]++;
                    triangles[v]++;
                    triangles[w]++;
                }
            }
        }
    }

    // 5. Calc Metrics
    std::vector<float> coeffs(n);
    for(int i=0; i<n; i++) {
        long long d = degree[i];
        if (d > 1) {
            coeffs[i] = (2.0 * triangles[i]) / (d * (d - 1));
        } else {
            coeffs[i] = 0.0;
        }
    }
    return coeffs;
}



std::vector<NodeFeaturesNormalized> calculateFeatures(std::vector<std::vector<int>>& g) {
    int n = g.size();

    std::vector<float> degrees_log(n, 0);
    
    // 1. Pre-calculate degrees for all nodes
    for(int i = 0; i < n; ++i) {
        // we do log here to give higher impact to lower weights 
        // (distinction between 2 and 4 degree should be bigger than 1001 degree and 1030 degree)
        // we need to use log1p to avoid division by 0 later while calculating harmonic mean
        degrees_log[i] = log1p(g[i].size()); 
    }

    std::vector<float> coeffs = getClusteringCoeffs(g);
    double running_sum = 0.0f;
    for (double coeff : coeffs) {
        running_sum += coeff;
    }
    double global_mean_coeff = running_sum / n;

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
            double running_sum = 0.0;

            for (int neighbor : g[i]) {
                float d_neighbor = degrees_log[neighbor];
                
                if (d_neighbor < min_d) min_d = d_neighbor;
                if (d_neighbor > max_d) max_d = d_neighbor;
                
                sum_d += d_neighbor;
                sum_reciprocal_d += 1.0 / d_neighbor;
                running_sum += coeffs[neighbor];
            }

            f.min_neighbour_degree = normalize(min_d, global_min_d, global_max_d);
            f.max_neighbour_degree = normalize(max_d, global_min_d, global_max_d);
            f.arithmetic_mean_neighbour_degree = normalize((float)(sum_d / g[i].size()), global_min_d, global_max_d);
            f.harmonic_mean_neighbour_degree = normalize((float)(g[i].size() / sum_reciprocal_d), global_min_d, global_max_d);
            f.clustering_coeff = coeffs[i];
            f.mean_neighbour_clustering_coeff = running_sum / g[i].size();
            f.mean_global_clustering_coeff = global_mean_coeff;
        }
    }

    return features;
}


// Represents an unweighted undirected graph
class Graph {
public:
    std::vector<std::vector<int>> adj;
    std::vector<NNOutput> scores;

    Graph(int V)  {
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int numOfVertices() const {
        return adj.size();
    }

    void postInit() {

        auto features = calculateFeatures(adj);

        scores = std::vector<NNOutput>(adj.size());
        for (int i = 0; i < adj.size(); i++) {
            scores[i] = forwardPropagation(features[i].to_vector());
        }
    }
};


// --- Individual Structure ---
class Individual {
public:
    std::vector<int> m_path;
    
    std::vector<bool> m_second_neighbour; 
    std::vector<bool> m_first_neighbour;
    std::vector<bool> m_in_path;
    
    float m_fitness;
    float max_valid_extension_score = 0.0f;

    Individual(int n, float total_score) : m_fitness(total_score), m_first_neighbour(n, false), m_second_neighbour(n, false), m_in_path(n, false), max_valid_extension_score(0.0f) {}

    float getFitness() const {
        return m_fitness + max_valid_extension_score;
    }

    // Safely add a node and update state
    void pushNode(const Graph& g, int u) {
        m_path.push_back(u);
        m_in_path[u] = true;
        // For every neighbor of u, increment their connection count to the path
        max_valid_extension_score = 0.0f;
        for (int v : g.adj[u]) {
            if (!m_first_neighbour[v]) {
                m_fitness -= g.scores[v].chanceOfNodeBeingInPath;
                m_first_neighbour[v] = true;
                if (!m_in_path[v]) {
                    max_valid_extension_score = std::max(max_valid_extension_score, g.scores[v].chanceOfNodeBeingInPath);
                }
            } else {
                m_second_neighbour[v] = true;
            }
        }
    }

    // Check validity in O(1)
    // Candidate must be a neighbor of path.back() (checked by caller via adjacency list)
    // Constraint: Candidate must NOT connect to any OTHER node in the path.
    // Therefore, neighborCounts[candidate] must be exactly 1 (the connection to path.back())
    bool isValidExtension(int candidate) const {
        if (m_in_path[candidate]) return false;
        // If neighborCounts is 1, it means it connects ONLY to the current tail (valid).
        // If > 1, it connects to tail AND someone earlier (chord -> invalid).
        // If 0, it doesn't connect to tail (shouldn't happen if iterating adj[tail]).
        return m_first_neighbour[candidate] && !m_second_neighbour[candidate];
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

class Solver {
    Graph m_graph;
    TimeGate m_time_gate;
    int m_beam_width;
    int m_dfs_size;
    std::vector<int> m_max_path;

public:
    Solver(Graph&& g, int beam_width, int dfs_size, TimeGate timegate) : m_beam_width(beam_width), m_dfs_size(dfs_size), m_graph(std::move(g)), m_time_gate(timegate) {}

    const std::vector<int>& run() {

        // init
        std::vector<std::tuple<float, float, int>> nodes(m_graph.numOfVertices());

        
        
        float total_score = 0.0f;
        for (int i = 0; i < m_graph.numOfVertices(); i++) {
            nodes[i] = std::make_tuple(
                -m_graph.scores[i].chanceOfNodeBeingStarterNode,
                -m_graph.scores[i].chanceOfNodeBeingInPath,
                i);
            total_score += m_graph.scores[i].chanceOfNodeBeingInPath;
        }
            
        std::sort(nodes.begin(), nodes.end());

        // dfs
        for (int i = 0; i < m_dfs_size; i++) {
            int path_length = 1;
            Individual ind(m_graph.numOfVertices(), total_score);
            ind.pushNode(m_graph, std::get<2>(nodes[i]));
            while(true) {
                bool extended = false;
                float best_score = -1.0f;
                int best_extenson = 0;
                for (int next : m_graph.adj[ind.m_path.back()]) {
                    if (ind.isValidExtension(next)) {
                        if (m_graph.scores[next].chanceOfNodeBeingInPath >= best_score) {
                            extended = true;
                            best_score = m_graph.scores[next].chanceOfNodeBeingInPath;
                            best_extenson = next;
                        }
                    }
                }
                if (!extended) {
                    if (ind.m_path.size() > m_max_path.size()) {
                        m_max_path = ind.m_path;
                    }
                    break;
                }
                ind.pushNode(m_graph, best_extenson);
            }
        }


        // beamsearch
        int batch_start = 0;
        int batch_end = std::min(m_beam_width, m_graph.numOfVertices());

        std::vector<Individual> population(std::min(m_beam_width, m_graph.numOfVertices()), Individual(m_graph.numOfVertices(), total_score));
        
        again:
        for (int i = 0; i < (batch_end - batch_start); i++) {
            population[i].pushNode(m_graph, std::get<2>(nodes[i]));
        }
        
        bool something_extended = true;
        int actual_beam_width = batch_end - batch_start;
        while(!m_time_gate.expired() && something_extended) {
            m_time_gate.updateNow();

            something_extended = false;

            int new_pops = 0;
            for(int i = 0; i < actual_beam_width; i++) {
                for (int next : m_graph.adj[population[i].m_path.back()]) {
                    if (population[i].isValidExtension(next)) {
                        int idx = actual_beam_width + new_pops;
                        if (idx < population.size()) {
                            population[idx] = population[i];
                            population[idx].pushNode(m_graph, next);
                        } else {
                            Individual new_ind = population[i];
                            new_ind.pushNode(m_graph, next);
                            population.push_back(new_ind);
                        }
                        something_extended = true;
                        new_pops++;
                    }
                }
            }

            // move stuff
            for (int i = 0; i < new_pops; i++) {
                std::swap(population[i], population[i + actual_beam_width]);
            }

            // sort only relevant
            std::sort(population.begin(), population.begin() + new_pops, [&](const Individual& i1, const Individual& i2) {
                if (i1.m_path.size() == i2.m_path.size()) {
                    return i1.getFitness() > i2.getFitness();
                }
                return i1.m_path.size() > i2.m_path.size();
            });

            actual_beam_width = std::min(new_pops, m_beam_width);
        }
        
        if (population[0].m_path.size() > m_max_path.size()) {
            m_max_path = population[0].m_path;
        }

        if (!m_time_gate.expired()) {
            batch_start = batch_end;
            batch_end = std::min((batch_end + m_beam_width), m_graph.numOfVertices());
            if (batch_start != m_graph.numOfVertices()) {
                for (int i = 0; i < (batch_end - batch_start); i++) {
                    population[i] = Individual(m_graph.numOfVertices(), total_score);
                }
                goto again;
            }
        }


        return m_max_path;
    }
};

int calculateBeamWidth(int n, int m) {
    return 200;
}

int calculateDfsSize(int n, int m) {
    return std::min(n, 20);
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

    g.postInit();

    Solver solver(std::move(g), calculateBeamWidth(n, m), calculateDfsSize(n, m), timegate);

    const std::vector<int>& result = solver.run();

    printPath(result);

    return 0;
}