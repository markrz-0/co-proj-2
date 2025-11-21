#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <cmath>

#include "model_weights.h"

#define TIME_LIMIT_SECONDS 18
#define UPDATE_NOW_AFTER_TICKS 0

// --- Configuration ---
const int POPULATION_SIZE = 50;
const int MUTATION_RATE = 400; // promiles
const int SHRINK_PROBABILITY = 200; // promiles 
const int REVERSAL_CHANCE = 500; // promiles 

int random(int min_num_inclusive, int max_number_exclusive) {
    int delta = max_number_exclusive - min_num_inclusive;
    return rand() % delta + min_num_inclusive;
}

// --- Time Gate for TLE Prevention ---
struct TimeGate {
    time_t startTime;
    bool timeExpired = false;
    int nextUpdate = UPDATE_NOW_AFTER_TICKS;

    void init() {
        startTime = time(0);
        timeExpired = false;
        nextUpdate = UPDATE_NOW_AFTER_TICKS;
    }

    void updateNow() {
        if (nextUpdate <= 0) {
            auto now = time(0);
            if (now - startTime > TIME_LIMIT_SECONDS) {
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

// --- Graph Structure ---
class Graph {
public:
    int V;
    std::vector<std::vector<int>> adj;

    Graph(int V) : V(V) {
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    bool hasEdge(int u, int v) const {
        return std::binary_search(adj[u].begin(), adj[u].end(), v);
    }
};

// --- Individual Structure ---
struct Individual {
    std::vector<int> path;
    
    // Optimization Structures
    // neighborCounts[u] = how many nodes in the current 'path' are neighbors of 'u'
    std::vector<int> neighborCounts; 
    std::vector<bool> inPath;
    
    int fitness;

    Individual(int n) : fitness(0), neighborCounts(n, 0), inPath(n, false) {}

    void calculateFitness() {
        fitness = path.size();
    }

    // Safely add a node and update state
    void pushNode(const Graph& g, int u) {
        path.push_back(u);
        inPath[u] = true;
        // For every neighbor of u, increment their connection count to the path
        for (int v : g.adj[u]) {
            neighborCounts[v]++;
        }
    }

    // Safely remove the last node and update state
    void popNode(const Graph& g) {
        if (path.empty()) return;
        int u = path.back();
        path.pop_back();
        inPath[u] = false;
        // For every neighbor of u, decrement their connection count
        for (int v : g.adj[u]) {
            neighborCounts[v]--;
        }
    }

    // Check validity in O(1)
    // Candidate must be a neighbor of path.back() (checked by caller via adjacency list)
    // Constraint: Candidate must NOT connect to any OTHER node in the path.
    // Therefore, neighborCounts[candidate] must be exactly 1 (the connection to path.back())
    bool isValidExtension(int candidate) const {
        if (inPath[candidate]) return false;
        // If neighborCounts is 1, it means it connects ONLY to the current tail (valid).
        // If > 1, it connects to tail AND someone earlier (chord -> invalid).
        // If 0, it doesn't connect to tail (shouldn't happen if iterating adj[tail]).
        return neighborCounts[candidate] == 1;
    }
};

// --- Genetic Algorithm Class ---
class GeneticAlgorithm {
private:
    Graph& graph;
    std::vector<float> nodeScores;
    std::vector<Individual> population;
    Individual globalBest;

public:
    GeneticAlgorithm(Graph& g) : graph(g), globalBest(0), nodeScores(g.V) {}

    void initNodeScores() {
        auto features = calculateFeatures(graph.adj);
        for (int node = 0; node < graph.V; node++) {
            nodeScores[node] = forwardPropagation(features[node].to_vector());
        }
    }

    const Individual& getGlobalBest() const {
        return globalBest;
    }

    // Initialize with random valid induced paths
    void initializePopulation() { 
        population.clear();
        std::vector<std::pair<float, int>> x(graph.V);
        for (int i = 0; i < graph.V; i++) {
            x[i] = std::make_pair(-nodeScores[i], i);
        }
        std::sort(x.begin(), x.end());
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            population.push_back(generateRandomPath(
                x[random(0, std::min(10, graph.V - 1))].second
            ));
        }
    }

    Individual generateRandomPath(int startNode) { 
        Individual ind(graph.V);
        ind.pushNode(graph, startNode);

        // Greedily grow
        grow(ind);
        
        ind.calculateFitness();
        return ind;
    }

    // Helper to grow a path as much as possible
    void grow(Individual& ind) {
        bool stuck = false;
        int attempts = 0;
        // Safety cap to prevent infinite loops in weird cases, though strictly not needed for DAG logic
        while (!stuck && attempts < 100) { 
            int last = ind.path.back();
            std::vector<int> candidates;

            float totalWeight = 0.0f;
            // Only check neighbors of the tail
            for (int neighbor : graph.adj[last]) {
                // Using the O(1) check
                if (ind.isValidExtension(neighbor)) {
                    candidates.push_back(neighbor);
                    totalWeight += nodeScores[neighbor];
                }
            }

            if (candidates.empty()) {
                stuck = true;
            } else {
                // Pick random valid neighbor
                int chosen = -1;
                if (totalWeight <= 0.05) {
                    // fallback for small weight
                    chosen = candidates[random(0, candidates.size())];
                } else {
                    float r = (float)(((double)rand() / RAND_MAX) * totalWeight);

                    float cumulative = 0.0f;
                    for (int c : candidates) {
                        cumulative += nodeScores[c];
                        if (cumulative >= r) {
                            chosen = c;
                            break;
                        }
                    }

                    if (chosen == -1) { // fall back for floating point rounding errors
                         chosen = candidates[random(0, candidates.size())];
                    }
                }
                ind.pushNode(graph, chosen);
            }
            attempts++;
        }
    }

    // Tournament Selection
    Individual tournamentSelection(int k = 5) {
        Individual best = population[random(0, POPULATION_SIZE)];
        
        for (int i = 1; i < k; ++i) {
            Individual contender = population[random(0, POPULATION_SIZE)];
            if (contender.fitness > best.fitness) {
                best = contender;
            }
        }
        return best;
    }

    void mutate(Individual& ind) {
        
        // 1. Reverse (Head/Tail Flip)
        if (random(0, 1000) < REVERSAL_CHANCE) {
            std::reverse(ind.path.begin(), ind.path.end());
            // Note: neighborCounts and inPath are SET based, so they remain valid after reverse!
            // No rebuild needed, which is great.
        }

        // 2. Shrink (Backtracking)
        if (!ind.path.empty() && random(0, 1000) < SHRINK_PROBABILITY) {
            int remove_count = 1;
            if (ind.path.size() > 2) remove_count = (rand() % 2) + 1; 
            
            for(int k=0; k<remove_count && !ind.path.empty(); ++k) {
                ind.popNode(graph); // Uses helper to keep state sync
            }
        }

        // 3. Grow
        if (ind.path.empty()) {
            // Restart if empty
            ind.pushNode(graph, random(0, graph.V));
        }
        
        grow(ind);
        ind.calculateFitness();
    }

    void run(TimeGate& timegate) {
        initializePopulation();
        
        // Init global best
        if (!population.empty()) globalBest = population[0];

        for( ; !timegate.expired(); timegate.updateNow()) {
            std::vector<Individual> newPopulation;

            // Elitism
            std::sort(population.begin(), population.end(), 
                [](const Individual& a, const Individual& b) {
                    return a.fitness > b.fitness;
                });
            
            if (population[0].fitness > globalBest.fitness) {
                globalBest = population[0];
            }

            newPopulation.push_back(population[0]);

            // Generate rest
            while (newPopulation.size() < POPULATION_SIZE) {
                Individual parent = tournamentSelection();
                Individual offspring = parent; // Clone
                
                if (random(0, 1000) < MUTATION_RATE) {
                    mutate(offspring);
                }
                newPopulation.push_back(offspring);

                if (timegate.expired()) { break; }
            }
            population = newPopulation;
        }
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);

    TimeGate timegate;
    timegate.init();

    int n, m;
    if (!(std::cin >> n >> m)) return 0;

    srand(m);

    Graph g(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        std::cin >> u >> v;
        g.addEdge(u, v);
    }

    GeneticAlgorithm ga(g);
    ga.initNodeScores();
    ga.run(timegate);

    const Individual& best = ga.getGlobalBest();

    std::cout << best.path.size() << "\n";
    for (size_t i = 0; i < best.path.size(); ++i) {
        std::cout << best.path[i] << (i == best.path.size() - 1 ? "" : " ");
    }
    std::cout << "\n";

    return 0;
}