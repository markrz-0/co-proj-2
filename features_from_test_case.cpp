#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// so we can set a precision. Maybe calculating on floats will be faster. we will see
typedef double nnfloat;

// number of features cannot be larger than 20 for small problem (ideally smaller than 10)
// N - number of nodes
// F - number of features [input layer]
// H - number of hidden nodes [middle/hidden layer] (preferably larger than F)
// (output layer's size is just 1 so i dont include it)
// Forward propagation of the network: O(NFH) + we also need to run DFS/some other pathfinding alg later and we only have 20s
struct NodeFeaturesNormalized {
    nnfloat degree;
    nnfloat min_neighbour_degree;
    nnfloat max_neighbour_degree;
    nnfloat arithmetic_mean_neighbour_degree;
    nnfloat harmonic_mean_neighbour_degree;
    nnfloat arithmetic_mean_global_degree;
    nnfloat harmonic_mean_global_degree;
};

nnfloat normalize(nnfloat x, nnfloat min_val, nnfloat max_val) {
    if (abs(max_val - min_val) < 1e-9) return 0.0f;
    return (x - min_val) / (max_val - min_val);
}


vector<NodeFeaturesNormalized> calculateFeatures(const vector<vector<int>>& g) {
    int n = g.size();

    vector<nnfloat> degrees_log(n, 0);
    
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

    nnfloat global_min_d = numeric_limits<float>::max();
    nnfloat global_max_d = 0.0f;

    for(nnfloat d : degrees_log) {
        global_min_d = min(global_min_d, d);
        global_max_d = max(global_max_d, d);

        global_sum += d;
        if (d > 0) {
            global_reciprocal_sum += 1.0 / d;
        }
    }

    nnfloat global_arithmetic_mean = (n > 0) ? (nnfloat)(global_sum / n) : 0.0f;
    
    // Note: Technically Harmonic mean of a set containing 0 is 0. 
    // Since we treated isolated nodes as 0, checking if global_reciprocal_sum > 0 is a proxy.
    nnfloat global_harmonic_mean = 0.0f;
    if (global_reciprocal_sum > 1e-9) {
        global_harmonic_mean = (nnfloat)(n / global_reciprocal_sum);
    }

    nnfloat norm_global_arith = normalize(global_arithmetic_mean, global_min_d, global_max_d);
    nnfloat norm_global_harm = normalize(global_harmonic_mean, global_min_d, global_max_d);

    // 3. Calculate Node Features
    vector<NodeFeaturesNormalized> features(n);

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
            nnfloat min_d = numeric_limits<float>::max();
            nnfloat max_d = 0.0f;
            double sum_d = 0;
            double sum_reciprocal_d = 0;

            for (int neighbor : g[i]) {
                nnfloat d_neighbor = degrees_log[neighbor];
                
                if (d_neighbor < min_d) min_d = d_neighbor;
                if (d_neighbor > max_d) max_d = d_neighbor;
                
                sum_d += d_neighbor;
                sum_reciprocal_d += 1.0 / d_neighbor;
            }

            f.min_neighbour_degree = normalize(min_d, global_min_d, global_max_d);
            f.max_neighbour_degree = normalize(max_d, global_min_d, global_max_d);
            f.arithmetic_mean_neighbour_degree = normalize((nnfloat)(sum_d / g[i].size()), global_min_d, global_max_d);
            f.harmonic_mean_neighbour_degree = normalize((nnfloat)(g[i].size() / sum_reciprocal_d), global_min_d, global_max_d);
        }
    }

    return features;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> g(n);

    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<NodeFeaturesNormalized> node_features = calculateFeatures(g);

    for (auto& features : node_features) {
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.degree << " ";
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.min_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.max_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.arithmetic_mean_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.harmonic_mean_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.arithmetic_mean_global_degree << " ";
        cout << fixed << setprecision(numeric_limits<nnfloat>::max_digits10) << features.harmonic_mean_global_degree << "\n";
    }

    return 0;
}