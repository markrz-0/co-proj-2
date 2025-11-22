#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;


// number of features cannot be larger than 20 for small problem (ideally smaller than 10)
// N - number of nodes
// F - number of features [input layer]
// H - number of hidden nodes [middle/hidden layer] (preferably larger than F)
// (output layer's size is just 1 so i dont include it)
// Forward propagation of the network: O(NFH) + we also need to run DFS/some other pathfinding alg later and we only have 20s
struct NodeFeaturesNormalized {
    double degree;
    double min_neighbour_degree;
    double max_neighbour_degree;
    double arithmetic_mean_neighbour_degree;
    double harmonic_mean_neighbour_degree;
    double arithmetic_mean_global_degree;
    double harmonic_mean_global_degree;
    double clustering_coeff;
    double mean_neighbour_clustering_coeff;
    double mean_global_clustering_coeff;
};

double normalize(double x, double min_val, double max_val) {
    if (abs(max_val - min_val) < 1e-9) return 0.0f;
    return (x - min_val) / (max_val - min_val);
}

// Represents the graph
vector<double> getClusteringCoeffs(std::vector<std::vector<int>>& g) {
    int n = g.size();
    vector<int> degree(n, 0);

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
    vector<double> coeffs(n);
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

vector<NodeFeaturesNormalized> calculateFeatures(vector<vector<int>>& g) {
    int n = g.size();

    vector<double> degrees_log(n, 0);
    
    // 1. Pre-calculate degrees for all nodes
    for(int i = 0; i < n; ++i) {
        // we do log here to give higher impact to lower weights 
        // (distinction between 2 and 4 degree should be bigger than 1001 degree and 1030 degree)
        // we need to use log1p to avoid division by 0 later while calculating harmonic mean
        degrees_log[i] = log1p(g[i].size()); 
    }

    vector<double> coeffs = getClusteringCoeffs(g);
    
    double running_sum = 0.0f;
    for (double coeff : coeffs) {
        running_sum += coeff;
    }
    double global_mean_coeff = running_sum / n;

    // 2. Calculate Global Statistics
    double global_sum = 0;
    double global_reciprocal_sum = 0;

    double global_min_d = numeric_limits<float>::max();
    double global_max_d = 0.0f;

    for(double d : degrees_log) {
        global_min_d = min(global_min_d, d);
        global_max_d = max(global_max_d, d);

        global_sum += d;
        if (d > 0) {
            global_reciprocal_sum += 1.0 / d;
        }
    }

    double global_arithmetic_mean = (n > 0) ? (double)(global_sum / n) : 0.0f;
    
    // Note: Technically Harmonic mean of a set containing 0 is 0. 
    // Since we treated isolated nodes as 0, checking if global_reciprocal_sum > 0 is a proxy.
    double global_harmonic_mean = 0.0f;
    if (global_reciprocal_sum > 1e-9) {
        global_harmonic_mean = (double)(n / global_reciprocal_sum);
    }

    double norm_global_arith = normalize(global_arithmetic_mean, global_min_d, global_max_d);
    double norm_global_harm = normalize(global_harmonic_mean, global_min_d, global_max_d);

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
            double min_d = numeric_limits<float>::max();
            double max_d = 0.0f;
            double sum_d = 0;
            double sum_reciprocal_d = 0;
            double running_sum = 0.0;

            for (int neighbor : g[i]) {
                double d_neighbor = degrees_log[neighbor];
                
                if (d_neighbor < min_d) min_d = d_neighbor;
                if (d_neighbor > max_d) max_d = d_neighbor;
                
                sum_d += d_neighbor;
                sum_reciprocal_d += 1.0 / d_neighbor;
                running_sum += coeffs[neighbor];
            }

            f.min_neighbour_degree = normalize(min_d, global_min_d, global_max_d);
            f.max_neighbour_degree = normalize(max_d, global_min_d, global_max_d);
            f.arithmetic_mean_neighbour_degree = normalize((double)(sum_d / g[i].size()), global_min_d, global_max_d);
            f.harmonic_mean_neighbour_degree = normalize((double)(g[i].size() / sum_reciprocal_d), global_min_d, global_max_d);
            f.clustering_coeff = coeffs[i];
            f.mean_neighbour_clustering_coeff = running_sum / g[i].size();
            f.mean_global_clustering_coeff = global_mean_coeff;
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
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.min_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.max_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.arithmetic_mean_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.harmonic_mean_neighbour_degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.arithmetic_mean_global_degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.harmonic_mean_global_degree << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.clustering_coeff << " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.mean_neighbour_clustering_coeff<< " ";
        cout << fixed << setprecision(numeric_limits<double>::max_digits10) << features.mean_global_clustering_coeff << " ";
        cout << "\n";
    }

    return 0;
}