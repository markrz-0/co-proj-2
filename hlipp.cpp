#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>

/*
MAXPATHS: int the higher the more accurate and slower

UPDATE_NOW_AFTER_TICKS: int the higher the slower the alg gets. 
        It value states after how many calls of 
        timegate.updateNow() will now actually be updated 
        [now() call has relatively big overhead] 

TIME_LIMIT_SECONDS: double
*/

#define ENABLE_MAX_PATHS false
#define MAXPATHS 5000

#define ENABLE_TIME_LIMIT false
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

// Represents an unweighted undirected graph
class Graph {
public:
    std::vector<std::vector<int>> adj;

    Graph(int V) {
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
        for (int i = 0; i < adj.size(); i++) {
            sort(adj[i].begin(), adj[i].end(), [&](int a, int b) {
                return adj[a].size() > adj[b].size();
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
            if ((m_num_paths - m_last_improv) > MAXPATHS && ENABLE_MAX_PATHS) {
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