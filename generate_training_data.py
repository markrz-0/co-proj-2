import random
import sys
import os
import time
import networkx as nx


def save_graph(G: nx.Graph, n, m, filename: str):
    with open(filename, 'w') as f:
        f.write(f'{n} {len(G.edges)}\n')
        for e in G.edges:
            f.write(f'{e[0]} {e[1]}\n')

def generate_simple(n, m) -> nx.Graph:
    G: nx.Graph = nx.random_labeled_tree(n)
    left = m - (n - 1)
    for _ in range(left):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G

try:
    k = int(sys.argv[1])
    min_n = int(sys.argv[2])
    max_n = int(sys.argv[3])
    solver_name = sys.argv[4]
    features_generator_name = sys.argv[5]
    
    assert min_n > 0
    assert max_n > 0
    assert min_n <= max_n

    if not os.path.exists('tests'):
        os.mkdir('tests')

    if not os.path.exists('tests/in'):
        os.mkdir('tests/in')

    if not os.path.exists('tests/out'):
        os.mkdir('tests/out')
        
    if not os.path.exists('tests/features'):
        os.mkdir('tests/features')

    if not os.path.exists('data'):
        os.mkdir('data')

    total_feature_data = []
    total_output_data = []

    for i in range(1, k + 1):
        n = random.randint(min_n, max_n)
        m = int(random.random()  * n * n)
        
        print(f"{i}/{k} (N = {n}; M = {m})")
        
        g = generate_simple(n, m)
        save_graph(g, n, m, f'tests/in/data{i}.in')

        os.system(f"{solver_name} < tests/in/data{i}.in > tests/out/data{i}.out")
        os.system(f"{features_generator_name} < tests/in/data{i}.in > tests/features/data{i}.out")

        output_data = ['0.0\n'] * n

        with open(f"tests/out/data{i}.out", 'r') as f:
            _ = f.readline()
            nodes = list(map(int, f.readline().strip().split(" ")))
            for node in nodes:
                output_data[node] = '1.0\n'
            total_output_data.extend(output_data)
        with open(f"tests/features/data{i}.out", 'r') as f:
            lines = f.readlines()
            total_feature_data.extend(lines)
    
    time_id = int(time.time())
    with open(f'data/features_pack_{time_id}.txt', 'w') as f:
        f.writelines(total_feature_data)
    with open(f'data/output_pack_{time_id}.txt', 'w') as f:
        f.writelines(total_output_data)



except:
    print(f"usage: py {sys.argv[0]} <test_num> <min_n> <max_n> <solver_name> <features_generator_name>")
    print("On linux you might want to prefix solver and feature generator with ./")
    print(f"Both programs should be resolvable from directory of {sys.argv[0]}")
    print("LINUX:")
    print(f"py {sys.argv[0]} 1000 10 20 ./solver ./features")
    print("WINDOWS:")
    print(f"py {sys.argv[0]} 1000 10 20 solver.exe features.exe")

