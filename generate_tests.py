import random
import sys
import os
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
    
    assert min_n > 0
    assert max_n > 0
    assert min_n <= max_n

    if not os.path.exists('tests'):
        os.mkdir('tests')

    if not os.path.exists('tests/in'):
        os.mkdir('tests/in')

    if not os.path.exists('tests/out'):
        os.mkdir('tests/out')

    for i in range(1, k + 1):
        n = random.randint(min_n, max_n)
        m = int(random.random()  * n * n)
        print(f"{i}/{k} (N = {n}; M = {m})")
        g = generate_simple(n, m)
        save_graph(g, n, m, f'tests/in/data{i}.in')
        if os.name == "posix":
            os.system(f"./{solver_name} < tests/in/data{i}.in > tests/out/data{i}.out")
        elif os.name == "nt":
            os.system(f"{solver_name} < tests/in/data{i}.in > tests/out/data{i}.out")
except:
    print(f"usage: py {sys.argv[0]} <test_num> <min_n> <max_n> <solver_name>")

