import random
import sys
import os

class Graph:
    def __init__(self, n, edges):
        self.n = n
        self.m = len(edges)
        self.edges = edges

    def save(self, filename: str):
        with open(filename, 'w') as f:
            f.write(f'{self.n} {self.m}\n')
            for e in self.edges:
                f.write(f'{e[0]} {e[1]}\n')

def generate(n, m):
    edges = set()
    for e in range(m):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        edges.add((i, j))

    return Graph(n, edges)

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
        print(f"Generating {i}/{k} (N = {n}; M = {m})")
        g = generate(n, m)
        g.save(f'tests/in/data{i}.in')
        os.system(f"./{solver_name} < tests/in/data{i}.in > tests/out/data{i}.out")
except:
    print(f"usage: py {sys.argv[0]} <test_num> <min_n> <max_n> <solver_name>")

