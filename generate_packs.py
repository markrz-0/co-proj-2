import time

total_feature_data = []
total_output_data = []

k = 5000
for i in range(1, k + 1):
    with open(f'tests/in/data{i}.in', 'r') as f:
        n, m = f.readline().strip().split(' ')
        n = int(n)

    with open(f"tests/out/data{i}.out", 'r') as f:
        _ = f.readline()
        output_data = ['0.0 0.0\n'] * n
        nodes = list(map(int, f.readline().strip().split(" ")))
        p = len(nodes) - 1
        mlt = p ** (-4/3)
        for idx, node in enumerate(nodes):
            output_data[node] = '1.0 ' + str(mlt * idx * (idx - p)) + '\n'

        total_output_data.extend(output_data)
    with open(f"tests/features/data{i}.out", 'r') as f:
        lines = f.readlines()
        total_feature_data.extend(lines)
    
time_id = int(time.time())
with open(f'data/features_pack_{time_id}.txt', 'w') as f:
    f.writelines(total_feature_data)
with open(f'data/output_pack_{time_id}.txt', 'w') as f:
    f.writelines(total_output_data)