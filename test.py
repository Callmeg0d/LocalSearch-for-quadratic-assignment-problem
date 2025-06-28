import os
import numpy as np
import time
from local_search import local_search, guided_local_search


def test(path, type='best-improvement'):
    with open(path) as f:
        n = int(f.readline())  # кол-во заводов

        dist = []
        for i in range(n):
            dist.append(list(map(int, f.readline().split())))
        f.readline()

        flow = []
        for i in range(n):
            flow.append(list(map(int, f.readline().split())))

    start_time = time.time()
    if (type == 'best-improvement'):
        solution, cost = local_search(dist, flow, np.random.permutation(n))
    elif (type == 'guided'):
        solution, cost = guided_local_search(dist, flow)
    end_time = time.time()

    print(f'File: {path}, type: {type}')
    print('Solution: ')
    print(solution.tolist())
    print('Cost:', cost)
    print(f'Elapsed time: {end_time - start_time:.2f}s')

    base_name = os.path.basename(path) + "33"

    if type == 'guided':
        output_dir = os.path.join(os.path.dirname(path), 'guided')
    else:
        output_dir = os.path.join(os.path.dirname(path), 'best_improve')

    os.makedirs(output_dir, exist_ok=True)

    sol_file_path = os.path.join(output_dir, base_name + '.sol')

    with open(sol_file_path, 'a') as sol_file:
        sol_file.write(' '.join(map(str, solution.tolist())) + '\n')


test('benchmarks&results/tai20a')
test('benchmarks&results/tai20a', 'guided')
print('--------------------')
test('benchmarks&results/tai40a')
test('benchmarks&results/tai40a', 'guided')
print('--------------------')
test('benchmarks&results/tai60a')
test('benchmarks&results/tai60a', 'guided')
print('--------------------')
test('benchmarks&results/tai80a')
test('benchmarks&results/tai80a', 'guided')
print('--------------------')
test('benchmarks&results/tai100a')
test('benchmarks&results/tai100a', 'guided')