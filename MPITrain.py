import numpy as np
import neat
import random
import pickle
from mpi4py import MPI
import os

ROWS = 6
COLS = 7

def create_board():
    return [[" " for _ in range(COLS)] for _ in range(ROWS)]

def add_piece(board, col, player):
    for row in reversed(board):
        if row[col] == " ":
            row[col] = player
            return True
    return False

def check_win(board):
    for row in range(ROWS):
        for col in range(COLS - 3):
            if board[row][col] != " " and all(board[row][col + i] == board[row][col] for i in range(4)):
                return board[row][col]
    for row in range(ROWS - 3):
        for col in range(COLS):
            if board[row][col] != " " and all(board[row + i][col] == board[row][col] for i in range(4)):
                return board[row][col]
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if board[row][col] != " " and all(board[row + i][col + i] == board[row][col] for i in range(4)):
                return board[row][col]
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if board[row][col] != " " and all(board[row - i][col + i] == board[row][col] for i in range(4)):
                return board[row][col]
    return None

def is_full(board):
    return all(cell != " " for row in board for cell in row)

def get_inputs(board, player):
    flat = []
    for row in board:
        for cell in row:
            if cell == " ":
                flat.append(0.0)
            elif cell == player:
                flat.append(1.0)
            else:
                flat.append(-1.0)
    flat.append(1.0 if player == "X" else -1.0)
    return flat

def get_move(net, board, player):
    inputs = get_inputs(board, player)
    outputs = net.activate(inputs)
    sorted_moves = sorted(list(enumerate(outputs)), key=lambda x: -x[1])
    for col, _ in sorted_moves:
        if board[0][col] == " ":
            return col
    return None

def play_game(net1, net2):
    board = create_board()
    turn = 0
    players = [("X", net1), ("O", net2)]
    while True:
        p, net = players[turn % 2]
        move = get_move(net, board, p)
        if move is None or not add_piece(board, move, p):
            return 1 - (turn % 2)  # Opponent wins
        winner = check_win(board)
        if winner:
            return 0 if winner == "X" else 1
        if is_full(board):
            return 0.5
        turn += 1

# Master process: assign work and collect scores
def master_eval_genomes(genomes, config):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
    scores = [0.0 for _ in genomes]
    matchups = [(i, j) for i in range(len(genomes)) for j in range(i+1, len(genomes))]
    chunks = [matchups[i::size - 1] for i in range(size - 1)]

    # Send data to workers
    for worker in range(1, size):
        data = {
            "chunk": chunks[worker - 1],
            "nets": [(i, pickle.dumps(nets[i])) for i in set(i for pair in chunks[worker - 1] for i in pair)],
        }
        comm.send(data, dest=worker)

    # Collect results
    for _ in range(1, size):
        results = comm.recv(source=MPI.ANY_SOURCE)
        for i1, i2, outcome in results:
            if outcome == 0:
                scores[i1] += 1
            elif outcome == 1:
                scores[i2] += 1
            else:
                scores[i1] += 0.5
                scores[i2] += 0.5

    # Assign fitness
    for i, (_, g) in enumerate(genomes):
        g.fitness = scores[i]

# Worker loop: receive matchups and return results
def worker_loop(config):
    comm = MPI.COMM_WORLD
    data = comm.recv(source=0)
    nets = {i: pickle.loads(b) for i, b in data["nets"]}
    results = []
    for i1, i2 in data["chunk"]:
        outcome = play_game(nets[i1], nets[i2])
        results.append((i1, i2, outcome))
    comm.send(results, dest=0)

def run_neat(config_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if rank == 0:
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.StatisticsReporter())
        winner = p.run(lambda genomes, config: master_eval_genomes(genomes, config), 50)
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("✅ Best genome saved to best_genome.pkl")
    else:
        worker_loop(config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path)
