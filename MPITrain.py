import numpy as np
import neat
import random
import pickle
from mpi4py import MPI
import os

ROWS = 6
COLS = 7

# Create empty board
def create_board():
    return [[" " for _ in range(COLS)] for _ in range(ROWS)]

# Drop a piece into a column
def add_piece(board, col, player):
    for row in reversed(board):
        if row[col] == " ":
            row[col] = player
            return True
    return False

# Check for a win
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

# Check if board is full
def is_full(board):
    return all(cell != " " for row in board for cell in row)

# Get input features for a board and player
def get_inputs(board, player):
    enemy = "O" if player == "X" else "X"
    flat = []
    for row in board:
        for cell in row:
            if cell == " ":
                flat.append(0.0)
            elif cell == player:
                flat.append(1.0)
            else:
                flat.append(-1.0)
    return flat + [1.0 if player == "X" else -1.0]

# Get the move from a genome's output
def get_move(net, board, player):
    inputs = get_inputs(board, player)
    outputs = net.activate(inputs)
    sorted_moves = sorted(list(enumerate(outputs)), key=lambda x: -x[1])
    for col, _ in sorted_moves:
        if board[0][col] == " ":
            return col
    return None  # No valid move

# Play one game between two genomes
def play_game(genome1, genome2, config):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
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
            return 0.5  # Draw
        turn += 1

# Evaluate genomes in parallel using tournament play
def eval_genomes(genomes, config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        n = len(genomes)
        scores = [0.0] * n
        matchups = [(i, j) for i in range(n) for j in range(i+1, n)]
        chunk_size = len(matchups) // (size - 1) + 1
        chunks = [matchups[i:i+chunk_size] for i in range(0, len(matchups), chunk_size)]

        # Send matchups to workers
        for i in range(1, size):
            comm.send(chunks[i-1] if i-1 < len(chunks) else [], dest=i)

        # Receive results
        for i in range(1, size):
            results = comm.recv(source=i)
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

    else:
        received = comm.recv(source=0)
        results = []
        for i1, i2 in received:
            g1 = genomes[i1][1]
            g2 = genomes[i2][1]
            outcome = play_game(g1, g2, config)
            results.append((i1, i2, outcome))
        comm.send(results, dest=0)

# Run NEAT with MPI tournament evaluation
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
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(eval_genomes, 50)

        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("✅ Best genome saved to best_genome.pkl")
    else:
        # Workers do nothing outside eval_genomes
        pass

    comm.Barrier()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path)
