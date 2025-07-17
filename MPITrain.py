import math
import numpy as np
import neat
import random
import os
import pickle
from mpi4py import MPI

ROWS = 6
COLS = 7
TOURNAMENT_MATCHES = 5  # matches per genome

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
    for col in range(COLS):
        for row in range(ROWS - 3):
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
    return 0

def board_to_input(board, player):
    inputs = []
    for row in board:
        for cell in row:
            if cell == player:
                inputs.append(1.0)
            elif cell == " ":
                inputs.append(0.0)
            else:
                inputs.append(-1.0)
    return inputs

def get_valid_columns(board):
    return [c for c in range(COLS) if board[0][c] == " "]

def get_move_from_net(net, board, player):
    inputs = board_to_input(board, player)
    outputs = net.activate(inputs)
    valid_columns = get_valid_columns(board)
    sorted_indices = np.argsort(outputs)[::-1]
    for i in sorted_indices:
        if i in valid_columns:
            return i
    return random.choice(valid_columns)

def play_game_with_nets(net1, net2):
    board = create_board()
    players = ["X", "O"]
    nets = {"X": net1, "O": net2}
    turn = 0
    while True:
        current_player = players[turn % 2]
        move = get_move_from_net(nets[current_player], board, current_player)
        add_piece(board, move, current_player)
        winner = check_win(board)
        if winner:
            return winner
        if all(board[0][c] != " " for c in range(COLS)):
            return "draw"
        turn += 1

def evaluate_genome_pair(genome1, genome2, config):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

    if random.random() < 0.5:
        result = play_game_with_nets(net1, net2)
        if result == "X":
            return 1, -1
        elif result == "O":
            return -1, 1
        else:
            return 0.25, 0.25
    else:
        result = play_game_with_nets(net2, net1)
        if result == "X":
            return -1, 1
        elif result == "O":
            return 1, -1
        else:
            return 0.25, 0.25

def eval_genomes(genomes, config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute genomes among processes
    local_genomes = [g for i, g in enumerate(genomes) if i % size == rank]
    genome_map = {g[0]: g[1] for g in genomes}

    local_fitness = {g[0]: 0.0 for g in local_genomes}

    for genome_id, genome in local_genomes:
        opponents = random.sample(genomes, k=TOURNAMENT_MATCHES)
        for opp_id, opponent in opponents:
            if genome_id == opp_id:
                continue
            score1, score2 = evaluate_genome_pair(genome, opponent, config)
            local_fitness[genome_id] += score1

    # Gather all fitnesses to rank 0
    all_fitness = comm.gather(local_fitness, root=0)

    if rank == 0:
        total_fitness = {}
        for fdict in all_fitness:
            for gid, val in fdict.items():
                total_fitness[gid] = total_fitness.get(gid, 0) + val
        for genome_id, genome in genomes:
            genome.fitness = total_fitness.get(genome_id, 0)

def run_neat(config_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    if rank == 0:
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

    winner = p.run(eval_genomes, 100)

    if rank == 0:
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("✅ Best genome saved to best_genome.pkl")

        best_net = neat.nn.FeedForwardNetwork.create(winner, config)
        visualize_game(best_net, best_net)

def visualize_game(net1, net2, delay=0.5):
    import time
    board = create_board()
    players = ["X", "O"]
    nets = {"X": net1, "O": net2}
    turn = 0

    while True:
        current_player = players[turn % 2]
        move = get_move_from_net(nets[current_player], board, current_player)
        add_piece(board, move, current_player)

        print(f"\nTurn {turn+1}: Player {current_player} moves in column {move}")
        for row in board:
            print(" | ".join(row))
        print("-" * 29)
        time.sleep(delay)

        winner = check_win(board)
        if winner:
            print(f"\n🎉 Player {winner} wins!")
            break
        if all(board[0][c] != " " for c in range(COLS)):
            print("\n🤝 It's a draw!")
            break
        turn += 1

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path)
