import math
import numpy as np
import neat
import random
import os
import pickle
from mpi4py import MPI

ROWS = 6
COLS = 7

# --- Same game logic functions (create_board, add_piece, check_win, etc.) ---
# (Keep your existing definitions for these)

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

def simulate_matchup(args):
    genome_id1, genome1, genome_id2, genome2, config = args
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

    g1_score, g2_score = 0, 0

    if random.random() < 0.5:
        result = play_game_with_nets(net1, net2)  # net1 = X
        if result == "X":
            g1_score, g2_score = 1, -1
        elif result == "O":
            g1_score, g2_score = -1, 1
        else:
            g1_score = g2_score = 0.25
    else:
        result = play_game_with_nets(net2, net1)  # net2 = X
        if result == "X":
            g1_score, g2_score = -1, 1
        elif result == "O":
            g1_score, g2_score = 1, -1
        else:
            g1_score = g2_score = 0.25

    return (genome_id1, g1_score), (genome_id2, g2_score)

def eval_genomes(genomes, config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Reset fitness on all ranks
    for _, genome in genomes:
        genome.fitness = 0

    if rank == 0:
        # Prepare all matchups (only on rank 0)
        matchups = []
        for i, (id1, g1) in enumerate(genomes):
            for j in range(i + 1, len(genomes)):
                id2, g2 = genomes[j]
                matchups.append((id1, g1, id2, g2, config))
    else:
        matchups = None

    # Scatter the matchups roughly evenly across ranks
    matchups = comm.bcast(matchups, root=0)
    chunk_size = math.ceil(len(matchups) / size)
    local_matchups = matchups[rank*chunk_size:(rank+1)*chunk_size]

    # Each rank runs its chunk of matchups
    local_results = [simulate_matchup(m) for m in local_matchups]

    # Gather all results back on rank 0
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten the list of lists
        all_results = [item for sublist in gathered_results for item in sublist]

        fitness_map = {}
        for (id1, score1), (id2, score2) in all_results:
            fitness_map[id1] = fitness_map.get(id1, 0) + score1
            fitness_map[id2] = fitness_map.get(id2, 0) + score2

        for genome_id, genome in genomes:
            genome.fitness = fitness_map.get(genome_id, 0)

    # Broadcast updated genomes with fitness back to all ranks (optional if needed)
    genomes = comm.bcast(genomes, root=0)

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(lambda genomes, config: eval_genomes(list(genomes), config), 5)

    # Save best genome on rank 0 only
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        print("✅ Best genome saved to best_genome.pkl")

        # Visualize top agent vs itself
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

        # Print board
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
    import sys
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")

    # Run only after mpi4py is imported and MPI is initialized
    run_neat(config_path)
