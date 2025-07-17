import math
import numpy as np
import neat
import random
import os
import pickle

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

def eval_genomes(genomes, config):
    # Reset fitness
    for genome_id, genome in genomes:
        genome.fitness = 0

    # Play round-robin matches
    for i, (genome_id1, genome1) in enumerate(genomes):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        for j in range(i + 1, len(genomes)):
            genome_id2, genome2 = genomes[j]
            net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

            # Randomly assign players for fairness
            if random.random() < 0.5:
                result = play_game_with_nets(net1, net2)  # net1 = X, net2 = O
                if result == "X":
                    genome1.fitness += 1
                    genome2.fitness -= 0.5
                elif result == "O":
                    genome2.fitness += 1
                    genome1.fitness -= 0.5
                else:
                    genome1.fitness += 0.25
                    genome2.fitness += 0.25
            else:
                result = play_game_with_nets(net2, net1)  # net2 = X, net1 = O
                if result == "X":
                    genome2.fitness += 1
                    genome1.fitness -= 0.5
                elif result == "O":
                    genome1.fitness += 1
                    genome2.fitness -= 0.5
                else:
                    genome1.fitness += 0.25
                    genome2.fitness += 0.25

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(eval_genomes, 200)

    # Save best genome
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
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path)
