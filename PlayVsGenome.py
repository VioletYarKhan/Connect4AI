import pickle
import neat
import os
import random

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

def board_to_input(board, player_char):
    inputs = []
    for row in board:
        for cell in row:
            if cell == player_char:
                inputs.append(1.0)
            elif cell == " ":
                inputs.append(0.0)
            else:
                inputs.append(-1.0)
    return inputs

def get_valid_columns(board):
    return [c for c in range(COLS) if board[0][c] == " "]

def get_move_from_net(net, board, player_char):
    inputs = board_to_input(board, player_char)
    outputs = net.activate(inputs)
    valid_columns = get_valid_columns(board)
    sorted_indices = sorted(range(len(outputs)), key=lambda i: outputs[i], reverse=True)
    for i in sorted_indices:
        if i in valid_columns:
            return i
    return random.choice(valid_columns)

def print_board(board):
    print("\n0  1   2   3   4   5   6")
    for row in board:
        print(" | ".join(row))
    print("-" * 29)

def play_against_genome(config_path, genome_path):
    # Load config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load genome
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Choose side
    while True:
        player_char = input("Do you want to play first (X) or second (O)? ").upper()
        if player_char in ["X", "O"]:
            break
        print("Invalid input. Please choose X or O.")

    ai_char = "O" if player_char == "X" else "X"
    board = create_board()
    turn = 0

    while True:
        print_board(board)
        current_char = ["X", "O"][turn % 2]

        if current_char == player_char:
            while True:
                try:
                    col = int(input(f"Your move (0-6): "))
                    if 0 <= col <= 6 and board[0][col] == " ":
                        break
                    else:
                        print("Invalid column or full. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number 0-6.")
            add_piece(board, col, player_char)
        else:
            col = get_move_from_net(net, board, ai_char)
            print(f"AI ({ai_char}) chooses column {col}")
            add_piece(board, col, ai_char)

        winner = check_win(board)
        if winner:
            print_board(board)
            if winner == player_char:
                print("🎉 You win!")
            else:
                print("💀 The AI wins!")
            break

        if all(board[0][c] != " " for c in range(COLS)):
            print_board(board)
            print("🤝 It's a draw!")
            break

        turn += 1


if __name__ == "__main__":
    config_file = "neat-config.txt"
    genome_file = "best_genome.pkl"
    play_against_genome(config_file, genome_file)

