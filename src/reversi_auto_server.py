
from reversi import reversi

# Algorithm 1 -- update the 'from' to choose a different player
from minimax_alpha_beta_h_nic_nn import choose_move as algorithm_1

# Algorithm 2 -- update the 'from' to choose a different player
from minimax_alpha_beta_updated_weights import choose_move as algorithm_2

class AutoGameServer:
    def __init__(self, player1, player2):
        """
        player1 = white (turn = 1)
        player2 = black (turn = -1)
        """
        self.game = reversi()
        self.player1 = player1
        self.player2 = player2
        self.turn = 1  # White starts

    def play_game(self):
        consecutive_passes = 0

        while True:
            current_player = self.player1 if self.turn == 1 else self.player2

            # Ask AI for move
            move = current_player(self.turn, self.game.board.copy(), self.game)

            x, y = move

            # Player passes
            if x == -1 and y == -1:
                consecutive_passes += 1
                print(f"Player {'White' if self.turn == 1 else 'Black'} has no valid moves, passes.")
            else:
                result = self.game.step(x, y, self.turn)

                if result >= 0:
                    consecutive_passes = 0
                    # print(f"Player {'White' if self.turn == 1 else 'Black'} plays ({x},{y})")
                else:
                    # Illegal move → treat as pass
                    consecutive_passes += 1
                    print(f"Illegal move by {'White' if self.turn == 1 else 'Black'} → treated as pass.")

            # End condition
            if consecutive_passes >= 2:
                break

            # Switch turn
            self.turn *= -1

        # print("\nFinal Board:")
        # print(self.game.board) #Note the board printed out is mirrored from the actual board
        white = self.game.white_count
        black = self.game.black_count
        if white > black:
            print(f"White wins {white} to {black}!\n")
            return 1
        elif black > white:
            print(f"Black wins {black} to {white}!\n")
            return -1
        else:
            print(f"Game is a draw, {black} to {white}!\n")
            return 0

if __name__ == "__main__":
    algorithm_1_wins = 0
    algorithm_2_wins = 0
    draws = 0

    algorithm_1_name = algorithm_1.__module__
    algorithm_2_name = algorithm_2.__module__

    print(f"Beginning game one, algorithm 1 ({algorithm_1_name}) is white, algorithm 2 ({algorithm_2_name}) is black.")
    game1 = AutoGameServer(
        player1=algorithm_1,  # White
        player2=algorithm_2   # Black
    )

    game1_winner = game1.play_game()

    if(game1_winner == 1):
        algorithm_1_wins += 1
    elif (game1_winner == -1):
        algorithm_2_wins += 1
    else:
        draws += 1

    print(f"Beginning game two, algorithm 1 ({algorithm_1_name}) is black, algorithm 2 ({algorithm_2_name}) is white.")
    game2 = AutoGameServer(
        player1=algorithm_2,  # White
        player2=algorithm_1   # Black
    )

    game2_winner = game2.play_game()

    if game2_winner == -1:
        algorithm_1_wins += 1
    elif game2_winner == 1:
        algorithm_2_wins += 1
    else:
        draws += 1

    final_result = game1_winner + game2_winner

    if algorithm_1_wins == algorithm_2_wins:
        print(f"Final result: both algorithms tied ({algorithm_1_name} and {algorithm_2_name}).")
    elif algorithm_1_wins > algorithm_2_wins:
        print(f"Final result: algorithm 1 ({algorithm_1_name}) wins.")
    else:
        print(f"Final result: algorithm 2 ({algorithm_2_name}) wins.")
