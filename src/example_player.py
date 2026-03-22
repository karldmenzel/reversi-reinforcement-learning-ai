
import socket, pickle
from reversi import reversi

# The main function is only called when running in visual mode.
def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        # Receive play request from the server
        # turn : 1 --> you are playing as white | -1 --> you are playing as black
        # board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        next_move = choose_move(turn, board, game)

        # Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps(next_move))

# This function is called either by the server in command line mode,
# or by the main function in visual mode.
def choose_move(turn, board, game) -> list[int]:
    # Input data
    print(turn)
    print(board)
    print(game)

    # Use an algorithm to pick an X and Y
    x = -1
    y = -1

    # Return that move as a list of two integers
    return [x, y]

if __name__ == '__main__':
    main()