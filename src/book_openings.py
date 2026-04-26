
W = 1
B = -1

book_openings_still_exist = True

def get_known_position(board, turn):
    global book_openings_still_exist
    if not book_openings_still_exist:
        return None

    board_as_tuple = tuple(tuple(row) for row in board)
    if turn == W:
        move = white_book_moves.get(board_as_tuple)
    else:
        move = black_book_moves.get(board_as_tuple)

    if move is None:
        book_openings_still_exist = False

    print('book move:', move)

    return move

starting_board = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, W, B, 0, 0, 0),
    (0, 0, 0, B, W, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0)
)

white_book_moves = {
    starting_board: [2, 4]
}

b1v1 = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, W, 0, 0, 0),
    (0, 0, 0, W, W, 0, 0, 0),
    (0, 0, 0, B, W, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0)
)

b1v2 = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, W, W, W, 0, 0),
    (0, 0, 0, B, W, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0)
)

black_book_moves = {
    b1v1: [2, 5],
    b1v2: [2, 5]
}
