import pygame
from sys import exit
import numpy as np
import itertools
import threading
from reversi import reversi
from minimax_alpha_beta_h_nic_nn import choose_move as ai_choose_move, get_legal_moves


class drawable_reversi(reversi):
    def __init__(self, _white_pic, _black_pic):
        super().__init__()
        self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.small_font = pygame.font.Font('freesansbold.ttf', 20)
        self.white_pic = _white_pic
        self.black_pic = _black_pic

    def render_text(self, _screen, _text, x, y, color=(255, 255, 255), font=None):
        if font is None:
            font = self.font
        text = font.render(_text, True, color)
        text_rect = text.get_rect()
        text_rect.center = (x, y)
        _screen.fill((0, 0, 0), text_rect)
        _screen.blit(text, text_rect)

    def render(self, _screen):
        if self.white_count > 0:
            white_cords = np.c_[np.where(self.board == 1)] * 100 + 15
            white_pics = list(zip(itertools.repeat(self.white_pic, white_cords.shape[0]),
                                  tuple(map(tuple, white_cords))))
            _screen.blits(white_pics)

        if self.black_count > 0:
            black_cords = np.c_[np.where(self.board == -1)] * 100 + 15
            black_pics = list(zip(itertools.repeat(self.black_pic, black_cords.shape[0]),
                                  tuple(map(tuple, black_cords))))
            _screen.blits(black_pics)

        self.render_text(_screen, f'White : {self.white_count}', 1000, 100)
        self.render_text(_screen, f'Black : {self.black_count}', 1000, 200)
        self.render_text(_screen, f'Hand : {"White" if self.turn == 1 else "Black"}', 1000, 500)


def draw_legal_moves(screen, legal_moves):
    """Draw semi-transparent green circles on legal move squares."""
    for (r, c) in legal_moves:
        cx = r * 100 + 50
        cy = c * 100 + 50
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(surf, (0, 200, 0, 100), (20, 20), 20)
        screen.blit(surf, (cx - 20, cy - 20))


def draw_board(screen, background, game, legal_moves=None):
    """Redraw the full board, pieces, and UI."""
    screen.blit(background, (0, 0))
    for i in range(7):
        pygame.draw.line(screen, (255, 255, 255), (100 * i + 100, 0), (100 * i + 100, 800), 2)
        pygame.draw.line(screen, (255, 255, 255), (0, 100 * i + 100), (800, 100 * i + 100), 2)
    game.render(screen)
    if legal_moves:
        draw_legal_moves(screen, legal_moves)


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption('Reversi - Human vs AI')
    clock = pygame.time.Clock()

    background_surface = pygame.image.load('data/background.jpeg')
    background_surface = pygame.transform.scale(background_surface, (800, 800))

    white_piece = pygame.image.load('data/white_piece.png')
    white_piece = pygame.transform.scale(white_piece, (70, 70))
    black_piece = pygame.image.load('data/black_piece.png')
    black_piece = pygame.transform.scale(black_piece, (70, 70))

    game = drawable_reversi(white_piece, black_piece)

    # Color selection screen
    game.render_text(screen, 'Click to choose your color:', 600, 300)
    game.render_text(screen, '[W] = White (go first)', 600, 400)
    game.render_text(screen, '[B] = Black (go second)', 600, 500)
    pygame.display.update()

    human_color = None
    while human_color is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    human_color = 1
                elif event.key == pygame.K_b:
                    human_color = -1

    ai_color = -human_color
    game.turn = 1  # White always starts
    consecutive_passes = 0
    game_over = False

    # State for AI thinking in background thread
    ai_thinking = False
    ai_result = [None]  # mutable container for thread result

    def ai_think():
        move = ai_choose_move(game.turn, game.board.copy(), game)
        ai_result[0] = move

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if game_over:
                # Click to exit after game ends
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.quit()
                    exit()
                continue

            # Human's turn: handle clicks
            if game.turn == human_color and not ai_thinking:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if mx < 800 and my < 800:
                        row = mx // 100
                        col = my // 100
                        search_game = reversi()
                        search_game.board = game.board.copy()
                        legal = get_legal_moves(search_game, game.turn)

                        if len(legal) == 0:
                            # Human has no moves, auto-pass
                            consecutive_passes += 1
                            game.turn *= -1
                        elif (row, col) in legal:
                            result = game.step(row, col, game.turn)
                            if result >= 0:
                                consecutive_passes = 0
                                game.turn *= -1
                            # else: illegal, ignore click

        if game_over:
            draw_board(screen, background_surface, game)
            white = game.white_count
            black = game.black_count
            game.render_text(screen, f'Game Over!', 1000, 400)
            if white > black:
                winner_text = 'You Win!' if human_color == 1 else 'AI Wins!'
            elif black > white:
                winner_text = 'You Win!' if human_color == -1 else 'AI Wins!'
            else:
                winner_text = 'Draw!'
            game.render_text(screen, winner_text, 1000, 450)
            game.render_text(screen, 'Click to exit', 1000, 700, font=game.small_font)
            pygame.display.update()
            clock.tick(30)
            continue

        # Check for game end
        if consecutive_passes >= 2:
            game_over = True
            continue

        # Check if current player must pass (no legal moves)
        search_game = reversi()
        search_game.board = game.board.copy()
        current_legal = get_legal_moves(search_game, game.turn)

        if len(current_legal) == 0 and not ai_thinking:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                game_over = True
                continue
            game.turn *= -1
            # Recompute legal moves for the new turn
            search_game.board = game.board.copy()
            current_legal = get_legal_moves(search_game, game.turn)

        # AI's turn: kick off thinking thread
        if game.turn == ai_color and not ai_thinking and not game_over:
            ai_thinking = True
            ai_result[0] = None
            t = threading.Thread(target=ai_think, daemon=True)
            t.start()

        # Check if AI is done thinking
        if ai_thinking and ai_result[0] is not None:
            move = ai_result[0]
            x, y = move
            if x == -1 and y == -1:
                consecutive_passes += 1
            else:
                result = game.step(x, y, game.turn)
                if result >= 0:
                    consecutive_passes = 0
                else:
                    consecutive_passes += 1
            game.turn *= -1
            ai_thinking = False

        # Draw
        if game.turn == human_color and not ai_thinking:
            search_game = reversi()
            search_game.board = game.board.copy()
            legal = get_legal_moves(search_game, game.turn)
            draw_board(screen, background_surface, game, legal)
        else:
            draw_board(screen, background_surface, game)

        if ai_thinking:
            game.render_text(screen, 'AI thinking...', 1000, 400)

        pygame.display.update()
        clock.tick(30)


if __name__ == '__main__':
    main()
