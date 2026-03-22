# Reversi Reinforcement Learning AI

The program is an AI built to play the game Reversi, using reinforcement learning techniques.

## How to Set Up

1. Create a virtual environment: `python3 -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate`
3. Install dependencies: `pip3 install -r requirements.txt`

## How to Run in Command Line Mode

1. Run `python3 src/reversi_auto_server.py`

### How to Update Algorithms Used in Command Line Mode
1. Pick two algorithms to compete.
2. Open `src/reversi_auto_server.py`
3. Update the import statements for algorithm 1 and algorithm 2 to choose different files.

### How to Create a New Player

Players must implement the following function to be able to use command line mode:

`def choose_move(turn, board, game) -> list[int]:`

See src/example_player.py for details.

## How To Run in Visual Mode

1. Start the server. WARNING: Do not click the screen yet. `python3 src/reversi_server.py` (or Windows `python .\src\reversi_server.py`)
2. Start the professor's player. `python3 src/greedy_player.py` (or Windows `python .\src\greedy_player.py`)
3. Start our player. (Currently: greedy_bfs_player.py) `python3 src/greedy_bfs_player.py` (or Windows `python .\src\greedy_bfs_player.py`)
4. Click inside the game screen to start the game.
5. Click inside the game after it has ended to close the window.

Note: You may experience crashes/infinite loops if you try to close the screen before the game has ended.