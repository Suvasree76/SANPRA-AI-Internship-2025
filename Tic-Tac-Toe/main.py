import math
class Board:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
    def print_board(self):
        """Prints the Tic-Tac-Toe board."""
        for i in range(0, 9, 3):
            print(f' {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} ')
            if i < 6:
                print('---|---|---')
    def available_moves(self):
        """Returns a list of available moves (empty spots) on the board."""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    def make_move(self, square, letter):
        """Makes a move on the board."""
        if self.board[square] == ' ':
            self.board[square] = letter
            return True
        return False
def get_player_move(board):
    """Gets the human player's move."""
    valid_square = False
    while not valid_square:
        try:
            square = int(input("Enter your move (0-8): "))
            if square in board.available_moves():
                valid_square = True
            else:
                print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return square
def check_winner(board, letter):
    """Checks if the given letter has won the game."""
    for i in range(0, 9, 3):
        if all(board.board[j] == letter for j in range(i, i + 3)):
            return True
    for i in range(3):
        if all(board.board[j] == letter for j in range(i, 9, 3)):
            return True
    if all(board.board[j] == letter for j in [0, 4, 8]):
        return True
    if all(board.board[j] == letter for j in [2, 4, 6]):
        return True
    return False
def game_over(board):
    """Checks if the game is over (win or draw)."""
    if check_winner(board, 'X') or check_winner(board, 'O') or len(board.available_moves()) == 0:
        return True
    return False
def minimax(board, depth, maximizing_player):
    """Minimax algorithm implementation."""
    if check_winner(board, 'O'):
        return 1
    elif check_winner(board, 'X'):
        return -1
    elif len(board.available_moves()) == 0:
        return 0
    if maximizing_player:
        max_eval = -math.inf
        for move in board.available_moves():
            board.make_move(move, 'O')
            evaluation = minimax(board, depth + 1, False)
            board.board[move] = ' '
            max_eval = max(max_eval, evaluation)
        return max_eval
    else:
        min_eval = math.inf
        for move in board.available_moves():
            board.make_move(move, 'X')
            evaluation = minimax(board, depth + 1, True)
            board.board[move] = ' '
            min_eval = min(min_eval, evaluation)
        return min_eval
def get_ai_move(board):
    """Gets the AI's move using the minimax algorithm."""
    best_score = -math.inf
    best_move = None
    for move in board.available_moves():
        board.make_move(move, 'O')
        score = minimax(board, 0, False)
        board.board[move] = ' '
        if score > best_score:
            best_score = score
            best_move = move
    return best_move
def play_game():
    """Main function to play the Tic-Tac-Toe game."""
    board = Board()
    print("Welcome to Tic-Tac-Toe!")
    board.print_board()
    while not game_over(board):
        player_move = get_player_move(board)
        board.make_move(player_move, 'X')
        board.print_board()
        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break
        if len(board.available_moves()) == 0:
            print("It's a draw!")
            break
        print("AI's turn...")
        ai_move = get_ai_move(board)
        board.make_move(ai_move, 'O')
        board.print_board()
        if check_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break
if __name__ == '__main__':
    play_game()
