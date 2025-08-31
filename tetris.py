import random
from time import sleep

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }


    def __init__(self):
        self._window_initialized = False
        self._rotation_cache = {}
        self.reset()

    
    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board)


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        key = (self.current_piece, self.current_rotation)
        piece = self._rotation_cache.get(key)
        if piece is None:
            piece = np.array(Tetris.TETROMINOS[self.current_piece][self.current_rotation])
            self._rotation_cache[key] = piece
        return piece


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece() + self.current_pos
        board = [row[:] for row in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score
    

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''        
        board = [row[:] for row in self.board]
        px, py = int(pos[0]), int(pos[1])
        for x, y in piece:
            board[y + py][x + px] = Tetris.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        board_np = np.array(board)
        full_lines = np.where(np.all(board_np == Tetris.MAP_BLOCK, axis=1))[0]
        if full_lines.size:
            board_np = np.delete(board_np, full_lines, axis=0)
            new_rows = np.zeros((full_lines.size, Tetris.BOARD_WIDTH), dtype=int)
            board_np = np.vstack((new_rows, board_np))
        return int(full_lines.size), board_np.tolist()


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty square with at least one block above it)'''
        board_np = np.array(board)
        filled = board_np == Tetris.MAP_BLOCK
        filled_cumsum = np.cumsum(filled, axis=0)
        holes = np.sum((board_np == Tetris.MAP_EMPTY) & (filled_cumsum > 0))
        return int(holes)


    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness


    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height


    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]


    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            key = (piece_id, rotation)
            piece = self._rotation_cache.get(key)
            if piece is None:
                piece = np.array(Tetris.TETROMINOS[piece_id][rotation])
                self._rotation_cache[key] = piece
            min_x = int(piece[:, 0].min())
            max_x = int(piece[:, 0].max())

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = np.array([x, 0])

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states


    def get_state_size(self):
        '''Size of the state'''
        return 4


    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score        
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over


    def render(self):
        '''Renders the current board'''
        if cv2 is None:
            raise ImportError('OpenCV is required for rendering')
        if not self._window_initialized:
            cv2.startWindowThread()
            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            self._window_initialized = True
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)

    def close(self):
        '''Releases rendering resources'''
        if cv2 is not None and self._window_initialized:
            cv2.destroyAllWindows()
            self._window_initialized = False
