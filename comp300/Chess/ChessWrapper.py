import random
import tempfile

import chess
import chess.engine
import chess.svg
import cv2
import numpy as np
from cairosvg import svg2png
from gym import Env
from gym.spaces import Discrete, Box, Dict

WHITEWIN = '1-0'
BLACKWIN = '0-1'
DRAW = '1/2-1/2'

class ChessEnv(Env):
    """The chess env that defines the interaction between gym and pychess."""

    metadata = {'render.modes': ['human', 'asci', 'rgb_array']}

    def __init__(self, opponent = "stockfish", startingSide = None, limit = chess.engine.Limit(depth=1)):
        """
        The constructor that initialises the chess board and the opponent engine.

        Parameters
        ----------
        opponent : str
            A sting describing the opponent to use (either 'stockfish' or 'random').
        startingSide : str
            The side to start on either 'white' or 'black'. if nothing given default is random.
        limit : chess.engine.Limit
            The limit on the opponent when searching for moves.
        """
        #set up the board and the opponent
        self.board = chess.Board()
        self.opponent = opponent
        if opponent == "stockfish":
            self.opponent_engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
            self.opponent_engine.configure({"Skill Level": 5})
        self.limit = limit

        if startingSide == "white":
            self.side = 0
        elif startingSide == "black":
            self.side = 1
        else:
            self.side = random.choice([0,1])

        #set up the action and observation space also set up reward range
        #the observation space is just a grid with 8x8 squares of intergers. 0 = empty, n = white piece, -n = black piece
        #the observation space also has an input for what side the agent is
        self.observation_space = Dict(dict(
            obs = Box(-np.inf, np.inf, shape=(8,8), dtype=np.int64),
            side = Discrete(2)))
        #action space is the index in possible moves
        self.action_space = Discrete(218) #most number of moves in a single turn

        #look at base board class for info on how to access board states
        self.done = False
        self.illegal_moves = 0
        self.legal_moves = 0

    def reset(self):
        """
        Reset the board to the start and pick a new side

        Returns
        -------

        """
        self.board.reset()
        self.done = False
        self.illegal_moves = 0
        self.legal_moves = 0
        self.side = random.choice([0, 1])
        if self.side == 1:
            #let oppenent act
            self.board.push(self.getOpponentMove())

        return self.getObs()

    def getOpponentMove(self):
        """
        Get the move from the opponent engine.

        Returns
        -------
        Move
            the move chosen by the opponent.
        """
        if self.opponent == "stockfish":
            result = self.opponent_engine.play(self.board, self.limit)
            opponentMove = result.move
        else:
            opponentMove = random.sample(list(self.board.generate_legal_moves()), 1)

        return opponentMove

    def find_piece(self, piece_number):
        """
        Finds the board positions of the piece.

        Parameters
        ----------
        piece_number : int
            The index of the piece from the start of the board.

        Returns
        -------
        int
            The board index of the piece.
        """
        piece_locations = []
        piece_map = self.board.piece_map()

        for square in range(len(piece_map)):
            piece= piece_map.get(square)
            if piece != None:
                piece_colour = piece.color
                if (self.side == 0 and piece_colour == chess.WHITE) or (self.side == 1 and piece_colour == chess.BLACK):
                    piece_locations.append(square)

        if piece_number >= len(piece_locations):
            return 0
        else:
            return piece_locations[piece_number]

    def step(self, action):
        """
        Take a the action on the board.

        Parameters
        ----------
        action : int
            The index of the move it wants to make in possible moves.

        Returns
        -------
        obs
            The dictionary of observations from the board.
        reward
            The reward from this time step.
        done
            A boolean of if the game is done yet.
        info
            A dictionary containing debug info about the env.
        """
        #generate all moves
        legalMoves = list(self.board.generate_legal_moves())


        if action < len(legalMoves):
            EquivilentMove = legalMoves[action]
            self.legal_moves += 1
            self.board.push(EquivilentMove)


            if self.board.is_game_over():
                self.done = True
                results = self.getResults()
                if self.side == 0: #if white
                    reward = results[0]
                else:
                    reward = results[1]

                return self.getObs(), reward, self.done, {}
            else:
                #if not done then take opponent turn
                self.board.push(self.getOpponentMove())
                if self.board.is_game_over():
                    self.done = True
                    results = self.getResults()
                    if self.side == 0:  # if white
                        reward = results[0]
                    else:               # if black
                        reward = results[1]

                    return self.getObs(), reward, self.done, {}
                else:
                    return self.getObs(), 1, self.done, {}
        else: #if not legal move
            self.illegal_moves += 1
            reward = -1
            if self.illegal_moves > 50:
                self.done = True
                reward = -100
            doneBefore = self.done
            obs, secondReward, self.done, _ = self.step(random.choice(range(len(legalMoves))))
            if not doneBefore and self.done:
              return obs, secondReward, self.done, {}
            return obs, reward, self.done, {}

    def getResults(self):
        """
        Calculates the results of the game if it is finished.

        Returns
        -------
        results
            A tuple containing the (reward for white, reward for black).

        """
        resultStr = self.board.result(claim_draw=True)
        if resultStr == WHITEWIN:
            result = (100, -100)
        elif resultStr == BLACKWIN:
            result = (-100, 100)
        elif resultStr == DRAW:
            result = (0, 0)
        else:
            raise Exception('Error unknown result found for game result string = {}'.format(resultStr))
            result = None

        return result

    def render(self, mode='human'):
        """
        Renders the board in the given mode.

        Parameters
        ----------
        mode : str
            The type of rendering that is needed. human = render a cv2 window, asci = print and return and asci
            representation of the board, rgb_array = render the board and return a 3d array of values.

        Returns
        -------
        board
            Either an rgb array of the board state or a string reprsentation.

        """
        if mode == 'human':
            #render the board in a human readable form or if needed in a chess form

            svgInfo = chess.svg.board(board=self.board)
            pngSurface = svg2png(bytestring=bytes(svgInfo, 'UTF-8'))
            save_file = tempfile.NamedTemporaryFile()
            name = save_file.name
            save_file.write(pngSurface)
            image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            save_file.close()
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [255, 255, 255, 255]
            new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            cv2.imshow('Chess Environment', new_img)
            cv2.waitKey(delay=100)

        elif mode == 'asci':
            print(self.board)
            return str(self.board)

        elif mode == 'rgb_array':
            svgInfo = chess.svg.board(board=self.board)
            pngSurface = svg2png(bytestring=bytes(svgInfo, 'UTF-8'))
            save_file = tempfile.NamedTemporaryFile()
            name = save_file.name
            save_file.write(pngSurface)
            image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            save_file.close()
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [255, 255, 255, 255]
            new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return new_img


    def close(self):
        """
        Closes the environment and clears the opponent engine
        Returns
        -------

        """
        #close the dependencies such as the game system and opponent
        self.board.clear()
        #also destroy opponent
        if self.opponent == "stockfish":
            self.opponent_engine.quit()

    def seed(self, seed=None):
        """
        Seeds the randomness in the env.

        Parameters
        ----------
        seed : int
            The seed for randomness generation.

        Returns
        -------

        """
        random.seed(seed)

    def getObs(self):
        """
        Calculates the observations from the current board position.

        Returns
        -------
        Dict
            A dictionary containing the obs, the board state encoded as a 8x8 array with ech piece given a number by
            type, and the side, and int encoding which side the player is on.
        """
        #get the array of the current board state
        boardState = np.zeros((8,8),dtype=np.int64)
        pieceMap = self.board.piece_map()

        for row in range(8):
            for column in range(8):
                index = column + (row * 8)
                piece = pieceMap.get(index)

                if piece != None:
                    if piece.color == chess.WHITE:
                        boardState[7-row][column] = piece.piece_type
                    else:
                        boardState[7-row][column] = -piece.piece_type


        obs = {
            'obs':boardState.copy(),
            'side':self.side,
        }
        return obs

if __name__ == '__main__':
    from baselines import run
    from gym import register
    import sys

    register(id='ChessSelf-v0',
             entry_point='comp300.Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)

    args = sys.argv
    run.main(args)
