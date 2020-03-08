from gym import Env, logger
from gym.spaces import Discrete, MultiDiscrete, Box, Dict
from gym.utils import colorize, seeding
import numpy as np
import chess
import chess.engine
import random
import chess.svg
from cairosvg import svg2png
import cv2
import tempfile

WHITEWIN = '1-0'
BLACKWIN = '0-1'
DRAW = '1/2-1/2'

class ChessEnv(Env):

    metadata = {'render.modes': ['human', 'asci', 'rgb_array']}

    def __init__(self, opponent = "stockfish", startingSide = None, limit = chess.engine.Limit(depth=1)):

        #set up the board and the opponent
        self.board = chess.Board()
        self.opponent = opponent
        if opponent == "stockfish":
            self.opponent_engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
            self.opponent_engine.configure({"Skill Level": 5})
        self.limit = limit

        if startingSide == None:
            self.side = random.choice([0,1])
        elif startingSide == "white":
            self.side = 0
        else:
            self.side = 1

        #set up the action and observation space also set up reward range
        #the observation space is just a grid with 8x8 squares of intergers. 0 = empty, n = white piece, -n = black piece
        #the observation space also needs an input for what side the agent is
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
        #reset the board to the start
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
        if self.opponent == "stockfish":
            result = self.opponent_engine.play(self.board, self.limit)
            opponentMove = result.move
        else:
            opponentMove = random.sample(list(self.board.generate_legal_moves()), 1)

        return opponentMove

    def find_piece(self, piece_number):
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
        #apply the action given to the game system but it needs to be converted to a string first
        #apply the given move
        #piece_position = self.find_piece(action[0])
        #EquivilentMove = chess.Move(piece_position, int(action[1]))
        legalMoves = list(self.board.generate_legal_moves())


        #if self.board.is_legal(EquivilentMove):
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
            obs, _, self.done, _ = self.step(random.choice(range(len(legalMoves))))
            return obs, reward, self.done, {}

    def getResults(self):
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

    def close(self):
        #close the dependencies such as the game system and opponent
        self.board.clear()
        #also destroy opponent

    def seed(self, seed=None):
        #seed the starting position (i guess this means make the opponet act differently or something not sure)
        return

    def getObs(self):
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
             entry_point='Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)

    envobj = ChessEnv()
    reset = envobj.reset()

    env = 'ChessSelf-v0'
    alg = 'ppo2'
    steps = '20000000' #20m
    save_path = "~/models/chessTest/chess20Mppo2"
    args = sys.argv
    args.append("--alg={}".format(alg))
    args.append("--env={}".format(env))
    args.append("--num_timesteps={}".format(steps))
    args.append("--save_path={}".format(save_path))
    args.append("--network=mlp")
    args.append("--num_layers=5")

    run.main(args)
