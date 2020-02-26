from gym import Env, logger
from gym.spaces import Discrete, Tuple, Box, Dict
from gym.utils import colorize, seeding
import numpy as np
import chess
import random

WHITEWIN = '1-0'
BLACKWIN = '0-1'
DRAW = '1/2-1/2'

class ChessEnv(Env):

    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}

    def __init__(self, opponent = "stockfish", startingSide = "white"):

        #set up the board and the opponent
        self.board = chess.Board()
        #todo: make real opponent using stockfish
        self.opponent = "random"

        #todo: make the starting side random
        if startingSide == "white":
            self.side = 1
        else:
            self.side = 2

        #set up the action and observation space also set up reward range
        #the observation space is just a grid with 8x8 squares of intergers. 0 = empty, n = white piece, -n = black piece
        #the observation space also needs an input for what side the agent is
        self.observation_space = Dict(dict(
            obs = Box(-np.inf, np.inf, shape=(8,8), dtype=np.int64),
            side = Discrete(2)))
        #look at base board class for info on how to access board states
        self.done = False
        #the action space will consist of a single int that is the index of the possible legal moves

    @property
    def action_space(self):
        #the action space needs to be a property that is calculated when called because the set of legal moves in
        #chess is much smaller than the set of illegal moves
        moves = list(self.board.generate_legal_moves())
        _action_space = Discrete(len(moves))
        return _action_space

    def reset(self):
        #reset the board to the start
        self.board.reset()
        self.done = False
        #also maybe set a new side??

        return self.getObs()

    def step(self, action):
        #apply the action given to the game system but it needs to be converted to a string first
        #apply the given move
        legalMoves = list(self.board.generate_legal_moves())
        EquivilentMove  = legalMoves[action]

        if self.board.is_legal(EquivilentMove):
            self.board.push(EquivilentMove)

            if self.board.is_game_over():
                self.done = True
                results = self.getResults()
                if self.side == 1: #if white
                    reward = results[0]
                else:
                    reward = results[1]

                return self.getObs(), reward, self.done, {}
            else:
                #if not done then take opponent turn
                #for now just sample random moves
                opponentMove = random.sample(list(self.board.generate_legal_moves()), 1)
                self.board.push(opponentMove[0])
                if self.board.is_game_over():
                    self.done = True
                    results = self.getResults()
                    if self.side == 1:  # if white
                        reward = results[0]
                    else:               # if black
                        reward = results[1]

                    return self.getObs(), reward, self.done, {}
                else:
                    return self.getObs(), 0, self.done, {}
        else: #if not legal move
            raise Exception("Error illegal move selected")

    def getResults(self):
        resultStr = self.board.result(claim_draw=True)
        if resultStr == WHITEWIN:
            result = (1, -1)
        elif resultStr == BLACKWIN:
            result = (-1, 1)
        elif resultStr == DRAW:
            result = (0, 0)
        else:
            raise Exception('Error unknown result found for game result string = {}'.format(resultStr))
            result = None

        return result

    def render(self, mode='human'):
        #render the board in a human readable form or if needed in a chess form
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
             max_episode_steps=100000)

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

    run.main(args)
