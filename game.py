import os
import numpy as np
import random

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

class State:
    def __init__(self, player1, player2):
        self.board = np.zeros((3,3),dtype=int) # 0 = empty, 1 = 'X', -1 = 'O'
        self.p1 = player1
        self.p2 = player2
        self.curPlayer = 1
        self.boardHash = None
        self.gameEnd = False
        self.winReward = 1
        self.drawReward = 0
        self.loseReward = -1
        self.wrongMoveReward = -1000

    def reset(self):
        self.board.fill(0)
        self.curPlayer = 1
        self.boardHash = None
        self.gameEnd = False

    def checkWin(self):
        if len(self.availPos()) == 0:
            return 0
        
        for p in [-1,1]:
            for i in range(3):
                if sum(self.board[i, :]) == p*3:
                    self.gameEnd = True
                    return p
                if sum(self.board[:, i]) == p*3:
                    self.gameEnd = True
                    return p
            
            if self.board[0][0] == p and self.board[1][1] == p and self.board[2][2] == p:
                self.gameEnd = True
                return p
            if self.board[0][2] == p and self.board[1][1] == p and self.board[2][0] == p:
                self.gameEnd = True
                return p

        return None
    
    def availPos(self):
        pos = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    pos.append((i,j))
        return pos
    
    def updateBoardHash(self):
        self.boardHash = str(self.board.reshape(9))
        return self.boardHash
    
    def makeMove(self, move):
        if move not in self.availPos():
            return -1
        row, col = move
        self.board[row][col] = self.curPlayer
        self.updateBoardHash()
        return 1
        
    def printBoard(self):
        clear_terminal()
        for i in range(3):
            seperator = " | "
            for j in range(3):
                if j == 2:
                    seperator = "\n"
                if self.board[i][j] == 1:
                    print("X", end=seperator)
                elif self.board[i][j] == -1:
                    print("O", end=seperator)
                else:
                    print(" ", end= seperator)
            print("---------")

    def game(self):
        while not self.gameEnd:
            move = self.p1.chooseAction(self) if self.curPlayer == 1 else self.p2.chooseAction(self)
            checkValidMove = self.makeMove(move)
            if checkValidMove == -1:
                if self.curPlayer == 1:
                    self.p1.updateQTable(self, move, self.wrongMoveReward)
                else:
                    self.p2.updateQTable(self, move, self.wrongMoveReward)
                self.gameEnd = True
                break
            win = self.checkWin()
            if win == 1 or win == -1:
                self.gameEnd = True
                # do stuff here
            elif win == 0:
                self.gameEnd = True
                # do stuff here

class Player:
    def __init__(self, alpha=0.2, gamma=0.8, epsilon=0.2):
        self.qTable = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def getQVals(self, state):
        stateHash = state.updateBoardHash()
        if stateHash not in self.qTable:
            self.qTable[stateHash] = np.zeros(3,3)
        return self.qTable[stateHash]
    
    def chooseAction(self, state):
        validMoves = state.availPos()
        if random.uniform(0,1) < self.epsilon:
            return random.choice(validMoves)
        qVals = self.getQVals(state)
        return np.argmax(qVals) # maybe issue with tuple output
    
    # this all needs work
    def updateQTable(self, state, action, reward, res = None):
        qVals = self.getQVals(state)
        row, col = action
        if res != None:
            qVals[row][col] += self.alpha * (reward - qVals[row][col])
            return
        nextState = state.copy()
        nextState.makeMove(action)
        nextQVals = self.getQVals(nextState)
        qVals[row][col] += self.alpha * (reward + self.gamma*np.max(nextQVals) - qVals[row][col])

    def saveQTable(self):
        with open("qTable.txt", "a") as f:
            f.write("begin\n")
            for i in self.qTable:
                f.write(f"{i} {self.qTable[i]}\n")
            f.write("end\n")


class HumanPlayer:
    def __init__(self):
        pass

    def chooseAction(self, state):
        validMoves = state.availPos()
        while True:
            move = input("Enter your move: (a,b)")
            move = move.split(',')
            move = (int(move[0]), int(move[1]))
            if move in validMoves:
                return move