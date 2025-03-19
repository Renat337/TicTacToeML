import os
import pickle
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
        row, col = move
        self.board[row][col] = self.curPlayer
        self.curPlayer *= -1
        self.updateBoardHash()
        
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
            self.makeMove(move)
            win = self.checkWin()
            if win == 1 or win == -1:
                # do stuff here
                break
            elif win == 0:
                # do stuff here
                break

class Player:
    def __init__(self, alpha=0.2, gamma=0.8, epsilon=0.2):
        self.qTable = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def getQVals(self, state):
        stateVal = state.updateBoardHash()
        if stateVal not in self.qTable:
            self.qTable[stateVal] = np.zeros(9)
        return self.qTable[stateVal]
    
    def chooseMove(self, state):
        validMoves = state.availPos()
        if random.uniform(0,1) < self.epsilon:
            return random.choice(validMoves)
        qVals = self.getQ(state)
        return divmod(np.argmax(qVals), 3)
    
    def updateQTable(self, state, action, reward, nextState):
        qVals = self.getQ(state)
        nextQVals = self.getQ(nextState)
        row, col = action
        index = row*3 + col
        qVals[index] += self.alpha * (reward + self.gamma*np.max(nextQVals) - qVals[index])

    def chooseAction(self, state: State, reward):
        action = self.chooseMove(state)
        nextState = state.copy()
        nextState.makeMove(action)
        self.updateQTable(state, action, reward, nextState)
        return action
    

class HumanPlayer:
    def __init__(self):
        pass

    def chooseAction(self, state):
        validMoves = state.availPos()
        while True:
            move = input("Enter your move: ")
            move = move.split(',')
            move = (int(move[0]), int(move[1]))
            if move in validMoves:
                return move