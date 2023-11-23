# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return 100000
        if currentGameState.isLose():
            return -10000000
        ghost=[]
        for g in newGhostStates:
            ghost.append(manhattanDistance(g.getPosition(),newPos))
        ghost_distance=min(ghost)
        if ghost_distance<2:
            return -100000
        food=[]
        for f in newFood.asList():
            food.append(manhattanDistance(newPos,f))
        food_distance=min(food)
        
        return successorGameState.getScore()+ghost_distance/food_distance

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def MaxValue(gameState,depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), None)
            v=float('-inf')
            best_action=None
            moves=gameState.getLegalActions(0)
            for a in  moves:
                successor= gameState.generateSuccessor(0, a)
                successor_score=MinValue(successor,depth,1)[0]
                if (successor_score > v) :
                    v=successor_score
                    best_action=a
            return (v,best_action)

        def MinValue(gameState,depth,agentIndex):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState),None)
            v=float('inf')
            best_action=None
            moves=gameState.getLegalActions(agentIndex)
            Agents=gameState.getNumAgents()
            if agentIndex == Agents-1:
                for a in moves :
                    successor=gameState.generateSuccessor(agentIndex, a)
                    successor_score=MaxValue(successor,(depth+1))[0]
                    if (successor_score < v) :
                        v=successor_score
                        best_action=a
            else:
                for a in  moves:
                    successor=gameState.generateSuccessor(agentIndex, a)
                    successor_score=MinValue(successor,depth,agentIndex+1)[0]
                    if (successor_score < v) :
                        v=successor_score
                        best_action=a
            return (v,best_action)

        return MaxValue(gameState,0)[1]
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a=float('-inf')
        b=float('inf')
        def MaxValue(gameState,depth,alpha,beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), None)
            v=float('-inf')
            best_action=None
            moves=gameState.getLegalActions(0)
            for a in  moves:
                successor= gameState.generateSuccessor(0, a)
                successor_score=MinValue(successor,depth,1,alpha,beta)[0]
                if (successor_score > v) :
                    v=successor_score
                    best_action=a
                if v > beta:
                    return (v,best_action)
                alpha=max(alpha,v)
            return (v,best_action)

        def MinValue(gameState,depth,agentIndex,alpha,beta):
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState),None)
            v=float('inf')
            best_action=None
            moves=gameState.getLegalActions(agentIndex)
            Agents=gameState.getNumAgents()
            if agentIndex == Agents-1:
                for a in moves :
                    successor=gameState.generateSuccessor(agentIndex, a)
                    successor_score=MaxValue(successor,(depth+1),alpha,beta)[0]
                    if (successor_score < v) :
                        v=successor_score
                        best_action=a
                    if v < alpha:
                        return (v,best_action)
                    beta=min(beta,v)
            else:
                for a in  moves:
                    successor=gameState.generateSuccessor(agentIndex, a)
                    successor_score=MinValue(successor,depth,agentIndex+1,alpha,beta)[0]
                    if (successor_score < v) :
                        v=successor_score
                        best_action=a
                    if v < alpha:
                        return (v,best_action)
                    beta=min(beta,v)
            return (v,best_action)

        return MaxValue(gameState,0,a,b)[1]
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def MaxValue(gameState,depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), None)
            v=float('-inf')
            best_action=None
            moves=gameState.getLegalActions(0)
            for a in  moves:
                successor= gameState.generateSuccessor(0, a)
                successor_score=ExpectimaxValue(successor,1,depth)
                if (successor_score > v) :
                    v=successor_score
                    best_action=a
            return (v,best_action)

        def ExpectimaxValue(gameState,agentIndex,depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v=0
            moves=gameState.getLegalActions(agentIndex)
            for actions in moves:
                if agentIndex == (gameState.getNumAgents()-1):
                    successor= gameState.generateSuccessor(agentIndex, actions)
                    successor_score=MaxValue(successor,depth+1)[0]
                else:
                    successor= gameState.generateSuccessor(agentIndex, actions)
                    successor_score=ExpectimaxValue(successor,agentIndex+1,depth)
                prob=1.0/len(moves)
                v=v+(prob*successor_score)
            return v
        return MaxValue(gameState,0)[1]
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 10000000
    if currentGameState.isLose():
        return -10000000
    capsule=currentGameState.getCapsules()
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates =currentGameState.getGhostPositions()
    closest_capsule=-1
    for i in capsule:
        if manhattanDistance(i,Pos)<closest_capsule or closest_capsule==-1:
            closest_capsule=manhattanDistance(i,Pos)
    closest_food=-1
    for i in Food.asList():
        if manhattanDistance(i,Pos)<closest_food or closest_food==-1:
            closest_food=manhattanDistance(i,Pos)
    closest_ghost=-1
    for i in GhostStates:
        if manhattanDistance(i,Pos)<closest_ghost or closest_ghost==-1:
            closest_ghost=manhattanDistance(i,Pos)
    return currentGameState.getScore()+(1.0/closest_capsule)+(closest_ghost/closest_food)
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
