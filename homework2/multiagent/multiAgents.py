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


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
float("Inf")The code below extracts some useful information from the state, like the
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
        ghost_distance = [manhattanDistance( newPos, ghost.getPosition() ) for ghost in newGhostStates]
        min_ghost = min(ghost_distance)
        if min_ghost <= 1: return float("-Inf")
        food_distance = [manhattanDistance( newPos, food_p ) for food_p in newFood.asList()]        
        min_fd = 0
        if food_distance: min_fd = min(food_distance)
        cur_food = currentGameState.getFood().asList()
        if len(cur_food) > len(food_distance): min_fd = 0
        bad_score = min_fd
        if action == Directions.STOP:
            bad_score += 50

        return successorGameState.getScore() - bad_score

def scoreEvaluationFunction(currentGameState):
    """
      This default float("Inf") function just returns the score of the state.
      The float("Inf") is the same one displayed in the Pacman GUI.

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
      float("Inf") another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():float("Inf")Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        maxAction = None
        maximumScore = float("-Inf")
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0,action)
            score = self.min_Max(state,self.depth*gameState.getNumAgents()-1)
            if maximumScore < score:
                maximumScore = score
                maxAction = action
        return maxAction

    def min_Max(self,gameState,depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        n = gameState.getNumAgents()
        agentIndex = (n - (depth % gameState.getNumAgents())) % n
        val = [] 
        for action in gameState.getLegalActions(agentIndex):
            val.append(self.min_Max(gameState.generateSuccessor(agentIndex,action),depth-1))
        if agentIndex == 0:
            return max(val)
        else:
            return min(val)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxAction = None
        maximumScore = float("-Inf")
        alpha = float("-Inf")
        beta = float("Inf")
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0,action)
            score = self.AlphaBeta(state,self.depth*gameState.getNumAgents()-1,alpha,beta)
            alpha = max(alpha, score)
            #print score,action
            if maximumScore < score:
                maximumScore = score
                maxAction = action
        return maxAction

    def AlphaBeta(self,gameState,depth,alpha,beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        maximumScore = float("Inf")
        n = gameState.getNumAgents()
        agentIndex = (n - (depth % gameState.getNumAgents())) % n
        if agentIndex == 0:
            maximumScore = float("-Inf")
        for action in gameState.getLegalActions(agentIndex):
            val = self.AlphaBeta(gameState.generateSuccessor(agentIndex,action),depth-1,alpha,beta)
            if agentIndex == 0:
                if val > beta:
                    return val
                alpha = max(alpha, val)
                maximumScore = max(maximumScore,val)
                continue
            elif val < alpha:
                return val
            beta = min(beta,val)
            maximumScore = min(maximumScore,val)
        return maximumScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxAction = None
        maximumScore = float("-Inf")
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0,action)
            score = self.Expectimax(state,self.depth*gameState.getNumAgents()-1)
            if maximumScore < score:
                maximumScore = score
                maxAction = action
        return maxAction

    def Expectimax(self,gameState,depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        n = gameState.getNumAgents()
        agentIndex = (n - (depth % gameState.getNumAgents())) % n
        val = [self.Expectimax(gameState.generateSuccessor(agentIndex,action),depth-1) for action in gameState.getLegalActions(agentIndex)]
        if agentIndex == 0:
            return max(val)
        return float(sum(val))/float(len(val))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    curPos = currentGameState.getPacmanPosition()
    foodDis = [manhattanDistance( curPos, fp ) for fp in currentGameState.getFood().asList()]
    ghost = currentGameState.getGhostStates()
    minimumFood = 0
    if foodDis: minimumFood = min(foodDis)
    ghost_dis = []
    for g in ghost:
        if g.scaredTimer < manhattanDistance(curPos, g.getPosition()):
            ghost_dis.append(manhattanDistance(curPos, g.getPosition()))
    min_ghost = float("Inf")
    if ghost_dis: min_ghost = min(ghost_dis)
    if min_ghost <= 0: return float("-Inf")
    bad_score = minimumFood
    return currentGameState.getScore() - bad_score

# Abbreviation
better = betterEvaluationFunction

