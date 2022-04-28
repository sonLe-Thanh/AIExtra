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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newCapsules = currentGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        foodList = newFood.asList()
        minFoodDis = -1
        minGhostDis = 1
        minCapsuleDis = -1

        for food in foodList:
            distance = manhattanDistance(food, newPos)
            if minFoodDis >= distance or minFoodDis == -1:
                minFoodDis = distance

        for capsule in newCapsules:
            distance = manhattanDistance(capsule, newPos)
            if minCapsuleDis >= distance or minCapsuleDis == -1:
                minCapsuleDis = distance
        if minCapsuleDis == 0:
            minCapsuleDis = 0.001

        for ghost in newGhostStates:
            distanceToGhost = manhattanDistance(ghost.getPosition(), newPos)
            if ghost.scaredTimer == 0:
                minGhostDis += distanceToGhost
            else:
                minGhostDis -= distanceToGhost
        if minGhostDis == 0:
            minGhostDis = -1

        score = float(4 / minFoodDis) - float(2 / minGhostDis) + float(5 / minCapsuleDis)

        return successorGameState.getScore() + score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, agent, depth):
            #Reset agent to pacman
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maximizer(gameState, agent, depth)
            else:
                return minimizer(gameState, agent, depth)

        def maximizer(gameState, agent, depth):
            bestAction = ["max", float("-inf")]
            pacmanActions = gameState.getLegalActions(agent)

            if not pacmanActions:
                return self.evaluationFunction(gameState)

            for action in pacmanActions:
                nextMove = gameState.generateSuccessor(agent, action)
                nextValue = minimax(nextMove, agent+1, depth)
                if isinstance(nextValue,list):
                    val = nextValue[1]
                else:
                    val = nextValue
                if val > bestAction[1]:
                    bestAction = [action, val]
            return bestAction

        def minimizer(gameState, agent, depth):
            bestAction = ["min", float("inf")]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                nextMove = gameState.generateSuccessor(agent, action)
                nextValue = minimax(nextMove, agent+1, depth)
                if isinstance(nextValue,list):
                    val = nextValue[1]
                else:
                    val = nextValue
                if val < bestAction[1]:
                    bestAction = [action, val]
            return bestAction

        return minimax(gameState, 0, 0)[0]
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(gameState, agent, depth, alpha, beta):
            # Reset agent to pacman
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maximizer(gameState, agent, depth, alpha, beta)
            else:
                return minimizer(gameState, agent, depth, alpha, beta)

        def maximizer(gameState, agent, depth, alpha, beta):
            bestAction = ["max", float("-inf")]
            pacmanActions = gameState.getLegalActions(agent)

            if not pacmanActions:
                return self.evaluationFunction(gameState)

            for action in pacmanActions:
                nextMove = gameState.generateSuccessor(agent, action)
                nextValue = alphaBeta(nextMove, agent + 1, depth, alpha, beta)
                if isinstance(nextValue, list):
                    val = nextValue[1]
                else:
                    val = nextValue
                if val > bestAction[1]:
                    bestAction = [action, val]
                #Pruning
                if val > beta:
                    return [action, val]
                alpha = max(alpha, val)
            return bestAction

        def minimizer(gameState, agent, depth, alpha, beta):
            bestAction = ["min", float("inf")]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)

            for action in ghostActions:
                nextMove = gameState.generateSuccessor(agent, action)
                nextValue = alphaBeta(nextMove, agent + 1, depth, alpha, beta)
                if isinstance(nextValue, list):
                    val = nextValue[1]
                else:
                    val = nextValue
                if val < bestAction[1]:
                    bestAction = [action, val]
                # Pruning
                if val < alpha:
                    return [action, val]
                beta = min(beta, val)
            return bestAction

        return alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))[0]



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
        def expectimax(gameState, agent, depth):
            #Reset agent to pacman
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maximizer(gameState, agent, depth)
            else:
                return expEpectimax(gameState, agent, depth)

        def maximizer(gameState, agent, depth):
            bestAction = ["max", float("-inf")]
            pacmanActions = gameState.getLegalActions(agent)

            if not pacmanActions:
                return self.evaluationFunction(gameState)

            for action in pacmanActions:
                nextMove = gameState.generateSuccessor(agent, action)
                nextValue = expectimax(nextMove, agent+1, depth)
                if isinstance(nextValue,list):
                    val = nextValue[1]
                else:
                    val = nextValue
                if val > bestAction[1]:
                    bestAction = [action, val]
            return bestAction

        def expEpectimax(gameState, agent, depth):
            nextAction = ["exp", 0]
            ghostActions = gameState.getLegalActions(agent)

            if not ghostActions:
                return self.evaluationFunction(gameState)
            prob = float(1.0/len(ghostActions))

            for action in ghostActions:
                nextMove = gameState.generateSuccessor(agent, action)
                nextValue = expectimax(nextMove, agent+1, depth)
                if isinstance(nextValue,list):
                    val = nextValue[1]
                else:
                    val = nextValue
                nextAction[0] = action
                nextAction[1] += val*prob
            return nextAction

        return expectimax(gameState, 0, 0)[0]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
