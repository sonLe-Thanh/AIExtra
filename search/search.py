# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    from util import Stack
    # (x,y),[path]
    stackPos = Stack()

    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    stackPos.push((problem.getStartState(), []))

    while True:
        if stackPos.isEmpty():
            return []

        # Get curr state
        pos, path = stackPos.pop()
        # print("pos")
        # print(pos)

        visited.append(pos)

        if problem.isGoalState(pos):
            return path

        # Get next move
        nextMove = problem.getSuccessors(pos)

        if nextMove:
            for item in nextMove:
                if item[0] not in visited:
                    newPath = path + [item[1]]
                    stackPos.push((item[0], newPath))
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queuePos = Queue()
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    queuePos.push((problem.getStartState(), []))
    while 1:
        if queuePos.isEmpty():
            return []

        pos, path = queuePos.pop()
        visited.append(pos)

        if problem.isGoalState(pos):
            return path

        nextMove = problem.getSuccessors(pos)

        if nextMove:
            for item in nextMove:
                if item[0] not in visited and \
                        item[0] not in (state[0] for state in queuePos.list):
                    newPath = path + [item[1]]
                    queuePos.push((item[0], newPath))

    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    queuePos = PriorityQueue()
    visited = []
    if problem.isGoalState(problem.getStartState()):
        return []

    queuePos.push((problem.getStartState(), []), 0)
    while True:
        if queuePos.isEmpty():
            return []

        pos, path = queuePos.pop()
        visited.append(pos)

        if problem.isGoalState(pos):
            return path

        nextMove = problem.getSuccessors(pos)

        if nextMove:
            for item in nextMove:
                item_list = [state[2][0] for state in queuePos.heap]
                oldPrice = 0
                if item[0] not in visited and \
                        item[0] not in item_list:
                    newPath = path + [item[1]]
                    price = problem.getCostOfActions(newPath)
                    queuePos.push((item[0], newPath), price)
                # Check for new price lower than old price
                elif item[0] not in visited and \
                        item[0] in item_list:
                    for state in queuePos.heap:
                        if state[2][0] == item[0]:
                            oldPrice = problem.getCostOfActions(state[2][1])

                    tempPath = path +[item[1]]
                    newPrice = problem.getCostOfActions(tempPath)
                    if oldPrice > newPrice:
                        newPath = tempPath
                        queuePos.update((item[0],newPath), newPrice)
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#f(n) = g(n) + h(n)
def f(problem, state, heuristic):
    return problem.getCostOfActions(state[1]) + heuristic(state[0], problem)

from util import PriorityQueue
class PriorityQueueWithFunction(PriorityQueue):
    def __init__(self, problem, function):
        self.function = function
        PriorityQueue.__init__(self)
        self.problem = problem

    def push(self, item, heuristic):
        PriorityQueue.push(self, item, self.function(self.problem, item, heuristic))

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queuePos = PriorityQueueWithFunction(problem, f)
    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    queuePos.push((problem.getStartState(),[]), heuristic)

    while True:
        if queuePos.isEmpty():
            return []

        pos, path = queuePos.pop()
        #State visited, pass since a lower cost found
        if pos in visited:
            continue

        visited.append(pos)

        if problem.isGoalState(pos):
            return path

        nextMove = problem.getSuccessors(pos)

        if nextMove:
            for item in nextMove:
                if item[0] not in visited:
                    newPath = path+[item[1]]
                    queuePos.push((item[0], newPath), heuristic)

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
