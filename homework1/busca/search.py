# search.py
# Lucas Stefan Abe 8531612
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    """print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    state = problem.getStartState()
    visit = {}
    actions = []
    visit[state] = True
    recursive_dfs(state, visit, actions, problem)
    return actions

def recursive_dfs(state, visit, actions, problem):
	if problem.isGoalState(state):
		return True 
	for sucessor in problem.getSuccessors(state):
		next_state, action = sucessor[0], sucessor[1]
		if next_state not in visit:
			visit[next_state] = True
			actions.append(action)
			if recursive_dfs(next_state, visit, actions, problem):
				return True
			actions.pop()
	return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    state = problem.getStartState()
    visit = {}
    queue = Queue()
    actions = {}
    parent = {}
    visit[state] = True
    parent[state] = state
    queue.push(state)
    while not queue.isEmpty():
    	state = queue.pop()
    	if problem.isGoalState(state):
    		break
    	for sucessor in problem.getSuccessors(state):
    		next_state, action = sucessor[0], sucessor[1]
    		if next_state not in visit:
    			visit[next_state] = True
    			parent[next_state] = state
    			actions[next_state] = action
    			queue.push(next_state)
    ans = list()
    while parent[state] != state:
    	ans.insert(0, actions[state])
    	state = parent[state]
    return ans

def iterativeDeepeningSearch(problem):
    """
    Start with depth = 0.
    Search the deepest nodes in the search tree first up to a given depth.
    If solution not found, increment depth limit and start over.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    state = problem.getStartState()
    cost = {}
    actions = []
    cost[state] = 0
    limit = 0
    ans = idsRec(state, cost, actions, problem, limit, 0)
    limit += 1
    while not ans:
	    state = problem.getStartState()
	    cost = {}
	    actions = []
	    cost[state] = 0
	    limit += 1
	    ans = idsRec(state, cost, actions, problem, limit, 0)
    return actions

def idsRec(state, cost, actions, problem, limit, depth):
	if depth > limit:
		return False
	if problem.isGoalState(state):
		return True
	for sucessor in problem.getSuccessors(state):
		next_state, action, step_cost = sucessor[0], sucessor[1], sucessor[2]
		if (next_state not in cost) or (cost[next_state] > cost[state] + step_cost):
			cost[next_state] = cost[state] + step_cost
			actions.append(action)
			if idsRec(next_state, cost, actions, problem, limit, depth+1):
				return True
			actions.pop()
	return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue 
    g = {}
    actions = {}
    parent = {}
    pq = PriorityQueue()
    state = problem.getStartState()
    g[state] = 0
    parent[state] = state
    pq.push(state, heuristic(state, problem))

    while not pq.isEmpty():
    	state = pq.pop()
    	if problem.isGoalState(state):
    		break
    	for sucessor in problem.getSuccessors(state):
    		next_state = sucessor[0]
    		if (next_state not in g) or (g[next_state] > g[state] + sucessor[2]):
				parent[next_state] = state
				actions[next_state] = sucessor[1]
				g[next_state] = g[state]+sucessor[2]
				if next_state not in g:
					pq.push(next_state, heuristic(next_state, problem) + g[next_state])
				else:
					pq.update(next_state, heuristic(next_state, problem) + g[next_state])

    ans = list()
    while parent[state] != state:
    	ans.insert(0, actions[state])
    	state = parent[state]
    return ans
	


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch