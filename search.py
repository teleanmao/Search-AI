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

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    mystack = util.Stack()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode)
    visited = set()
    while mystack :
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                mystack.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    myqueue= util.Queue()
    startNode = (problem.getStartState(), '', 0, [])
    myqueue.push(startNode)
    visited = set()
    while myqueue :
        node = myqueue.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                myqueue.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    #"""Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    myqueue= util.PriorityQueue()
    startstate = problem.getStartState()
    startNode = (startstate, [], 0)
    fstart= heuristic(startstate, problem)
    myqueue.push(startNode,fstart)
    # Push to the queue with priority.
    visited = set()
    actions = []

    bestg= {startstate:0}

    while not myqueue.isEmpty():    
        state, path, cost = myqueue.pop()
        g = problem.getCostOfActionSequence(path)
        if (state not in visited) or (g < bestg[state]):
        # duplicate detection and re-opening if better g.
            visited.add(state)
            bestg[state]= g
            if problem.isGoalState(state):
                actions = path
                break
            for child in problem.expand(state):
                succState, succAction, succCost = child
                newCost = cost + succCost
                newPath = path + [succAction]
                newNode = (succState, newPath, newCost)
                h = heuristic(succState, problem)
                if h < float('inf'):
                    myqueue.update(newNode, newCost + h)
                    #update the queue with priority f = g + h
    
    return actions



        
def recursivebfs(problem, heuristic=nullHeuristic) :
    #COMP90054 Task 2, Implement your Recursive Best First Search algorithm here
    "*** YOUR CODE HERE ***"
    startstate = problem.getStartState()
    startNode = (startstate, [], 0)
    actions, f_limit = rbfs(problem, startNode, float('inf'), heuristic)
    return actions
    
def rbfs(problem, node, f_limit, heuristic):
    # return a solution, or failure and a new f_limit.
    state, path, f = node
    if problem.isGoalState(state): 
        return path, 0
    successors = []
    children = problem.expand(state)

    if not children:
        return None, float('inf')
    for child in children:
        succState, succAction, succCost = child
        newPath = path + [succAction]
        succf = problem.getCostOfActionSequence(newPath) +  heuristic(succState, problem)	
        newf = max(succf, f)
        newNode = (succState, newPath, newf)
        successors.append(newNode)
    while(True):
        successors.sort(key=lambda x: x[2])
        best = successors[0]

        if best[2] > f_limit:
            return None, best[2]
        alternative = successors[1][2]
        result, newbestf =  rbfs(problem, best, min(f_limit, alternative), heuristic)
        #update the bestf
        beststate, bestpath, bestf = successors[0]
        successors[0] = (beststate, bestpath, newbestf)
        if result is not None:
            return result, 0





def aStarSearchpi2(problem, startState, goalState, heuristic):
# for Q4a, take startstate and goal state, return states and actions
    myqueue= util.PriorityQueue()
    #startNode =(startState, path, cost, states)
    startNode = (startState, [], 0,[])
    fstart= heuristic(startState, problem)
    myqueue.push(startNode,fstart)
    
    # Push to the queue with priority.
    visited = set()
    actions = []

    bestg= {startState:0}
    states= []

    while not myqueue.isEmpty():    
        state, path, cost, states = myqueue.pop()
        g = problem.getCostOfActionSequence(path)
        if (state not in visited) or (g < bestg[state]):
        # duplicate detection and re-opening if better g.
            visited.add(state)
            bestg[state]= g
            if state == goalState:
                actions = path
                returnstates = states
                break
            for child in problem.expand(state):
                succState, succAction, succCost = child
                newCost = cost + succCost
                newPath = path + [succAction]
                newstates = states + [state]
                newNode = (succState, newPath, newCost, newstates)
                h = heuristic(succState, problem)
                if h < float('inf'):
                    myqueue.update(newNode, newCost + h)
                    #update the queue with priority f = g + h
    
    return (returnstates, actions)








def aStarSearchpi3(problem, s, t, gr, gi, alpha, heuristic):
# for Q4b, take s, t, gr, gf and alpha, return actions 

    myqueue= util.PriorityQueue()
    #startNode =(startState, path, cost)
    startNode = (s, [], 0)

    # h(n,gr) 
    hngr = heuristic(s, gr)
    # h(n,gf) 
    hngi = heuristic(s, gi)
    # h(n,t) 
    hnt = heuristic(s, t)
    
    start = hnt
    if (hngr<hngi):
        start = alpha * hnt

    myqueue.push(startNode,start)
    
    # Push to the queue with priority.
    visited = set()
    actions = []
    bestg= {s:0}
    states= []

    while not myqueue.isEmpty():    
        state, path, cost = myqueue.pop()
        g = problem.getCostOfActionSequence(path)
        if (state not in visited) or (g < bestg[state]):
        # duplicate detection and re-opening if better g.
            visited.add(state)
            bestg[state]= g
            if state == t:
                actions = path
                
                break
            for child in problem.expand(state):
                succState, succAction, succCost = child
                newCost = cost + succCost
                newPath = path + [succAction]
                
                newNode = (succState, newPath, newCost)
                # h(n,gr)
                hngr = heuristic(succState, gr)
                # h(n,gf)
                hngi = heuristic(succState, gi)
                # h(n,t)
                hnt = heuristic(succState, t)
                if (hngr<hngi):
                    hnt = alpha * hnt

                if hnt < float('inf'):
                    myqueue.update(newNode, newCost + hnt)
                    #update the queue with priority f = g + h
    
    return actions
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
rebfs = recursivebfs
