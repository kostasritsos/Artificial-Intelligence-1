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

from os import curdir
from typing import Dict, Optional
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
    frontier=util.Stack()
    node={'place':problem.getStartState(),'cost':0}
    frontier.push(node)
    expanded=set()
    path=[]
    while not frontier.isEmpty():
        node=frontier.pop()
        if problem.isGoalState(node['place']):
            while 'parent' in node:
                path.insert(0,node['action'])
                node=node['parent']
            return path
        if node['place'] not in expanded:
            expanded.add(node['place'])
            nodess_children=problem.expand(node['place'])
            for child,action,stepCost in nodess_children:
                Child={'place':child,'action':action,'cost':stepCost,'parent':node}
                if Child['place'] not in expanded:
                    frontier.push(Child)
    return []

                

    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier=util.Queue()
    node={'place':problem.getStartState(),'cost':0}
    frontier.push(node)
    expanded=set()
    path=[]
    while not frontier.isEmpty():
        node=frontier.pop()
        if problem.isGoalState(node['place']):
            while 'parent' in node:
               path.insert(0,node['action'])
               node=node['parent']
            return path
        if node['place'] not in expanded:
            expanded.add(node['place'])
            nodess_children=problem.expand(node['place'])
            for child,action,stepCost in nodess_children:
                Child={'place':child,'action':action,'cost':stepCost,'parent':node}
                if Child['place'] not in expanded:
                    frontier.push(Child)
    return []
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier=util.PriorityQueue()
    node={'place':problem.getStartState(),'cost':0,'cost to go':0,'action':None}
    frontier.push(node,0)
    expanded=set()
    path=[]
    while not frontier.isEmpty():
        node=frontier.pop()
        if problem.isGoalState(node['place']):
            while 'parent' in node :
                path.insert(0,node['action'])
                node=node['parent']
            return path
        if node['place'] not in expanded:
            expanded.add(node['place'])
            nodess_children=problem.expand(node['place'])
            for child,action,stepCost in nodess_children:
                if child not in expanded:
                    Child={'place':child,'action':action,'cost':stepCost,'parent':node,'cost to go':node['cost to go']+stepCost}
                    g_n=node['cost to go']+stepCost
                    f_n=g_n+heuristic(Child['place'],problem)
                    if f_n <Child['cost to go'] or Child['cost to go']==0:
                        Child['cost to go']=f_n
                    frontier.push(Child,f_n)
    return []
    
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch