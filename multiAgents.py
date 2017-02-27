# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util,sys

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        foodList = oldFood.asList()
        ghostPosition = currentGameState.getGhostPosition(1)
        closestFoodDist = 1000
        foodLeft = len(foodList)

        #Get more points if close to a food
        for food in foodList:
            dist = util.manhattanDistance(newPos, food)
            if dist < closestFoodDist:
                closestFoodDist = dist
        score = -2 * closestFoodDist

        #Get moore points if far away from a ghost
        distFromGhost = util.manhattanDistance(ghostPosition, newPos)
        score += (distFromGhost)

        #Get more points for the number of food left to be eaten
        if newPos in foodList:
            foodLeft -= 1
        score - (3 * foodLeft)

        #Penalizes times when close to a ghost
        if distFromGhost <= 2:
            score -= 20

        return score


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

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def maxVal(state,depth):
            d = self.depth
            if state.isWin() or state.isLose() or depth == d:
                return (self.evaluationFunction(state))
            depth += 1
            legalMoves = [action for action in state.getLegalActions(0) if action != 'Stop']
            maxScore = -sys.maxint
            bestAction = ''
            for action in legalMoves:
                newState = state.generateSuccessor(0,action)
                prevScore = maxScore
                maxScore = max(maxScore,minVal(newState, depth, 1))
                if maxScore > prevScore:
                    bestAction = action
            return (maxScore,bestAction)


        def minVal(state,depth,adversary):
            d = self.depth
            if state.isWin() or state.isLose() or depth == d:
                return(self.evaluationFunction(state))
            depth += 1
            legalMoves = state.getLegalActions(adversary)
            maxScore = sys.maxint
            for action in legalMoves:
                newState = state.generateSuccessor(adversary,action)
                if adversary == state.getNumAgents() - 1:
                    maxScore = min(maxScore,maxVal(newState,depth))
                else:
                    maxScore = min(maxScore,minVal(newState, depth,adversary + 1))
            return maxScore

        move = maxVal(gameState,0)
        return move[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def abMax(state,depth,alpha,beta):
            d = self.depth
            if state.isWin() or state.isLose() or depth == d:
                return (self.evaluationFunction(state))
            depth += 1
            legalMoves = [action for action in state.getLegalActions(0) if action != 'Stop']
            maxScore = -sys.maxint
            for action in legalMoves:
                newState = state.generateSuccessor(0,action)
                maxScore = max(maxScore,abMin(newState, depth, 1,alpha,beta))
                if maxScore >= beta:
                    return maxScore
                alpha = max(alpha, maxScore)
            return maxScore


        def abMin(state,depth,adversary,alpha,beta):
            d = self.depth
            if state.isWin() or state.isLose() or depth == d:
                return(self.evaluationFunction(state))
            depth += 1
            legalMoves = state.getLegalActions(adversary)
            maxScore = sys.maxint
            for action in legalMoves:
                newState = state.generateSuccessor(adversary,action)
                if (adversary == state.getNumAgents() - 1):
                    maxScore = min(maxScore,abMax(newState,depth,alpha,beta))
                else:
                    maxScore = min(maxScore,abMin(newState, depth, adversary + 1,alpha,beta))
                if maxScore <= alpha:
                    return maxScore
                beta = min(beta,maxScore)
            return maxScore

        actions = [action for action in gameState.getLegalActions(0) if action != 'Stop']
        alpha = -sys.maxint
        beta = sys.maxint
        maximum = -sys.maxint
        maxAction = ''
        for action in actions:
            Depth = 0
            currentMax = abMin(gameState.generateSuccessor(0, action), Depth, 1,alpha, beta)
            if currentMax > maximum:
                maximum = currentMax
                maxAction = action
        return maxAction

class AgentState():
    def __init__(self,ghostState,agent):
        self.Agent = agent
        self.Position = ghostState.configuration.pos
        self.Timer = ghostState.scaredTimer

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
        def maxValue(state,depth):
            d = self.depth
            if state.isWin() or state.isLose() or depth == d:
                return (self.evaluationFunction(state))
            depth += 1
            legalMoves = [action for action in state.getLegalActions(0) if action != 'Stop']
            maxScore = -sys.maxint
            for action in legalMoves:
                newState = state.generateSuccessor(0, action)
                maxScore = max(maxScore, expValue(newState, depth, 1))
            return (maxScore)


        def expValue(state,depth,adversary):
            d = self.depth
            if state.isWin() or state.isLose() or depth == d:
                return(self.evaluationFunction(state))
            depth += 1
            legalMoves = state.getLegalActions(adversary)
            value = 0
            for action in legalMoves:
                newState = state.generateSuccessor(adversary,action)
                if adversary == state.getNumAgents() - 1:
                    value += maxValue(newState, depth - 1)
                else:
                    value += expValue(newState,depth,adversary+1)
            return value/len(legalMoves)


        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions = [action for action in gameState.getLegalActions(0) if action != 'Stop']
        bestAction = ''
        score = -sys.maxint
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevscore = score
            score = max(score, expValue(nextState, 1, 1))
            if score > prevscore:
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      This evaluation function uses the food left to be eaten, the location (and absence of) power pellets, pacman's
      position, and the state of the ghosts to return a value to "score" the given state.

      Each section of the evaluation function has comments for more clarification.

      The score is mainly based on the amount of food that is left. Eating food in a state is given a lot of points
      (to help with the "thrashing" that can occur). Avoiding the ghosts is given a very high priority. The points
      try to keep pacman at least 1 space away if possible (high penalty for being one space away and max
      penality for 0 spaces away).
      Dealing with the power pellet involves a boolean (whether or not it is possible to eat a ghost in this state) and
      keeps track of the ghost's scared-timers to check if a nearby ghost can be eaten.
    """

    #The necessary information about the current state needed for my evaluation function
    foodList = currentGameState.getFood().asList()
    foodLeft = currentGameState.getNumFood()
    capsules = currentGameState.getCapsules()
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = {}
    for ghost in range(1,currentGameState.getNumAgents()):
        ghostStates[ghost] = AgentState(currentGameState.getGhostState(ghost),ghost)

    # This finds the closest ghost and keeps track of which agent it is and the estimated distance from pacman
    ghostDistance = sys.maxint
    closestGhost = None
    for ghost in ghostStates.values():
        distance = manhattanDistance(ghost.Position, pacmanPosition)
        if distance < ghostDistance:
            ghostDistance = distance
            closestGhost = ghost

    #This checks whether or not a ghost can get eaten (pacman has eaten a power pellet)
    for ghost in ghostStates.values():
        if ghost.Timer != 0:
            powerPelletTime = True
        else:
            powerPelletTime = False

    #If the game is win or lost, give this priority accordingly
    if currentGameState.isWin():
        return sys.maxint
    if currentGameState.isLose():
        return -sys.maxint + 1

    #This sets the baseline score based on how much food is left and how close pacman is to the
    #nearest food
    if foodLeft != 0:
        currentClosestFood = min([manhattanDistance(pacmanPosition,food) for food in foodList])
        score = abs(int(((54.0 - foodLeft)/54.0)*1000) - currentClosestFood)
    else:
        score = abs(int(((54.0 - foodLeft) / 54.0) * 1000))

    for capsule in capsules:
        if pacmanPosition == capsule:
            score += 50

    # This makes sure pacman doesn't run into a ghost that cant be eaten
    #If it is a ghost that can be eaten though, this is given high priority
    if powerPelletTime:
        if ghostDistance == 0 and closestGhost.Timer != 0:
            score += 200
        else:
            score += (1/ghostDistance)*190
    else:
        score += ghostDistance * 2
        if ghostDistance < 2:
            score -= ghostDistance * 2


    return score


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
