"""Importing required libraries"""
import sys
#from random import randint
from BaseAI_3 import BaseAI

vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)
########TODO: move(), getAvailableCells(), insertTile(), and clone()  --> improve

class PlayerAI(BaseAI):
    """Player AI"""

    def getMove(self, grid):
        #result = self.maximize(grid, -1 * sys.maxsize, sys.maxsize)
        # moves = grid.getAvailableMoves()
        # dummy_result = moves[randint(0, len(moves) - 1)] if moves else None
        print(self.get_possible_cells(grid))
        return 0 #result[0]

    def is_terminal_state(self, grid): ##TODO: Use heuristic functions
        """Determine whether this is a terminal state or not""" ##TODO: Assign heuristic weights (more than one heuristic function and assign weights)

########TODO: the absolute value of tiles
########TODO: the difference in value between adjacent tiles
########TODO: the potential for merging of similar tiles
########TODO: the ordering of tiles across rows, columns, and diagonals
        pass #TODO: Implementation pending

    def get_possible_cells(self, grid):
        """Gets the possible cells"""
        #H1. Always assumes the grid has number on walls
        side = grid.size
        result_dict = {}
        for x_iter in range(0, side):
            for y_iter in range(0, side):
                if grid.map[x_iter][y_iter] > 0:
                    if x_iter > 0 and grid.map[x_iter-1][y_iter] == 0: #Can look up
                        result_dict[((x_iter-1) * side) + y_iter] = 0
                        # mini_count = 0        #TODO: Apply weights
                        # while grid.map[x_iter-1 - minicount][y_iter] == 0 and (x_iter-1 - minicount) >= 0:
                        #     mini_count += 1
                        # if result_dict[((x_iter-1) * side) + y_iter] is None:
                        #     result_dict[((x_iter-1) * side) + y_iter] = mini_count
                    if x_iter < side - 1 and grid.map[x_iter+1][y_iter] == 0: #can look down
                        result_dict[((x_iter+1) * side) + y_iter] = 0
                    if y_iter > 0 and grid.map[x_iter][y_iter-1] == 0:  #can look left
                        result_dict[(x_iter * side) + y_iter - 1] = 0
                    if y_iter < side - 1 and grid.map[x_iter][y_iter-1] == 0: #can look right
                        result_dict[(x_iter * side) + y_iter + 1] = 0
        return result_dict

    def get_available_grid_moves(self, grid, dirs=vecIndex):
        """Returns the available moves for a grid state"""
        available_moves = {}
        for move in dirs:
            grid_copy = grid.clone()
            if grid_copy.move(move):
                available_moves[move] = grid_copy
        return available_moves


    def maximize(self, grid, alpha, beta):
        """Maximizes the utility function"""
        if self.is_terminal_state(grid):
            return (None, 0) #TODO: Return the correct value #<Null,Eval(state)>
        max_child = None
        max_utility = -1 * sys.maxsize
        for key, value in self.get_available_grid_moves(grid).items():
            temp = self.minimize(value, alpha, beta)
            if temp[1] > max_utility:
                max_child = key
                max_utility = temp[1]
            if max_utility >= beta:
                break
            if max_utility > alpha:
                alpha = max_utility
        return (max_child, max_utility)


    def minimize(self, grid, alpha, beta):
        """Minimizes the utility function"""
        if self.is_terminal_state(grid):
            return (None, 0) #TODO: Return the correct value #<Null,Eval(state)>
        min_child = None
        min_utility = sys.maxsize
        for key, value in self.get_available_grid_moves(grid).items():
            temp = self.maximize(value, alpha, beta)
            if temp[1] < min_utility:
                min_child = key
                min_utility = temp[1]
            if min_utility <= alpha:
                break
            if min_utility < beta:
                beta = min_utility
        return (min_child, min_utility)
