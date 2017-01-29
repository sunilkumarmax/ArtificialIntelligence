"""
This is Week2 Project 1 Assignment for AI eDX course
"""
import sys
import time
import queue
#import resource
from math import sqrt

class Queue():
    """A queue class"""

    def __init__(self):
        self.items = []

    def is_empty(self):
        """Is empty function"""
        return self.items == []

    def enqueue(self, item):
        """Enqueue function"""
        self.items.insert(0, item)

    def remove(self):
        """Dequeue function"""
        return self.items.pop()

    def size(self):
        """Size of the queue function"""
        return len(self.items)

    def check_board_exists(self, board_instance):
        """checks whether the game board exists in the queue"""
        for item in self.items:
            if item.is_equal(board_instance):
                return True
        return False

class Stack(Queue):
    """A Stack class"""

    def push(self, item):
        """Push function"""
        self.items.append(item)

class Board:
    """A game board class"""
    parent = None
    move = None
    level = None

    def __init__(self, board, board_length):
        self.unique_board = board
        self.length = board_length
        self.side_length = int(sqrt(self.length))

    def is_equal(self, other):
        """Checks whether two boards are equal"""
        return self.unique_board == other.unique_board

    def get_neighbours(self):
        """Fetches the neighbours of a state"""
        neighbours = []
        index = self.unique_board.split(',').index("0")
        row_index = int(index / self.side_length)
        col_index = index % self.side_length
        if row_index > 0: # Move Up
            neighbours.append(self.move_zero(1, index, index - self.side_length))
        if row_index < (self.side_length - 1): # Move Down
            neighbours.append(self.move_zero(2, index, index + self.side_length))
        if col_index > 0: # Move Left
            neighbours.append(self.move_zero(3, index, index - 1))
        if col_index < (self.side_length - 1): # Move Right
            neighbours.append(self.move_zero(4, index, index + 1))
        return neighbours

    def move_zero(self, move, zero_index, other_index):
        """Returns a new board state with a move"""
        board_list = self.unique_board.split(',')
        board_list[zero_index] = board_list[other_index]
        board_list[other_index] = '0'
        code = ",".join(board_list)
        neighbour = Board(code, self.length)
        neighbour.move = move
        return neighbour

    def is_in_dictionary(self, dictionary):
        """Checks whether the board is present in dictionary"""
        for dict_board in dictionary.values():
            if self.is_equal(dict_board):
                return True
        return False

    # def print_board(self):
    #     """Prints the game board"""
    #     for row_iterator in range(0, self.side_length):
    #         for col_iterator in range(0, self.side_length):
    #             print(str(self.board[(row_iterator * self.side_length) + col_iterator]) + " ", end="")
    #         print("")
    #     return Board(list(self.board))

class GameResult:
    """Game result class"""
    nodes_expanded = 0
    fringe_size = 0
    max_fringe_size = 0
    max_search_depth = 0
    running_time = 0
    max_ram_usage = 0

    def __init__(self, nodes, final_goal):
        self.nodes = nodes
        self.goal = final_goal
        self.result_dict = {}

    def print_result_dict(self):
        """Printing the result dictionary"""
        for key, value in self.result_dict.items():
            print(key, end="")
            print(": ", end="")
            print(value)

    def print_result(self):
        """Prints the game result in user friendly format"""
        path_to_goal = []
        node = self.goal
        while node.parent is not None:
            if node.move == 1:
                move = "Up"
            elif node.move == 2:
                move = "Down"
            elif node.move == 3:
                move = "Left"
            elif node.move == 4:
                move = "Right"
            path_to_goal.insert(0, move)
            node = self.nodes[node.parent]
        self.result_dict["path_to_goal"] = path_to_goal
        self.result_dict["cost_of_path"] = len(path_to_goal)
        self.result_dict["nodes_expanded"] = self.nodes_expanded
        self.result_dict["fringe_size"] = self.fringe_size
        self.result_dict["max_fringe_size"] = self.max_fringe_size
        self.result_dict["search_depth"] = self.goal.level
        self.result_dict["max_search_depth"] = self.max_search_depth
        self.result_dict["running_time"] = self.running_time
        self.result_dict["max_ram_usage"] = self.max_ram_usage
        self.print_result_dict()

def print_explored_dictionary(dictionary):
    """Printing the explored dictionary"""
    for value in dictionary.values():
        value.print_board()

def bfs_dfs(initial_board, final_goal, is_BFS):
    """BFS and DFS Algorithm"""
    time_start = time.time()
    check_time_counter = 0
    if is_BFS:
        frontier = Queue()
    else:
        frontier = Stack()
    explored = {}
    frontier_hashed = {}
    nodes_expanded = 0
    max_fringe_size = 0
    max_level = 0

    initial_board.level = 0
    if is_BFS:
        frontier.enqueue(initial_board)
    else:
        frontier.push(initial_board)
    frontier_hashed[initial_board.unique_board] = ""

    while not frontier.is_empty():
        fringe_size = frontier.size()
        #print(max_fringe_size)
        if fringe_size > max_fringe_size:
            max_fringe_size = fringe_size
        state = frontier.remove()
        del frontier_hashed[state.unique_board]
        explored[state.unique_board] = state # doing this so that I can backtrack the path later
        if state.is_equal(final_goal):
            result = GameResult(explored, state)
            result.nodes_expanded = nodes_expanded
            result.fringe_size = frontier.size()
            result.max_fringe_size = max_fringe_size
            result.max_search_depth = max_level
            result.running_time = round(time.time() - time_start, 8)
            result.max_ram_usage = 0 #round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024/ 1024, 8) # Converting into MB
            return result
        is_expanded = False
        neighbours = state.get_neighbours()
        if not is_BFS:
            neighbours = neighbours[::-1] #reverse
        for neighbour in neighbours:
            time_check_start = time.time()
            cond1 = frontier_hashed.get(neighbour.unique_board) is None
            cond2 = explored.get(neighbour.unique_board) is None
            check_time_counter += time.time() - time_check_start
            if cond1 and cond2:
                is_expanded = True
                new_level = state.level + 1
                neighbour.level = new_level
                if new_level > max_level:
                    max_level = new_level
                neighbour.parent = state.unique_board
                if is_BFS:
                    frontier.enqueue(neighbour)
                else:
                    frontier.push(neighbour)
                frontier_hashed[neighbour.unique_board] = ""
        if is_expanded:
            nodes_expanded += 1
    return None

def ast(initial_board, final_goal):
    """DFS Algorithm"""
    print(initial_board)

def ida(initial_board, final_goal):
    """IDA Algorithm"""
    print(initial_board)

arguments = sys.argv
#if len(arguments) < 3:
#    print("Usage driver.py <method> <board>")
#    exit(-2)

#method = arguments[1]
#input_board = arguments[2].split(',')
method = "dfs"
input_board = "5,3,2,7,4,0,6,1,8"
input_board_array = input_board.split(',')

for num in input_board_array:
    try:
        n = int(num)
    except ValueError:
        print("The text %s is not a number. Please renter the board" % num)
        exit(-2)

length = len(input_board_array)
side_length = sqrt(length)
if side_length**2 != length:
    print("The dimensions of the boards seems not right. Please enter the board such that the number of elements is a perfect square!!")
    exit(-2)

game_board = Board(input_board, length)

goal = []
for num in range(0, length):
    goal.append(str(num))
goal_state = Board(",".join(goal), length)

if method == "bfs" or method == "dfs":
    game_result = bfs_dfs(game_board, goal_state, method == "bfs")
    if game_result is None:
        print("Couldn't find the desired steps in " + method.upper() +" algorithm")
    else:
        game_result.print_result()
elif method == "ast":
    ast(game_board, goal_state)
elif method == "ida":
    ida(game_board, goal_state)
else:
    print("Incorrect syntax for method. Use one of the "/
          "following algorithms - <bfs>, <dfs>, <ast>, <ida>")
