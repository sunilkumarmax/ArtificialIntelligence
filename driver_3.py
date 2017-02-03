"""
This is Week2 Project 1 Assignment for AI eDX course
"""
import copy
from math import sqrt
import resource
import sys
import time
import traceback

class IndexLevelHolder():
    """Acts as a class to store the index and level"""
    def __init__(self, index, level):
        self.index = index
        self.level = level

class HashHelper():
    """Helps to index the game board"""
    def __init__(self):
        self.hashed_set = {}

    def hash_item(self, key, value):
        """Hash the item in the dictionary"""
        self.hashed_set[key] = value

    def remove_hash_item(self, key):
        """Removing the hashed item in the dictionary"""
        del self.hashed_set[key]

    def get_item(self, key):
        """Retrieves the item from the hashed set"""
        return self.hashed_set.get(key)

    def get_level(self, key): # Value is IndexLevelHolder object
        """Retrieves the level from the IndexLevel Holder object"""
        obj = self.hashed_set.get(key)
        if obj is None:
            return None
        return obj.level

    def get_index(self, key): # Value is IndexLevelHolder object
        """Retrieves the index from the IndexLevel Holder object"""
        obj = self.hashed_set.get(key)
        if obj is None:
            return None
        return obj.index

class Queue(HashHelper):
    """A queue class"""

    def __init__(self):
        self.items = []
        super(Queue, self).__init__()

    def is_empty(self):
        """Is empty function"""
        return self.items == []

    def insert(self, item):
        """Enqueue function"""
        self.items.insert(0, item)
        return 0

    def remove(self):
        """Dequeue function"""
        return self.items.pop()

    def size(self):
        """Size of the queue class"""
        return len(self.items)

class Stack(Queue):
    """A Stack class"""

    def insert(self, item):
        """Push function"""
        self.items.append(item)
        return len(self.items) - 1

class Heap(HashHelper):
    """A generic Heap class"""
    is_less = None

    def __init__(self, is_min):
        """Init function for the heap class"""
        self.is_min = is_min
        self.items = []
        super(Heap, self).__init__()

    def size(self):
        """Size of the heap"""
        return len(self.items)

    def is_empty(self):
        """Is empty function"""
        return self.items == []

    def rehash_boards(self, index, next_index):
        """Rehashes the indexes in the dictionary"""
        main_board = self.items[next_index] ##Rehashing
        next_board = self.items[index]
        super(Heap, self).hash_item(main_board.unique_board
                                    , IndexLevelHolder(next_index, main_board.level))
        super(Heap, self).hash_item(next_board.unique_board
                                    , IndexLevelHolder(index, next_board.level))

    def swap_boards(self, index, next_index):
        """Swaps the positions of the board in the heap"""
        temp = self.items[index]
        self.items[index] = self.items[next_index]
        self.items[next_index] = temp

    def percolate_up(self, index):
        """Percolate up function in Heap"""
        next_index = int((index - 1)/ 2)
        while index > 0:
            if self.is_min:
                condition = self.is_less(self.items[index], self.items[next_index])
            else:
                condition = self.is_less(self.items[next_index], self.items[index])
            if condition:
                self.swap_boards(index, next_index)
                self.rehash_boards(index, next_index)
                if next_index == 0:
                    return next_index
                index = next_index
                next_index = int((index - 1)/ 2)
            else:
                return index
        return index

    def percolate_down(self):
        """Percolate up function in Heap"""
        index = 0
        next_index = (index * 2) + 1
        while next_index < self.size():
            if self.is_min:
                condition = self.is_less(self.items[next_index], self.items[index])
                if not condition and (next_index + 1) < self.size():
                    condition = self.is_less(self.items[next_index + 1], self.items[index])
                    next_index += 1
            else:
                condition = self.is_less(self.items[index], self.items[next_index])
                if not condition and (next_index + 1) < self.size():
                    condition = self.is_less(self.items[index], self.items[next_index + 1])
                    next_index += 1
            if condition:
                self.swap_boards(index, next_index)
                self.rehash_boards(index, next_index)
                index = next_index
                next_index = (index * 2) + 1
            else:
                return index
        return index

    def insert(self, item):
        """Inserts an item in the heap"""
        self.items.append(item)
        super(Heap, self).hash_item(item.unique_board
                                    , IndexLevelHolder(self.size() - 1, item.level))
        return self.percolate_up(self.size() - 1)

    def remove(self):
        """Removes an item from the heap"""
        item = self.items.pop(0)
        if self.size() > 0:
            last_item = self.items.pop()
            self.items.insert(0, last_item)
            super(Heap, self).hash_item(last_item.unique_board
                                        , IndexLevelHolder(0, last_item.level))
            self.percolate_down()
        return item

class Board:
    """A game board class"""
    parent = None
    move = None
    level = None

    def __init__(self, board, board_length, is_heuristic):
        self.zero_index = board.split(',').index("0")
        self.unique_board = board
        self.length = board_length
        self.side_length = int(sqrt(self.length))
        self.is_heuristic = is_heuristic
        if self.is_heuristic:
            self.manhattan_distance = self.calculate_manhattan_distance()

    def is_equal(self, other):
        """Checks whether two boards are equal"""
        return self.unique_board == other.unique_board

    def compare_cost(self, other, frontier):
        """Checks whether manhattan distance of current board is less than other board"""
        return self.calculate_cost(frontier) < other.calculate_cost(frontier)

    def calculate_cost(self, frontier):
        return self.manhattan_distance + frontier.get_level(self.unique_board)

    def get_neighbours(self):
        """Fetches the neighbours of a state"""
        neighbours = []
        row_index = int(self.zero_index / self.side_length)
        col_index = self.zero_index % self.side_length
        if row_index > 0: # Move Up
            neighbours.append(self.move_zero(1, self.zero_index
                                             , self.zero_index - self.side_length))
        if row_index < (self.side_length - 1): # Move Down
            neighbours.append(self.move_zero(2, self.zero_index
                                             , self.zero_index + self.side_length))
        if col_index > 0: # Move Left
            neighbours.append(self.move_zero(3, self.zero_index, self.zero_index - 1))
        if col_index < (self.side_length - 1): # Move Right
            neighbours.append(self.move_zero(4, self.zero_index, self.zero_index + 1))
        return neighbours

    def move_zero(self, move, zero_index, other_index):
        """Returns a new board state with a move"""
        board_list = self.unique_board.split(',')
        board_list[zero_index] = board_list[other_index]
        board_list[other_index] = '0'
        code = ",".join(board_list)
        neighbour = Board(code, self.length, self.is_heuristic)
        neighbour.move = move
        neighbour.parent = self.unique_board
        neighbour.level = self.level + 1
        return neighbour

    def is_in_dictionary(self, dictionary):
        """Checks whether the board is present in dictionary"""
        for dict_board in dictionary.values():
            if self.is_equal(dict_board):
                return True
        return False

    def calculate_manhattan_distance(self):
        """Returns the manhattan distance of the current board position"""
        distance = 0
        board = self.unique_board.split(',')
        for iterator in range(0, self.length):
            num = int(board[iterator])
            if num == 0:
                continue
            source_row = int(iterator / 3)
            source_col = iterator % 3
            dest_row = int(num / 3)
            dest_col = num % 3
            distance += abs(source_row - dest_row) + abs(source_col - dest_col)
        return distance

    def print_board(self):
        """Prints the game board"""
        board_list = self.unique_board.split(',')
        for row_iterator in range(0, self.side_length):
            for col_iterator in range(0, self.side_length):
                print(str(board_list[(row_iterator * self.side_length) + col_iterator]) + " "
                      , end="")
            print("")

class GameResult:
    """Game result class"""
    nodes_expanded = 0
    fringe_size = 0
    max_fringe_size = 0
    max_search_depth = 0
    running_time = 0
    max_ram_usage = 0
    next_threshold = 0

    def __init__(self, nodes, final_goal):
        self.nodes = nodes
        self.goal = final_goal
        self.result_dict = {}
        self.result_found = False

    def print_result_dict(self, filename):
        """Printing the result dictionary"""
        with open(filename, 'w') as output_file:
            for key, value in self.result_dict.items():
                output_file.write(key + ": " + str(value) + "\n")
        if not output_file.closed:
            output_file.close()

    def print_result(self, filename):
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
        self.print_result_dict(filename)

class Helper:
    """Class for the helper functions"""
    def print_explored_dictionary(self, dictionary):
        """Printing the explored dictionary"""
        for value in dictionary.values():
            value.print_board()

class PathSolver:
    """ Path solver class for solving the puzzle"""
    def __init__(self, initial_board, final_goal):
        """Init function for the class"""
        self.initial_board = initial_board
        self.final_goal = final_goal

    def solve_puzzle(self, method, threshold):
        """BFS and DFS Algorithm"""
        time_start = time.time()
        if method == "bfs":
            frontier = Queue()
        elif method == "dfs":
            frontier = Stack()
        elif method == "astar":
            frontier = Heap(True)
            frontier.is_less = lambda board_a, board_b: board_a.compare_cost(board_b, frontier)
        else:
            return None
        explored = {}
        nodes_expanded = 0
        max_fringe_size = 0
        max_level = 0
        next_threshold = sys.maxsize

        self.initial_board.level = 0
        frontier.insert(self.initial_board)
        frontier.hash_item(self.initial_board.unique_board
                           , IndexLevelHolder(0, self.initial_board.level))

        while not frontier.is_empty():
            fringe_size = frontier.size()
            if fringe_size > max_fringe_size:
                max_fringe_size = fringe_size
            state = frontier.remove()
            frontier.remove_hash_item(state.unique_board)
            explored[state.unique_board] = state # doing this so that I can backtrack the path later
            if state.is_equal(self.final_goal):
                result = GameResult(explored, state)
                result.nodes_expanded = nodes_expanded
                result.fringe_size = frontier.size()
                result.max_fringe_size = max_fringe_size
                result.max_search_depth = max_level
                result.running_time = round(time.time() - time_start, 8)
                ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                result.max_ram_usage = round(ram_usage / 1024/ 1024, 8) # Converting into MB
                result.result_found = True
                return result
            neighbours = state.get_neighbours()
            if method == "dfs":
                neighbours = neighbours[::-1] #reverse
            nodes_expanded += 1
            for neighbour in neighbours:
                new_level = neighbour.level
                if method == "astar":
                    cost = neighbour.manhattan_distance + new_level
                    if cost > threshold:
                        if cost < next_threshold:
                            next_threshold = cost
                        continue
                cond1 = frontier.get_item(neighbour.unique_board) is None
                cond2 = explored.get(neighbour.unique_board) is None
                if cond1 and cond2:
                    if new_level > max_level:
                        max_level = new_level
                    index_frontier = frontier.insert(neighbour)
                    frontier.hash_item(neighbour.unique_board
                                       , IndexLevelHolder(index_frontier, neighbour.level))
                elif method == "astar" and not cond1:
                    index_level = frontier.get_item(neighbour.unique_board)
                    if new_level < index_level.level:
                        frontier.hash_item(neighbour.unique_board
                                           , IndexLevelHolder(
                                               frontier.percolate_up(index_level.index)
                                               , new_level))
        if method == "astar":
            game_result = GameResult(explored, self.final_goal)
            game_result.next_threshold = next_threshold
            return game_result
        return None

    def bfs(self):
        """Implements the bfs algorithm"""
        return self.solve_puzzle("bfs", None)

    def dfs(self):
        """Implements the dfs algorithm"""
        return self.solve_puzzle("dfs", None)

    def ast(self):
        """DFS Algorithm"""
        game_result = self.solve_puzzle("astar", sys.maxsize)
        if game_result.result_found:
            return game_result
        return None

    def ida(self):
        """IDA Algorithm"""
        threshold = 1
        prev_threshold = 0
        while threshold < sys.maxsize:
            game_result = self.solve_puzzle("astar", threshold)
            if game_result.result_found:
                return game_result
            else:
                prev_threshold = threshold
                threshold = game_result.next_threshold
                if prev_threshold == threshold:
                    return None

def main():
    """Main entry function"""
    arguments = sys.argv
    if len(arguments) < 3:
        print("Usage driver.py <method> <board>")
        exit(-2)

    method = arguments[1]
    input_board = arguments[2]
    input_board_array = input_board.split(',')
    is_heuristic = False
    trace = None

    for num in input_board_array:
        try:
            int(num)
        except ValueError:
            print("The text %s is not a number. Please renter the board" % num)
            exit(-2)

    length = len(input_board_array)
    side_length = sqrt(length)
    if side_length**2 != length:
        print("The dimensions of the boards seems not right. Please enter the " +
              "board such that the number of elements is a perfect square!!")
        exit(-2)
    if method == "ast" or method == "ida":
        is_heuristic = True
    game_result = None

    try:
        game_board = Board(input_board, length, is_heuristic)

        goal = []
        for num in range(0, length):
            goal.append(str(num))
        goal_state = Board(",".join(goal), length, is_heuristic)

        algorithm = PathSolver(game_board, goal_state)
        if method == "bfs":
            game_result = algorithm.bfs()
        elif method == "dfs":
            game_result = algorithm.dfs()
        elif method == "ast":
            game_result = algorithm.ast()
        elif method == "ida":
            game_result = algorithm.ida()
        else:
            print("Incorrect syntax for method. Use one of the "/
                  "following algorithms - <bfs>, <dfs>, <ast>, <ida>")
    except TypeError:
        trace = traceback.format_exc()
    else:
        trace = None
    finally:
        if trace is not None:
            #print(trace)   #Remove before commit
            pass

    if trace is not None:
        print("An error occured. See the trace to find out the issue!")
    elif game_result is None:
        print("Couldn't find the desired steps in " + method.upper() +" algorithm")
    else:
        game_result.print_result("output.txt")

if __name__ == "__main__":
    sys.exit(main())
