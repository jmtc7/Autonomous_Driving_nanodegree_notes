# -----------
# User Instructions:
#
# Modify the the search function so that it returns
# a shortest path as follows:
# 
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left, 
# up, and down motions. Note that the 'v' should be 
# lowercase. '*' should mark the goal cell.
#
# You may assume that all test cases for this function
# will have a path from init to goal.
# ----------

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid, init, goal, cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    expand = None
    
    # List to store our result (cost, x, y)
    path = []
    
    # Get grid dimensions
    n_rows = len(grid[0])
    n_cols = len(grid)
    
    # Create grid saying which cells have been already visited
    visited = [[0 for row in range(n_rows)] for col in range(n_cols)]
    
    # Create expansion grid (initialized with -1) and a expansion counter
    expand = [[-1 for row in range(n_rows)] for col in range(n_cols)]
    n_expansions = 0
    
    # Create a grid to store how I arrived to each cell
    movements = [[-1 for row in range(n_rows)] for col in range(n_cols)]
    
    
    # Set initial cell as visited and as 0 expansions in the expansion grid
    visited[init[0]][init[1]] = 1
    expand[init[0]][init[1]] = 0
    
    # Register initial node in the list of cells to expand
    x = init[0]
    y = init[1]
    g = 0 # Current cost
    
    paths_to_expand = [[g, x, y]]
    
    # Flags
    success = False # If the goal is reached
    fail = False # If the goal is unreachable
    
    while success==False and fail==False:
        # If we ran out of cells to expand
        if len(paths_to_expand) == 0:
            fail=True
            path = [-1, -1, -1]
        else:
            # Expand the next cell with the smallest cost
            paths_to_expand.sort() # Sort according 1st element (cost)
            paths_to_expand.reverse() # Reverse to have the smallest cost at the end
            next_cell = paths_to_expand.pop() # Pop last element

            # Get information about next cell to expand
            g = next_cell[0]
            x = next_cell[1]
            y = next_cell[2]
            
            # BASE CASE. If we are in the goal, save the next cell
            if x==goal[0] and y==goal[1]:
                success = True
                path = next_cell
            else:
                # GENERAL CASE. Expand the cell
                # For each possible movement
                for move in range(len(delta)):
                    # Get new cell
                    new_x = x + delta[move][0]
                    new_y = y + delta[move][1]
                    
                    # Check if we will be inside the grid
                    if new_x>=0 and new_x<n_cols and new_y>=0 and new_y<n_rows:
                        # Check if the cell is visited and there is no obstacle in the map
                        if visited[new_x][new_y]==0 and grid[new_x][new_y]==0:
                            # Update cost and append to expansion list
                            new_g = g + cost
                            paths_to_expand.append([new_g, new_x, new_y])
                            
                            # Mark cell as visited
                            visited[new_x][new_y] = 1
                            
                            # Store the movement done to get here
                            movements[new_x][new_y] = move
                            
                            # Update expansion counter and expansion grid
                            n_expansions += 1
                            expand[new_x][new_y] = n_expansions
    
    
    
    # Use the stored movements to draw the used path
    # Create grid saying which movements have been done
    movements_map = [[' ' for row in range(n_rows)] for col in range(n_cols)]
    
    # Draw goal
    x = goal[0]
    y = goal[1]
    movements_map[x][y] = '*'
    
    # Draw the path advancing from the goal (x,y) to the beginnnig (init)
    ## Do-While loop emulation. See "if" at the end for condition
    while True:
        # Read movement
        move = movements[x][y]
        
        # Get previous cell
        past_x = x - delta[move][0]
        past_y = y - delta[move][1]
        
        # Print symbol
        movements_map[past_x][past_y] = delta_name[move]
        
        # Update current cell to process (with the one in which we were before it)
        x = past_x
        y = past_y
        
        # Break when we reach the origin
        if x==init[0] and y==init[1]:
            break
    
    for row in movements_map:
        print(row)
        
    return movements_map

search(grid, init, goal, cost)
