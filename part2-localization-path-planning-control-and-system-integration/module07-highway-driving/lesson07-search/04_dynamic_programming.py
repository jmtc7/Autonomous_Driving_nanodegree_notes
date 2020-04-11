# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def compute_value(grid, goal, cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------
    n_rows = len(grid)
    n_cols = len(grid[0])
    
    # Generate initial value matrix (full of 99s)
    value = [[99 for col in range(n_cols)] for row in range(n_rows)]
    
    
    # Iterate while we manage to do any change
    change = True
    while change:
        change = False
    
        # Go over the whole map
        for row in range(n_rows):
            for col in range(n_cols):
                # If we are in the goal, add a value of 0
                if row==goal[0] and col==goal[1]:
                    # Check if we have already changed it to avoid infinite loops
                    if value[row][col]>0:
                        value[row][col] = 0
                        change = True
                    
                # If it is not the goal and is not an obstacle
                elif grid[row][col] == 0:
                    # Perform all the posible movements
                    for i in range(len(delta)):
                        next_row = row + delta[i][0]
                        next_col = col + delta[i][1]
                        
                        # Check if the expansion it is a valid cell
                        ## Inside the grid and not an obstacle
                        if next_row>=0 and next_row<n_rows and next_col>=0 and next_col<n_cols and grid[next_row][next_col]==0:
                            # Compute its value
                            next_value = value[next_row][next_col] + cost
                            
                            # Check if the value
                            if next_value < value[row][col]:
                                value[row][col] = next_value
                                change = True
                                
    # Print resulting value function                            
    for row in value:
        print(row)
        
    return value 


# Call function
compute_value(grid, goal, cost)
