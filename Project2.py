
# coding: utf-8

# In[1]:

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

#########################
def trig_exp_func(x, y):
#########################    
    #r = sqrt(x^2 + y^2)
    #function: (sin(x^2 + 3y^2)/(.1+r^2))  + (x^2 + 5y^2) * (exp(1-r^2) / 2)

    r = (math.sqrt((x)**2 + (y)**2))
    e = (math.exp(1 - (r)**2)) / 2
    t = math.sin(x**2 + (3 * y**2 ))
    z = ((t)/(.1 + r**2)) + (x**2 + (5 * y**2)) * (e)
    return z

######################
def create_array(A,B):
######################    
    return np.array([trig_exp_func(x,y) for x,y in zip(A,B)])

#################################################################################
def hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax):
#################################################################################    
    # First find the minimumvalue fo Z on X and Y cordinate
    
    x = np.arange(xmin, xmax, step_size)
    y = np.arange(ymin, ymax, step_size)
    X, Y = np.meshgrid(x,y)
    x_range = np.ravel(X)
    y_range = np.ravel(Y)
    array_set = create_array(x_range,y_range)
    Z = array_set.reshape(X.shape)
    xPt = random.uniform(xmin, xmax)
    yPt = random.uniform(ymin, ymax)
    
    current = function_to_optimize(xPt, yPt)
    xVal = [xPt]
    yVal = [yPt]
    zVal = [current]

    # if same zVal found 20 times then stop the loop
    sameValueCnt=0
    
    #check to see whether the points are between the minimum and maximum value
    while ( xPt <= xmax and xPt >= xmin and yPt <= ymax and yPt >= ymin  ):
        xIncr = function_to_optimize(xPt+step_size, yPt)
        xDecr = function_to_optimize(xPt-step_size, yPt)
        yIncr = function_to_optimize(xPt, yPt+step_size)
        yDecr = function_to_optimize(xPt, yPt-step_size)
        
        if (sameValueCnt == 20):
            break
  
        xVal.append(xPt)
        yVal.append(yPt)
        zVal.append(current)
        selectedPath = random.randint(1,4)

        if (xIncr < current and (xPt + step_size) <= xmax and selectedPath == 1 ):
            xPt = xPt + step_size
            sameValueCnt = 0
            current = xIncr
        elif (xDecr < current and (xPt - step_size) >= xmin and selectedPath == 2):
            xPt = xPt - step_size
            sameValueCnt = 0
            current = xDecr
        elif (yIncr < current and (yPt + step_size) <= ymax and selectedPath == 3):
            yPt = yPt + step_size
            sameValueCnt = 0
            current = yIncr
        elif (yDecr < current and (yPt - step_size) >= ymin and selectedPath == 4):
            yPt = yPt - step_size
            sameValueCnt = 0
            current = yDecr
        else:
            sameValueCnt += 1
    
    return xPt, yPt, current, xVal, yVal, zVal, X, Y, Z

######################################################################################################
def hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax):
######################################################################################################
    allCoordList = []
    restartCnt = 0
 
    while restartCnt < num_restarts:
        allCoordList.append(hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax))
        restartCnt += 1
    return min(allCoordList)

############################################################################################
def simulated_annealing(function_to_optimize, step_size, max_temp, xmin, xmax, ymin, ymax):
############################################################################################
    # Get starting position with max_temp
    xPt = random.uniform(xmin, xmax)
    yPt = random.uniform(ymin, ymax)

    # Get function value for starting postion 
    currFuncVal = function_to_optimize(xPt, yPt)
    temp = max_temp

    bestPos = currFuncVal

    for k in range(max_temp):
        temp = temp*.99

        # Get new position
        nxPt = random.uniform(xmin, xmax)
        nyPt = random.uniform(ymin, ymax)
        
        # Get function value for new postion 
        newFuncVal = function_to_optimize(nxPt, nyPt)

        probability = math.exp((newFuncVal - currFuncVal)/temp) 
 
        if bestPos > newFuncVal:
            bestPos = newFuncVal

    return bestPos
    
############
def main():
############
    hillClimb = time.clock()
    x,y,z, xVal, yVal, zVal, X, Y, Z = hill_climb(trig_exp_func, .1, -2.5, 2.5, -2.5, 2.5)
    print ("Hill Climbing for Minimum Result: (x,y,z)--->" , x,y,z)
    print ("Time taken for Hill Climb" , (time.clock() - hillClimb), "\n")
    
    #Create a graph for path taken to get minimum Z value on X and Y coordinate.
    graph = plt.figure()
    proj3d = graph.gca(projection = '3d')
    plt.plot(xVal, yVal, zVal, "w+")
    proj3d.plot_surface(X,Y,Z)
    plt.xlabel("X-Coords")
    plt.ylabel("Y-Coords")
    plt.show()

    hillClimbRestart = time.clock()
    x,y,z, xVal, yVal, zVal, X, Y, Z = hill_climb_random_restart(trig_exp_func, .1, 10, -2.5, 2.5, -2.5, 2.5)
    print ("Hill Climbing with Random Restart for Minimum Result: (x,y,z)--->" , x,y,z)
    print ("Time taken for Hill Climb with Random Restart" , (time.clock() - hillClimbRestart) , "\n")
    
    simulatedAnnealing = time.clock()
    z = simulated_annealing(trig_exp_func, .1, 50, -2.5, 2.5, -2.5, 2.5)
    print ("Simulated Annealing: (z) --->" , z)
    print ("Time taken for Simulated Annealing" , (time.clock() - simulatedAnnealing), "\n")
    
main()


# In[ ]:



