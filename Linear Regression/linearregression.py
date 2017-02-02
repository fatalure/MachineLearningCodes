import numpy as np

#MSE (Mean Squared Error)
def compute_error_for_line_given_points(b, m, points):
    #initialize it at 0
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        #get the x value
        x = points[i, 0]
        #get the y value
        y = points[i, 1]
        #get the difference, square it, and add it to the total
        totalError += (y - ((m*x)+b)) ** 2
    
    #return the average
    return totalError/float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iteration):
    #starting b and m
    b = starting_b
    m = starting_m
    
    #gradient descent
    for i in range(num_iteration):  
        #update b and m with the new more accurate b and m by performing gradient step
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

#gradient descent
def step_gradient(b_current, m_current, points, learning_rate):
    #starting point for gradient
    b_gradient = 0
    m_gradient = 0
       
    n = float(len(points))
     
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direction with respect to b and m
        #computing partial derivatives for our error function
        b_gradient += -(2/n) * (y - ((m_current*x) + b_current))
        m_gradient += -(2/n) * x * (y - ((m_current*x) + b_current))
        
    #update our b and m values using partial derivatives

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def run():
    #Step 1 - collect the data
    points = np.genfromtxt('data.csv', delimiter =',')
    
    #Step 2 - define hyperparameters
    #how fast should our model converge?
    learning_rate = 0.0001
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    #how many iterations
    num_iteration = 1000

    #Step 3 - train the model
    print 'starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iteration)

	#Step 4 - Output the result 
    print 'ending point at b = {1}, m = {2}, error = {3} in {0} iterations'.format(num_iteration, b, m, compute_error_for_line_given_points(b, m, points))
    

if __name__ == '__main__':
    run()
