# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    # Calculate min, max and mean for normialization.
    meanX = np.mean(X, axis = 0)
    minX = np.min(X, axis = 0)
    maxX = np.max(X, axis = 0)
    
    meanY = np.mean(y, axis = 0)
    minY = np.min(y, axis = 0)
    maxY = np.max(y, axis = 0)

    #Normailzation of labels and all features.
    X = (X - meanX)/(maxX - minX)
    y = (y - meanY)/(maxY - minY)
    
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # If X is 1D reshape into 2D.
    if X.ndim == 1:  # If X is a 1D array
        X = X.reshape(-1, 1)
    
    # Insert column of ones to the 0th column.
    X = np.insert(X, 0, 1, axis=1)

    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    m = len(X)
    hyp = np.dot(X, theta)
    # Calculate the error squared.
    sqrErr = np.square((hyp-y))
    # Calculate cost function.
    J = (1/(2*m))*np.sum(sqrErr)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(X)

    # Calculate and update theta vector number of times given.
    for i in range (0, num_iters):
        hyp = np.dot(X, theta)
        err = (hyp-y)
        gradient = (1/m)*(np.dot(X.T, err))
        theta = theta - alpha*gradient
        J_history.append(compute_cost(X, y, theta))
            
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    # Calculate pinv(X).
    calc1 = np.dot(X.T, X)
    inv = np.linalg.inv(calc1)
    pinv = np.dot(inv, X.T)
    
    #Calculate optimal parameters.
    pinv_theta = np.dot(pinv, y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(X)

    # Update theta vector number of times given or until loss value is smaller than 1e-8.
    for i in range (0, num_iters):
        hyp = np.dot(X, theta)
        err = (hyp-y)
        gradient = (1/m)*(np.dot(X.T, err))
        theta = theta - alpha*gradient
        J_history.append(compute_cost(X, y, theta))
        
        # If the improvement is less than 1e-8 then break.
        if i > 0 and J_history[i-1] - J_history[i] < 1e-8:
            break
        

    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    np.random.seed(42) # Initialize random seed
    
    # Iterate over all possible alphas
    for alpha in alphas:
        # Initialize theta with random values
        theta = np.random.randn(X_train.shape[1])
        theta, _ = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        # Compute cost with optimal thetas found in gradient decent
        J_val = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = J_val

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    m = X_train.shape[1]
    
    for j in range(1,6):
        minCost = None
        bestFeat= None
        
        # Initialize random seed
        np.random.seed(42)
    
        # Initialize theta with random values
        randTheta = np.random.random(j+1)
        
        # Iterate over all features
        for i in range (m):
            if (i not in selected_features):
                selected_features.append(i)
                
                # Select the colums that match the selected features and add bias column
                X_train_temp = apply_bias_trick(X_train[:, selected_features])
                X_val_temp = apply_bias_trick(X_val[:, selected_features])
                theta, _ = efficient_gradient_descent(X_train_temp , y_train, randTheta, best_alpha, iterations)
            
                # Compute cost with optimal thetas found in gradient decent
                J_val = compute_cost(X_val_temp, y_val, theta)
                
                # Update best feature if current feature has lower loss
                if minCost is None or J_val <= minCost:
                    minCost = J_val
                    bestFeat = i
                    
                selected_features.pop()
        
        # Add the feature with the least cost to selected features array
        selected_features.append(bestFeat)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    size = df.shape[1]
    # Create list of column names
    column_names = df.columns.values
    new_columns = []
    
    # Iterate over combinations of columns
    for i in range (size):
        for j in range (i, size):
            
            # Square all the columns and assign name
            if i == j:
                new_name = column_names[i]+"^2"
                new_column = df[column_names[i]] ** 2
                new_column.name = new_name
                
                new_columns.append(new_column)
            
            # Multiply all combinations of different columns
            if i != j:
                new_name = column_names[i] + "*" + column_names[j]
                new_column = df[column_names[i]]*df[column_names[j]]
                new_column.name = new_name
                
                new_columns.append(new_column)
    
    # Add all the new columns to the data frame
    df_poly = pd.concat([df_poly] + new_columns, axis=1)
    
    return df_poly
