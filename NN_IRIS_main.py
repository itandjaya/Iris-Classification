## NN_digits_main.py
## Main function to test the Neural Network - digits.
#####################################################

##  4-Layers NN: [28*28, 60, 15, 10].
##  Image size is 28 x 28 pixels.
##


from import_data import load_data;
from NN_Model import NN_Model, one_hot;
from random import randint;
from numpy import save as np_save, load as np_load;

import numpy as np;
import os;
import matplotlib.pyplot as plt;    #   To plot Cost vs. # of iterations.

CURRENT_PATH        =   os.getcwd();
#CURRENT_HOME        =   (os.path.expanduser('~'))

Grad_Descent_Method, Minimize_funct_method    =   True, False;

PLOT_COST           =   True;      ##  Plot J_cost vs. # of iterations to check if J_cost converges to minimum.    
LOAD_PREV_THETAS    =   False;      ##  Load the previous trained weights. Otherwise, randomize initial weights.
CONTINUOUS_TRAINING =   True;      ##  If True, then it will train the NN (10k training iterations.)
COUNT_ERROR         =   False;       ##  Compute the error rate of the trained prediction function.
                                    ##      against the test samples (60k images). So far, ~8.5% error.

CLASS_GROUPS = ("Iris Setosa", "Iris Versicolour", "Iris Virginica");

def plot_iris_data(x, y):
    ##  Using matplotlib to display the gray-scaled digit image.
    
    ## Preparing data by classification groups
    groups_sepal  =   [0]*max(y+1);
    groups_petal  =   [0]*max(y+1);

    for i in range(len(y)):

        if groups_sepal[y[i]] == 0:
            groups_sepal[y[i]]   =   (   [ x[i][0] ], [ x[i][1] ]);
            groups_petal[y[i]]   =   (   [ x[i][2] ], [ x[i][3] ]);
        
        else:
            groups_sepal[y[i]][0].append(  x[i][0] );
            groups_sepal[y[i]][1].append(  x[i][1] );
            groups_petal[y[i]][0].append(  x[i][2] );
            groups_petal[y[i]][1].append(  x[i][3] );
            


    ## Setting up the plot variables.
    COLOR_LOOKUP    =   {0:'red', 1:'green', 2:'yellow'};
    colors          =   ('red', 'green', 'blue');

    #fig       =   plt.figure();
    fig, (ax_sepal, ax_petal)    =   plt.subplots(2, sharex = True);

    ## Plotting Sepal length and width.
    for data, color, group in zip(groups_sepal, colors, CLASS_GROUPS):
        x, y    =   data;
        ax_sepal.scatter( x, y , alpha=0.8, c = color, edgecolors='none', s=30, label=group);

    plt.title('Iris Sepal & Petal Length and Width');    

    ## Plotting Petal length and width.
    for data, color, group in zip(groups_petal, colors, CLASS_GROUPS):
        x, y    =   data;
        ax_petal.scatter( x, y , alpha=0.8, c = color, edgecolors='none', s=30, label=group);

    plt.legend(loc=1);
    plt.show();
    return;

def main():

    x_data, y_data = load_data();   ##  Set parameter to True for initial download.
                                                    ##  Once data is present, set this to False to
                                                    ##      prevent re-downloading data.

    ## Plotting the Iris Petal and Sepal length and width.
    plot_iris_data(x_data, y_data);

    y_data      =   one_hot(y_data);

    #Split data: 80% test set, 20% validation set.
    i_80   =   int(len(y_data)*0.8);
    x_train, y_train    =   x_data[:i_80], y_data[:i_80];
    x_test, y_test      =   x_data[i_80:], y_data[i_80:];


    iris_nn     =   NN_Model (      x_train,        ## input data.
                                    y_train,        ## output data.
                                    3,              ## 3 NN layers: Input, hidden-layer, output.
                                    [4,4,3] );      ## num of nodes for each layer.

    


    if Grad_Descent_Method:
        print("\nNeural Network XNOR - using GRADIENT DESCENT ITERATION\n", "#"*30, "\n");    

        # File location where learned weight is saved.
        theta_file  =   CURRENT_PATH + r'/' + 'theta.npy';


        if  LOAD_PREV_THETAS:
            flat_thetas =   np_load(    theta_file);
            iris_nn.unflatten_Thetas( flat_thetas);

            if CONTINUOUS_TRAINING:
                iris_nn.train_NN();
                np_save(    theta_file, iris_nn.flatten_Thetas());

        else:
            iris_nn.train_NN();
            np_save(    theta_file, iris_nn.flatten_Thetas());
            
            # Display final cost after learning iterations.
            print("Final Cost J = ", iris_nn.J_cost(iris_nn.a[-1]));


        if PLOT_COST:
            
            #   Plot the J Cost vs. # of iterations. J should coverge as iteration increases.
            x_axis  =   range(len(iris_nn.J_cost_values));
            y_axis  =   iris_nn.J_cost_values;

            plt.plot(   x_axis, y_axis, label='J_cost vs. # of Iterations');
            plt.show();
            
        
        # Test model accuracy on Validation/Test set.
        acc_count = 0;
        for i in range( len(x_test)):

            x_input     =   x_test[i].flatten();
            y_val       =   np.argmax(y_test[i]);
            y_pred      =   iris_nn.predict(    x_input    )[0];
            #print(y_pred, y_val);

            if y_pred == y_val:   acc_count += 1;
        

        print(  "Test Accuraccy = {}".format( acc_count/len(x_test)));



    return 0;


if __name__ == "__main__":  main();
