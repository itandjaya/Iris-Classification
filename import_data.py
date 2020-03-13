"""
## import_data.py 
## imports and pre-process raw data.

Data Creator: R.A. Fisher
Data Donor: Michael Marshall (MARSHALL%PLU '@' io.arc.nasa.gov)

Data Attribute Information:

1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
    -- Iris Setosa:         0
    -- Iris Versicolour:    1
    -- Iris Virginica:      2
"""

import os
import numpy as np
from random import shuffle;

IRIS_CLASS          =   {'Iris-virginica\n': 2, 'Iris-versicolor\n': 1, 'Iris-setosa\n':0}
DATA_PATH           =   os.getcwd() + r'/data/';
FULL_FILE_NAME      =   DATA_PATH + r'iris.DATA';

def load_data( filename = FULL_FILE_NAME):

    single_data = [0];
    raw_data = [];
    with open( filename, 'r') as fo:
        
        while single_data:

            single_data =   fo.readline();
            single_data =   single_data.split(",");
            if len(single_data) < 5:    break;

            feature     =   list(   map(lambda s: float(s)/10., single_data[:4]));
            #feature     =   np.array(   list(feature), dtype = 'float64') / 10.;
            label       =   IRIS_CLASS[ single_data[-1]];

            raw_data.append(    (feature, label));

    shuffle(raw_data);
    x_data, y_data  =   list(zip(*raw_data));

    ## Converts data to numpy format.
    x_data  =   np.array(   x_data, dtype = 'float64');
    y_data  =   np.array(   y_data, dtype = 'int32');

    #for i in range(len(x_data)):    print(x_data[i], y_data[i]);
    #print(x_data.shape);
    #print(y_data.shape);

    return x_data, y_data;   
    