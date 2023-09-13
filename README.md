This function is used to bucketize a column as per you need and add the bucketized values as additional columns to your existing 
        dataframe.Also you don't have to worry about the number the columns you want to bucketize, this function can bucketize as many 
        columns as you want, you just have to pass the names of the columns with the bucketization ranges into the inputCols paramter as a
        dictionary with the name of the input columns as key and the necessary ranges in the form of tuple or list for the bucket size 
        as the value. For example inputCols = {'my_column' : (0,100,10)} or {'my_column' : [0,10,100,200,500,1000]}
        
        Parameters :: \n
                    1) df : type(pyspark dataframe)--> The base dataframe
                    2) inputCols : type(dictionary)--> Key : column names that need to be bucketized , value : Bucketization range or list
                    3) outputCols : type(list)--> Desired names of the final output columns
                    4) cat_alternative : type(boolean)--> Defines whether 2 columns should be created of each input column with second column being the indicator for the last value of a particular bucket (example,  main column:  10 ... 20 , second alternate column : <20) or just 1 main column
                                                        (example, main column:  10 ... 20). 
                                                        The default value is False, but if True is passed as parameter then only the function will create 2 columns for each input column. 
