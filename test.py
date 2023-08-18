# Author: Debarchan Chatterjee


# from pyspark.sql import functions as F
from pyspark.sql.functions import col, lag, lead, isnull, concat, when, row_number, last, sum, lit, format_string, to_timestamp, unix_timestamp, dayofweek, hour, countDistinct
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, StringType, IntegerType, DoubleType, ArrayType, BooleanType, LongType
import numpy as np
# from transforms.api import transform_df, Input, Output
from pyspark.ml.feature import Bucketizer
from pyspark.sql.dataframe import DataFrame

# from transforms.api import transform_df, Input, Output
# from transforms.api import configure

from datetime import datetime


# @configure(profile=['NUM_EXECUTORS_16', 'DRIVER_MEMORY_LARGE', 'EXECUTOR_MEMORY_MEDIUM'])

# @transform_df(
#     Output("/BMW/Data Analysis Driving Dynamics Components Cluster1/EF-5 Datenanalyse/EF-53_Datenanalyse/02_datasets/test"),
#     JoinData=Input("ri.foundry.main.dataset.135b2ed0-3760-4b61-a462-e98b7cf627f7")
#     # source_flm4 = Input('ri.foundry.main.dataset.3f3b0a32-35f1-4fde-9fb1-0c7158c34700')
# )


class Bucketizer_func():

    """
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

        """

    def __init__(self, df, inputCols, outputCols, cat_alternative=False):
        
        self.df = df
        self.inputCols = inputCols
        self.outputCols = outputCols
        self.cat_alternative = cat_alternative
        
        assert isinstance(self.df, DataFrame), "Only pyspark data frames are allowed as input dataframe and the provided data type is {}".format(type(self.df))
        assert isinstance(self.inputCols, dict), "Only dictionaries are allowed as inputCols parameter and the provided data type is {}".format(type(self.inputCols))
        assert isinstance(self.outputCols, list), "Only lists are allowed as outputCols parameter and the provided data type is {}".format(type(self.outputCols))
        assert len(self.inputCols) == len(self.outputCols), "length of the input columns and output columns must be equal"
        assert isinstance(self.cat_alternative , bool) , "Only boolean values like True and False are accepted as input for cat_alternative paramter and the provided data type is {}".format(type(self.cat_alternative))
        #super().__init__(splitsArray=splitsArray, inputCols=inputCols, outputCols=outputCols)
        self.temp_dict = {}
        self.temp_dict_new = {}
        self.temp_dict_alt_new = {}
        self.cat_symbol = ' ... '
        self.label_array_names = []
        self.label_array_alt_names = []


    def function(self):

        # Define the bins

        for items in self.inputCols.items():
            
            col_list_name = 'dist_' + str(items[0])
            if type(items[1]) is list:
                self.temp_dict.update({str(col_list_name) : [-float('inf')] + items[1] + [float('inf')]})
            else:
                self.temp_dict.update({str(col_list_name) : [-float('inf')] + list(range(*items[1])) + [float('inf')]})

         
        for items in self.temp_dict.items():
             
            temp_name = [float(i) for i in items[1]]

            if self.cat_alternative is True:
                cat_name = []
                cat_alt_name = []

                temp_name_new = str(str(items[0]).split('dist_')[1]) + '_label_array'
                temp_alt_name_new = str(str(items[0]).split('dist_')[1]) + '_label_alt_array'

                self.label_array_names.append(temp_name_new)
                self.label_array_alt_names.append(temp_alt_name_new)
            
            # temp_name = 'my_bins_' + str(str(items[0]).split('dist_')[1])
            # self.temp_dict_new.clear()
            # self.temp_dict_new.update({str(temp_name) : [float(item) for item in items[1]]})
            
                for i in range(len(temp_name)):
                    if i < len(temp_name)-1:

                        if temp_name[i] == float('-inf'):            
                            cat_name.append( "< " + str("{:02d}".format(int(temp_name[i+1]))))
                            cat_alt_name.append( "< " + str("{:02d}".format(int(temp_name[i+1]))))

                        elif temp_name[i+1] == float('inf'):            
                            cat_name.append( "> " + str("{:02d}".format(int(temp_name[i]))))
                            cat_alt_name.append( "> " + str("{:02d}".format(int(temp_name[i]))))

                        else:                        
                            cat_name.append(str("{:02d}".format(int(temp_name[i]))) + self.cat_symbol + str("{:02d}".format(int(temp_name[i+1]))))
                            cat_alt_name.append("< " + str("{:02d}".format(int(temp_name[i+1]))))
                
            # Modify the labels
                self.temp_dict_new.update({str(temp_name_new) : F.array(*(F.lit(label) for label in cat_name))})
                self.temp_dict_alt_new.update({str(temp_alt_name_new) : F.array(*(F.lit(label) for label in cat_alt_name))})
            # print(self.temp_dict_new)

            else:
                cat_name = []

                temp_name_new = str(str(items[0]).split('dist_')[1]) + '_label_array'

                self.label_array_names.append(temp_name_new)
            
            # temp_name = 'my_bins_' + str(str(items[0]).split('dist_')[1])
            # self.temp_dict_new.clear()
            # self.temp_dict_new.update({str(temp_name) : [float(item) for item in items[1]]})
            
                for i in range(len(temp_name)):
                    if i < len(temp_name)-1:

                        if temp_name[i] == float('-inf'):            
                            cat_name.append( "< " + str("{:02d}".format(int(temp_name[i+1]))))

                        elif temp_name[i+1] == float('inf'):            
                            cat_name.append( "> " + str("{:02d}".format(int(temp_name[i]))))

                        else:                        
                            cat_name.append(str("{:02d}".format(int(temp_name[i]))) + self.cat_symbol + str("{:02d}".format(int(temp_name[i+1]))))
                
            # Modify the labels
                self.temp_dict_new.update({str(temp_name_new) : F.array(*(F.lit(label) for label in cat_name))})

        if self.cat_alternative is True:
            return self.temp_dict_new, self.temp_dict_alt_new

        else:
            return self.temp_dict_new

    def Bucketizer(self):

        self.function()
        temp_list_splitarray = []
        temp_list_inputcols = []

        for item in self.temp_dict.items():
            temp_list_splitarray.append(item[1])

        for item in self.inputCols.items():
            temp_list_inputcols.append(str(item[0]))

        bucketizer = Bucketizer(splitsArray = temp_list_splitarray,
                            inputCols= temp_list_inputcols,
                            outputCols = self.outputCols)

        df = bucketizer.setHandleInvalid('keep').transform(self.df)
        # df = bucketizer.transform(self.df)

        for i in range(len(self.temp_dict_new)):

            if self.cat_alternative is True:

            # temp_label = self.label_array_names[i].split('_label_array')[0] + '_label'
                temp_label = self.outputCols[i] + '_label'
                temp_alt_label = self.outputCols[i] + '_label_alt'

                df = df \
                                .withColumn('{}'.format(temp_label), self.temp_dict_new[str(self.label_array_names[i])].getItem(F.col('{}'.format(self.outputCols[i])).cast('integer'))) \
                                .withColumn('{}'.format(temp_alt_label), self.temp_dict_alt_new[str(self.label_array_alt_names[i])].getItem(F.col('{}'.format(self.outputCols[i])).cast('integer')))

            else:

                temp_label = self.outputCols[i] + '_label'
            
                # df = df.withColumn('{}'.format(temp_label), self.temp_dict_new[str(self.label_array_names[i])].getItem(F.col('{}'.format(self.outputCols[i])).cast('integer')))

                df = df.withColumn('{}'.format(temp_label), \
                            self.temp_dict_new[str(self.label_array_names[i])].getItem(F.col('{}'.format(self.outputCols[i])).cast('integer')))

                df = df.withColumn('{}'.format(temp_label), \
                                F.concat(F.lit('cl_'), F.lit(F.format_string('%02d', F.col("{}".format(self.outputCols[i])).cast('integer'))), \
                                 F.lit('_'), F.col('{}'.format(temp_label))))


        return df
