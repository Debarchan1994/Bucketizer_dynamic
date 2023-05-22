# from pyspark.sql import functions as F
from pyspark.sql.functions import col, lag, lead, isnull, concat, when, row_number, last, sum, lit, format_string, to_timestamp, unix_timestamp, dayofweek, hour, countDistinct
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, StringType, IntegerType, DoubleType, ArrayType, BooleanType, LongType
import numpy as np
# from transforms.api import transform_df, Input, Output
from pyspark.ml.feature import Bucketizer
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
# spark = SparkSession.builder.master("local[1]") \
#                     .appName('DebSpark') \
#                     .getOrCreate()

# spark.sparkContext.stop()



class Bucketizer_func():

    def __init__(self, df, inputCols, outputCols):
        # self.splitsArray = splitsArray
        self.df = df
        self.inputCols = inputCols
        self.outputCols = outputCols
        # assert isinstance(self.splitsArray, list)
        # inputCols is a dictionary with the name of the input columns as key and the necessary ranges for the bucket size as the value in the form of tuple
        # for example inputCols = {'my_column' : (0,100,10)}
        assert isinstance(self.df, DataFrame)
        assert isinstance(self.inputCols, dict)    
        assert isinstance(self.outputCols, list)
        assert len(self.inputCols) == len(self.outputCols)
        #super().__init__(splitsArray=splitsArray, inputCols=inputCols, outputCols=outputCols)
        # self.df = df
        self.temp_dict = {}
        self.temp_dict_new = {}
        self.cat_symbol = ' ... '
        self.label_array_names = []

    def function(self):

        # Define the bins

        for items in self.inputCols.items():
            
            col_list_name = 'dist_' + str(items[0])
            self.temp_dict.update({str(col_list_name) : [-float('inf')] + list(range(*items[1])) + [float('inf')]})

         
        for items in self.temp_dict.items():
             
            temp_name = [float(i) for i in items[1]]

            cat_name = []

            my_list = []

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
            # print(cat_name)
            for label in cat_name:
                my_list.append(lit(label))

            # print(my_list)

            self.temp_dict_new.update({str(temp_name_new) : F.array(*(my_list))})

            # print(self.temp_dict_new['a_label_array'])
            # print(np.array(int(label) for label in cat_name))
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

        # df = bucketizer.setHandleInvalid('keep').transform(self.df)
        df = bucketizer.transform(self.df)

        for i in range(len(self.temp_dict_new)):

            temp_label = self.label_array_names[i].split('_label_array')[0] + '_label'

            df = df.withColumn('{}'.format(temp_label), 
                               self.temp_dict_new[str(self.label_array_names[i])].getItem(F.col('{}'.format(self.outputCols[i])).cast('integer')))

        return df
