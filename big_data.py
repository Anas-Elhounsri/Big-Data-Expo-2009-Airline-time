# Importing libraries
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, UnivariateFeatureSelector
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

# Set up the SparkSession
spark = SparkSession.builder.appName("BigData-Project").master("local[*]").config("spark.sql.debug.maxToStringFields", 100).getOrCreate()

# Load the data from the CSV file
data = "2007.csv"
df = spark.read.csv(data, header=True, inferSchema=True)

############# Data Processing #############
# Define the columns to be dropped
columns_to_drop = ["ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
                    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
                    "LateAircraftDelay", "Distance", "FlightNum", "CRSElapsedTime",
                    "CancellationCode", "Cancelled", "year", "TailNum" ]

# Filter out the cancelled flights and then drop the selected columns 
df = df.filter(col("Cancelled") != 1).drop(*columns_to_drop)

# Convert some numerical variables from string to integer
columns_to_convert = ["DepTime","DepDelay", "TaxiOut", "ArrDelay"] #ArrDelay is the target variable but it's loaded as a string, we can't run a regression model if it's not a number
for column in columns_to_convert:
    df = df.withColumn(column, col(column).cast("integer"))
df = df.dropna() #dropping null value rows

print("Data processed successfully.")
total_unique_count = df.select("ArrDelay").distinct().count()
print(f"Totall count of unique values: {total_unique_count}")
############# OneHot Encoding variables for train/test #############

numeric_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime","CRSArrTime", "DepDelay", "TaxiOut"]
categorical_cols = ["UniqueCarrier", "Origin", "Dest"]

"""
Applied StringIndexer for categorical columns using a for loop with list comprehension.
Stringindexer gives every instance of the categorical variable every instance in the variable
"""
indexer = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in categorical_cols]
"""
Example Output:
UniqueCarrier   UniqueCarrier_index
AA                     0.0
BB                     1.0     
CC                     2.0
"""

"""
Applied OneHotEncoder for indexed categorical columns using a for loop with list comprehension.
OneHotEncoder converts every index to binary encoded values.
"""
encoder = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in categorical_cols]
"""
Example Output:
UniqueCarrier_encoded
(19, [0], [1.0])
(19, [2], [1.0])
(19, [3], [1.0])

where 19 is the length of the vector (How many catgorical values), [0] is the index number, and [1.0] This the value 
corresponding to the non-zero indices, during the training process, the ML will consider all values 0 expect for 
that one instance.
"""
print("Successfully OneHot encoded.")
############# Building the train/test & the pipeline #############
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# VectorAssembler takes a list of input columns and assembles them into a single vector column
assembler = VectorAssembler(inputCols=numeric_cols + [f"{col}_encoded" for col in categorical_cols],
    outputCol="features")

# Define the pipeline, where each stage is defined
pipeline = Pipeline(stages=indexer + encoder + [assembler])

# Fit and transform the data using the pipeline and apply it on train_data and test_data
model = pipeline.fit(train_data)
transformed_train_data = model.transform(train_data)
transformed_test_data = model.transform(test_data)

# transformed_train_data.select("features", "ArrDelay").show(truncate=False)
# transformed_test_data.select("features", "ArrDelay").show(truncate=False)

print("Successfully created train/test data.")
# Numeric Assembler for running univariate filter and correlation matrix on the numeric variables
numericAssembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
numericPipeline = Pipeline(stages = [numericAssembler])
numericModel = numericPipeline.fit(df)
numeric_data = numericModel.transform(df)

############# Univariate Analysis #############

selector = UnivariateFeatureSelector(featuresCol="numeric_features", outputCol="selected_features", labelCol="ArrDelay", selectionMode="fpr")
selector.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(0.05)
univariate_df = selector.fit(numeric_data).transform(numeric_data)
print("As seen in the below output, all numeric variables were selected with a threshold p value of 0.05\n")
# univariate_df.select("selected_features", "ArrDelay").show(truncate=False)
# numeric_data.select("numeric_features", "ArrDelay").show(truncate=False)

############# Pearson Correlation Matrix #############

r1 = Correlation.corr(numeric_data, "numeric_features").head()
# print("All correlations are very close to 0, indicating low correlation between numeric variables, therefore all variables so far are worthwhile to keep\n")
# print("Pearson correlation matrix for numeric variables:\n" + str(r1[0]))

############# Linear Regression Model #############

# Linear Regression Estimator
lr = GeneralizedLinearRegression(family="gaussian", link="identity", featuresCol="features", labelCol="ArrDelay", maxIter=10)
# Defining the parameter grid
lrParamGrid = ParamGridBuilder().addGrid(lr.regParam,[0.0, 0.1, 1, 5, 10]).build()
# Defining the general lr evaluator
lrEvaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")

"""
Create a Cross Validator with 3 folds - CROSS VALIDATION NOT WORKING BECAUSE I GET AN ERROR: "[WinError 10061] No connection could be made because the target machine actively refused it"
lrCrossValidator = CrossValidator(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=lrEvaluator, numFolds=3)
Fit the cross-validator to the training data - CROSS VALIDATION NOT WORKING BECAUSE I GET AN ERROR: "[WinError 10061] No connection could be made because the target machine actively refused it"
lrModel = lrCrossValidator.fit(transformed_train_data)
"""

# Create a TrainValidationSplit
lrTvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=lrEvaluator, trainRatio=0.8)
# Fit the TrainValidationSplit model to the train data
lrModel = lrTvs.fit(transformed_train_data)

# Make predictions on the test set
lrPredictions = lrModel.transform(transformed_test_data)
# Evaluating the model
lrRMSE = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse").evaluate(lrPredictions)
lrMAE = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae").evaluate(lrPredictions)
lrr2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2").evaluate(lrPredictions)
print("Linear Regression - Root Mean Squared Error (RMSE) on test data =", lrRMSE)
print("Linear Regression - Mean Absolute Error (MAE) on test data =", lrMAE)
print("Linear Regression - R-squared (R2) on test data =", lrr2)

#############  Decision Tree Regression Model ############# 

# Decision Tree Regressor Estimator
dt = DecisionTreeRegressor(featuresCol="features", labelCol="ArrDelay")
# Fit Decision Tree Regressor
dtModel = dt.fit(transformed_train_data)
# Make predictions on the test set
dtPredictions = dtModel.transform(transformed_test_data)
# Evaluating the model
dtRMSE = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse").evaluate(dtPredictions)
dtMAE = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae").evaluate(dtPredictions)
dtr2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2").evaluate(dtPredictions)
print("Decision Tree Regression - Root Mean Squared Error (RMSE) on test data =", dtRMSE)
print("Decision Tree Regression - Mean Absolute Error (MAE) on test data =", dtMAE)
print("Decision Tree Regression - R-squared (R2) on test data =", dtr2)

#############  Random Forest Regression Model ############# 

# Random Forest Regressor Estimator
rf = RandomForestRegressor(featuresCol="features", labelCol="ArrDelay")
# Fit Decision Tree Regressor
rfModel = rf.fit(transformed_train_data)
# Make predictions on the test set
rfPredictions = rfModel.transform(transformed_test_data)
# Evaluating the model
rfRMSE = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse").evaluate(rfPredictions)
rfMAE = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae").evaluate(rfPredictions)
rfr2 = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2").evaluate(rfPredictions)
print("Random Forest Regression - Root Mean Squared Error (RMSE) on test data =", rfRMSE)
print("Random Forest Regression - Mean Absolute Error (MAE) on test data =", rfMAE)
print("Random Forest Regression - R-squared (R2) on test data =", rfr2)

spark.stop()