# Databricks notebook source
import ts.flint
from ts.flint import FlintContext
flintContext = FlintContext(sqlContext)

# COMMAND ----------

sp500 = flintContext.read.dataframe(spark.table('sp500_csv').withColumnRenamed('Date', 'time'))
sp500_return = sp500.withColumn('return', 10000 * (sp500['Close'] - sp500['Open']) / sp500['Open']).select('time', 'return')
sp500_return.show()

# COMMAND ----------

from ts.flint import windows

sp500_previous_day_return = sp500_return.shiftTime(windows.future_absolute_time('1day')).toDF('time', 'previous_day_return')
sp500_joined_return = sp500_return.leftJoin(sp500_previous_day_return)
sp500_joined_return.show()

# COMMAND ----------

sp500_joined_return = sp500_return.leftJoin(sp500_previous_day_return, tolerance='3days').dropna()
sp500_joined_return.show()

# COMMAND ----------

from ts.flint import summarizers

sp500_decayed_return = sp500_joined_return.summarizeWindows(
    window = windows.past_absolute_time('7day'),
    summarizer = summarizers.ewma('previous_day_return', alpha=0.5)
)

sp500_decayed_return.show()

# COMMAND ----------

from ts.flint import udf
import numpy as np

@udf('double', arg_type='numpy')
def decayed(columns): 
    v = columns[0]
    decay = np.power(0.5, np.arange(len(v)))[::-1]
    return (v * decay).sum()

sp500_decayed_return = sp500_joined_return.summarizeWindows(
    window = windows.past_absolute_time('7day'),
    summarizer = {'previous_day_return_decayed_sum': decayed(sp500_joined_return[['previous_day_return']])}
)

sp500_decayed_return.show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["previous_day_return", "previous_day_return_decayed_sum"],
    outputCol="features")

output = assembler.transform(sp500_decayed_return).select('return', 'features').toDF('label', 'features')

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

model = lr.fit(output)

# COMMAND ----------

model.summary.r2
