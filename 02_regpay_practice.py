# Databricks notebook source
# MAGIC %md You may find this accelerator at https://github.com/databricks-industry-solutions/regular-payments. Please clone this repo to run, instead of using the downloaded DBC file.

# COMMAND ----------

# MAGIC %md
# MAGIC # Financial Vulnerability 
# MAGIC Using signal processing theory and Fourier transforms, we extract regular payment informations from a large dataset of card transactions data. Although it is easy to eye ball regularity in payments when looking at specific transactions, doing so at scale across billions of card transactions requires a scientific (and programmatic) approach to a business problem. In this solution accelerator, we demonstrate a novel approach to consumer analytics by combining core mathematical concepts with engineering best practices and state of the art optimizations techniques to better model customers' behaviors and provide millions of customers with personalized insights. With 40% of americans struggling to come up with $400 for an unexpected expense [[source](https://www.cnbc.com/2019/07/20/heres-why-so-many-americans-cant-handle-a-400-unexpected-expense.html)], such a framework could be used to suggest financial goals and provide customers with recommended actions to better spread regular payments over billing cycles, minimize periods of financial vulnerability and better plan for unexpected events.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC We will use an hypothetical card transaction dataset but only focus on customer identifier, transaction date, transaction amount and merchant information. For more information about extracting merchant name from card transaction narrative, please refer to our transaction enrichment [solution accelerator](https://databricks.com/blog/2021/05/10/improving-customer-experience-with-transaction-enrichment.html). Although our sample data easily fits in memory, real life data set certainly would not, hence a need to a solid distributing computing business logic.

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
transactions = pd.read_csv('data/ledger.csv')[['customer', 'merchant', 'date', 'amount']]
transactions_df = spark.createDataFrame(transactions).withColumn('date', F.to_date('date'))
display(transactions)

# COMMAND ----------

# MAGIC %md
# MAGIC As we might expect, raw transactions are noisy and imbalanced. Some one-off high value expenses are mixed with highly regular payments we will aim at uncovering through this framework. See an example below for our hypothetical customer, Brittney Perritt.

# COMMAND ----------

from datetime import timedelta
import matplotlib.pyplot as plt

example_customer = transactions_df \
  .filter(F.col('customer') == 'Brittney Perritt') \
  .groupBy('date') \
  .agg(F.sum('amount').alias('amount')) \
  .toPandas()

plt.figure(figsize=(15, 5))
plt.stem(example_customer['date'], example_customer['amount'])
plt.title("Payments over time")
plt.xlabel("Date")
plt.ylabel("transaction amount [$]")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building Timeseries
# MAGIC As experimented in our previous notebook when dabbling on physics theory, a time series of events (non continuous signal) would only be modelled through spectrum analysis by an infinite number of sine waves. However, when converting pulse functions into triangular waves, our regular payments may yield dominant frequencies whilst maintaining key characteristics (such as transaction amount). In this section, we make use of different utility functions to transform raw transactions into "Fourier actionable" signal.

# COMMAND ----------

from utils.regpay_utils import *

# COMMAND ----------

# MAGIC %md
# MAGIC We group customers by merchant names and create a time series of transactions, ensuring dense representation (i.e. capturing dates with no transactions), hence creating a continuous signal. 

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import pandas as pd
import numpy as np
  
@udf('array<double>')
def build_timeseries(xs):

  # build a time indexed dataframe from our transactions tuples
  df = pd.DataFrame(
    [[x['date'], x['amount']] for x in xs], 
    columns=['date', 'amount']
  ).set_index('date')
  
  # create an index covering all dates
  min_date = np.min(df.index)
  max_date = np.max(df.index)
  calendar = create_calendar_df(min_date, max_date)
  
  # complete time series with zero value
  ts = pd.merge(df, calendar, left_index=True, right_index=True, how='right').fillna(0)['amount']
  
  # return a dense representation of our timeseries
  return ts.to_list()

# COMMAND ----------

# MAGIC %md
# MAGIC We aggregate all transactions for a given day and a given merchant and capture that information as a `struct` column. A first filter will be applied to ensure we have enough observations to drive meaningful insights. Although this threshold was empirically defined, one could read more about the [Nyquist Shannon theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem) and the need to observe `N` points (or to be more scientifically correct the influence of a sampling rate `S`) in order to recompose signal of `f` frequency. Note that our signal aggregated daily and spanning across 1 year exhibits a sampling rate of 1/365.

# COMMAND ----------

transactions_ts = (
  transactions_df
    .groupBy('customer', 'merchant', 'date')
    .agg(
      F.sum('amount').alias('amount')
    )
    .withColumn('struct', F.struct('date', 'amount'))
    .groupBy('customer', 'merchant')
    .agg(
      F.min('date').alias('min_date'),
      F.collect_list('struct').alias('transactions'),
      F.sum(F.lit(1)).alias('num_transactions')
    )
    .filter(
      # we need at least N points to observe enough periodicity
      F.col('num_transactions') > config['timeseries_minimum']
    )
    .select(
      F.col('customer'),
      F.col('merchant'),
      F.col('min_date'),
      build_timeseries('transactions').alias('transactions')
    )
)

display(transactions_ts)

# COMMAND ----------

# MAGIC %md
# MAGIC Using our utility function, we convert time series into triangular signal. Expensive operation at enterprise scale, we leverage [autoscaling](https://docs.databricks.com/clusters/configure.html#cluster-size-and-autoscaling) capability of our cluster and expose this logic as a user defined function to benefit from a maximum degree of parallelism. 

# COMMAND ----------

@udf('array<float>')
def to_triangle_signal_udf(xs):
  return to_triangle_signal(np.asarray(xs))

# COMMAND ----------

transactions_ts_tri = transactions_ts.withColumn(
  'triangles', 
  to_triangle_signal_udf(F.col('transactions'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can visualize a few records to ensure correct execution of our pre-processing step. Signals were converted from pulse functions to continuous signal of triangular form, better fit for fourier transform. See an example below for our same customer, Brittney Perritt and an hypothetical merchant, Bins.

# COMMAND ----------

from datetime import timedelta
import matplotlib.pyplot as plt

example_customer = transactions_ts_tri \
  .filter(F.col('customer') == 'Brittney Perritt') \
  .filter(F.col('merchant') == 'Bins') \
  .toPandas() \
  .iloc[0]

example_customer_ts1 = [[example_customer.min_date + timedelta(days=i), x] for i, x in enumerate(example_customer['transactions'])]
example_customer_ts1 = pd.DataFrame(example_customer_ts1, columns=['date', 'transaction'])
example_customer_ts1 = example_customer_ts1[example_customer_ts1['transaction'] > 0]

example_customer_ts2 = [[example_customer.min_date + timedelta(days=i), x] for i, x in enumerate(example_customer['triangles'])]
example_customer_ts2 = pd.DataFrame(example_customer_ts2, columns=['date', 'transaction'])

fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)

ax[0].stem(example_customer_ts1['date'], example_customer_ts1['transaction'])
ax[0].set_title("Payments over time")
ax[0].set_ylabel("transaction amount [$]")

ax[1].plot(example_customer_ts2['date'], example_customer_ts2['transaction'], lw=0.5, color='red')
ax[1].set_ylabel("regularity of payments [$]")
ax[1].set_xlabel("transaction date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fourier analysis
# MAGIC With our timeseries converted from pulse functions to triangular signal, we can detect dominant frequencies as explained in our previous notebook. We will capture this information in a `struct` column containing the amplitude (real and imaginary) or each dominant frequency. The goal will be to summarize signal into mathematical representations that can be recomposed on demand and aggregated across merchants.

# COMMAND ----------

from pyspark.sql.types import *

struct = ArrayType(StructType(
  [
    StructField('real', DoubleType(), True),
    StructField('imag', DoubleType(), True),
    StructField('freq', DoubleType(), True),
    StructField('size', DoubleType(), True),
  ]
))

# COMMAND ----------

# MAGIC %md
# MAGIC Given the overall time span of our dataset (365 days), we may ignore frequencies below a given threshold (i.e. periods above a certain limit). Set to 60 days in this example, we do not allow for transaction periodicity greater than 60 days (2 months). One could link this back to our Nyquist Shannon theorem explained earlier should we want to focus on the exact science behind this approach.

# COMMAND ----------

def filter_valid_frequencies(df):
  # zero frequency represents the mean of our signal
  max_p = config['periodicity_max']
  df = df[(df['freq'] == 0) | (np.abs(df['size'] / df['freq']) < max_p)]
  return df

# COMMAND ----------

# MAGIC %md
# MAGIC As part of this process, dominant frequencies will be isolated from the noise through a simple anomaly detection model (kernel density estimation) as explained in our previous notebook. 

# COMMAND ----------

@udf(struct)
def run_spectral_analysis(xs):
  
  # extract all frequencies
  fft = fourier_transform(xs)

  # identify dominant frequencies
  threshold = detect_anomalies(fft.amplitudes)
  fft_filtered = fft.filter(threshold)
  df = fft_filtered.to_df()

  # only focus on frequencies above a given threshold
  df = filter_valid_frequencies(df)

  # the only frequency we have left is the zero frequency (i.e. average)
  if df.shape[0] == 1:
    return []

  rs = []
  for i, r in df.iterrows():
    # converting numpy to primitives
    rs.append([
      r['real'].item(), 
      r['imag'].item(), 
      r['freq'].item(), 
      r['size'].item()
    ])
    
  # return all regular signals
  return rs

# COMMAND ----------

# MAGIC %md
# MAGIC Although Fourier tells us about the amplitude (that can be converted back into approximate dollar value), frequency and phase of each signal, we also need to capture information about the first transaction ever recorded for a given user. This will allow us to overlay signals from different merchants with a consistent date index. We do so by applying a simple window function returning the first recorded date for each of our customers and compute the time difference with each of its individual signals. 

# COMMAND ----------

from pyspark.sql import Window
first_day = Window.partitionBy('customer').orderBy(F.col('min_date'))

periodicity_df = (
  transactions_ts_tri
    .withColumn('fft', run_spectral_analysis('triangles'))
    .filter(F.size('fft') > 0)
    .withColumn('date', F.first('min_date').over(first_day))
    .withColumn('shift', F.datediff('min_date', 'date'))
    .drop('triangles', 'min_date')
    
)

display(periodicity_df)

# COMMAND ----------

# MAGIC %md
# MAGIC As we were able to model each regular payment behaviors through simple mathematical characteristics, we could store that data asset as a delta table that can be re-used across multiple scenario spanning from customer 360, marketing analytics, mobile banking experience or fraud prevention.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recomposing signals
# MAGIC In the previous section, we were able to summarize regular transactions into simple mathematical representations that could be grouped and additioned. In this section, we can generate data points to simulate those transactions and overlay multiple regular payments to better identify customers' expenditures, cash flow and payment behaviors. Given the dense representation of our timeseries, we express our logic through Spark ML `Vectors` objects that can be grouped together using built in spark ML functions (provided they have same dimensions)

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT

@udf(VectorUDT())
def transactions_to_vector(days, shift, transactions):
  appended = np.zeros(shift).tolist() + transactions
  # we ensure all vectors are of same dimensions
  if len(appended) < days:
    appended = appended + np.zeros(days - len(appended)).tolist()
  else:
    appended = appended[:days]
  return Vectors.dense(appended)

# COMMAND ----------

# MAGIC %md
# MAGIC We appreciate the maths may sound complex (no pun intended), but recomposing signal from fourier coefficients is relatively straightforward using cosine and sine numpy functions. For each dominant frequency, we combine cosine and sine signal from our real and imaginary numbers, respectively. Similar to raw transactions that can be mapped as vector, we create a dense vector representation of our recomposed regular payments.

# COMMAND ----------

def frequencies_to_signal(days, shift, fft):
  import numpy as np
  sampling_rate = 1/days
  x = np.arange(0 - shift, days - shift, 1)
  transactions_ts_recombined = np.zeros((len(x),))
  for i, f in enumerate(fft):
    cos = f['real'] * np.cos(f['freq'] * 2 * np.pi * x * sampling_rate)
    sin = f['imag'] * np.sin(f['freq'] * 2 * np.pi * x * sampling_rate)
    transactions_ts_recombined += sampling_rate * (cos - sin)
  return transactions_ts_recombined

# COMMAND ----------

@udf(VectorUDT())
def frequencies_to_vector(days, shift, fft):
  return Vectors.dense(frequencies_to_signal(days, shift, fft).tolist())

# COMMAND ----------

def recompose_signal(periodicity_df, days):
  return (
    periodicity_df
      .withColumn('transactions', transactions_to_vector(F.lit(days), F.col('shift'), F.col('transactions')))
      .withColumn('recomposed', frequencies_to_vector(F.lit(days), F.col('shift'), F.col('fft')))
      .withColumnRenamed('first_date', 'date')
      .select('customer', 'date', 'merchant', 'transactions', 'recomposed', 'shift')
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have built a mathematical model that captures all regular payments and that can be recomposed and aggregated for a given time window thanks to the additive property of our vector representation. Set to 365 days here, we represent periodic behaviors for each of our customers for this year to come. See an example below for our same customer, Brittney Perritt and an hypothetical merchant that exhibits a monthly regular pattern, Cartwright.

# COMMAND ----------

recomposed_signal_df = recompose_signal(periodicity_df, 365)

# COMMAND ----------

from datetime import timedelta
from datetime import date
import matplotlib.pyplot as plt

example_customer = recomposed_signal_df \
  .filter(F.col('customer') == 'Brittney Perritt') \
  .filter(F.col('merchant') == 'Cartwright') \
  .select('recomposed') \
  .toPandas() \
  .iloc[0]['recomposed']

today = date.today()

df = pd.DataFrame(
  [[today + timedelta(days=i), r] for i, r in enumerate(example_customer)], 
  columns=['date', 'amount']
)

plt.figure(figsize=(15, 5))
plt.plot(df['date'], df['amount'])
plt.title("Payments over time")
plt.xlabel("Date")
plt.ylabel("transaction amount [$]")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregating signals
# MAGIC Given the use of `Vector` objects, we can benefit from their additive properties in order to aggregate multiple payments across multiple merchants. Since we do not want to manipulate arrays within complex user aggregated functions, this can achieved via a simple `groupBy` clause and the excellent `Summarizer` class from Spark ML. In the code below, we aggregate both individual raw transactions and recompose signals for every customers.

# COMMAND ----------

from pyspark.ml.stat import Summarizer

def aggregate_signal(recomposed_signal_df):
  return (
    recomposed_signal_df
      .groupBy('customer', 'date')
      .agg(
        Summarizer.sum(F.col('transactions')).alias('transactions'),
        Summarizer.sum(F.col('recomposed')).alias('recomposed'),
        F.sum(F.lit(1)).alias('count')
      )
  )

# COMMAND ----------

recomposed_signal_agg_df = aggregate_signal(recomposed_signal_df)
display(recomposed_signal_agg_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can select a given customer and represent their raw transactions as well as individual payments and aggregated view modelled through fourier transform. See example below with our same customer, Brittney Perritt. 

# COMMAND ----------

from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)

# given a customer, we return all aggregated regular payments
customer_id = 'Brittney Perritt'
xs = recomposed_signal_agg_df.filter(F.col('customer') == customer_id).limit(1).toPandas().iloc[0]

# actual transactions
xs_agg_ts = [[xs['date'] + timedelta(days=i), x] for i, x in enumerate(xs['transactions'])]
xs_agg_ts = pd.DataFrame(xs_agg_ts, columns=['date', 'transaction'])
ax[0].stem(xs_agg_ts['date'], xs_agg_ts['transaction'], label='Transactions')
ax[0].set_title("Payments over time")
ax[0].set_ylabel("transaction amount [$]")
ax[0].legend(loc='upper left')

# modeled regular payments
xs_agg_df = [[xs['date'] + timedelta(days=i), x] for i, x in enumerate(xs['recomposed'])]
xs_agg_df = pd.DataFrame(xs_agg_df, columns=['date', 'transaction'])
ax[2].plot(xs_agg_df['date'], xs_agg_df['transaction'], lw=0.5, color='red', label='Composite')
ax[2].axhline(np.average(xs['recomposed']), lw=0.2, ls='--', label='Average')
ax[2].set_ylabel("regularity of payments [$]")
ax[2].legend(loc='upper left')

# modeled individual payments
for i, r in recomposed_signal_df.filter(F.col('customer')==customer_id).toPandas().iterrows():
  xs_ind_df = [[r['date'] + timedelta(days=i), x] for i, x in enumerate(r['recomposed'])]
  xs_ind_df = pd.DataFrame(xs_ind_df, columns=['date', 'transaction'])
  ax[1].plot(xs_ind_df['date'], xs_ind_df['transaction'], lw=0.5, label='Payments to {}'.format(r['merchant']))
ax[1].set_ylabel("regularity of individual payments [$]")
ax[1].set_xlabel("transaction date")
ax[1].legend(loc='upper left')

# COMMAND ----------

# MAGIC %md
# MAGIC Given the above visualization, we can conclude that this specific customer would need to maintain an average balance of $700 in order to cope with the many regular payments, whether those are subscription based services, utility bills, financial products, rents, or any other payment exhibiting regular behaviors (e.g. paying for gas). We observe periods of "financial stress" where this customer may be more financially vulnerable (around September) with aggregated payments of > $800 as well as period of financial relief (around June). 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimizing payments
# MAGIC In the previous sections, we were able to extract regular payments from raw transactions and model those in a way that can be aggregated in order to simulate customer most expected behaviors, over time. In this section, we will leverage techniques borrowed from the world of ML optimization to suggest changes in customer behaviors (e.g. automated saving) and payment dates.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType

optimized_struct = StructType(
  [
    StructField('customer', StringType(), True),
    StructField('merchant', StringType(), True),
    StructField('shift', IntegerType(), True)
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC The aim will be to play on different payment dates (within a same billing cycle) for each regular payment information and every customers. This becomes an optimization problem where our goal is to minimize the variance of our aggregated signal. An ideal optimization would be a transaction signal that is closer to the mean. As brute force optimization would not be ideal (the example above showed 25 regular payments, each that could theoretically span over ~30 days), we can borrow some best practices from the world of AI and hyper parameter tuning. Leveraging [Naive Bayes optimization](https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f) techniques through `hyperopt`, we will find the best parameters of our regular payments (i.e. the ideal payment dates) that minimizes period of financial vulnerability for each of our customers.

# COMMAND ----------

from hyperopt import tpe, fmin, STATUS_OK, Trials

def optimize_payments_pandas(pdf, space):
  
  # optimizing our signals over a 1 year history
  days = 365
  
  # Our objective function, optimizing variance of our signal
  def objective_function(params):
    
    # recreate signals for different phases and same periodicity
    ffts = zip(pdf['merchant'], pdf['fft'])
    signals = np.array(
      [frequencies_to_signal(days, params[merchant], fft) for merchant, fft in ffts]
    )
    
    # aggregate individual signals
    composite = signals.sum(axis=0)
    
    # compute composite signal variance
    # we try to minimize variance, hence split payments over time
    return {"loss": composite.std(), "status": STATUS_OK}
  
  # Optimize our signals to minimize variance
  trials = Trials()
  return fmin(
      fn = objective_function,
      space = space, 
      algo = tpe.suggest, 
      max_evals = config['optimization_runs'], 
      trials = trials
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Since fourier give us frequency of payments, we can easily extract the billing cycle for each regular transaction. Our optimizer will intelligently iterate through various calendar days during that billing period for each merchant.

# COMMAND ----------

def get_max_period(fft):
  max_period = 0
  for fft_rec in fft:
    if fft_rec['freq'] > 0:
      period = np.abs(fft_rec['size'] / fft_rec['freq'])
      if period > max_period:
        max_period = period
  return int(max_period)

def get_period_grid(x):
  from hyperopt import hp
  return hp.uniformint(x['merchant'], 0, get_max_period(x['fft']))

# COMMAND ----------

# MAGIC %md
# MAGIC Using in memory `hyperopts` in lieu of Spark Trials (distributed version in Databricks ML runtimes), we can easily apply such a logic for each customer, in parallel, through the use of `@pandas_udf` functions, returning the optimized payment date for each merchant that minimize transaction variance.

# COMMAND ----------

@pandas_udf(optimized_struct, PandasUDFType.GROUPED_MAP)
def optimize_payments(group, pdf):
  
  # grouped by customer Id, our dataframe contains all individual periodic signals
  customer_id = group[0]
  
  # Create our search space where we allow each payment to be made any day during their billing cycle
  space = dict(zip(pdf['merchant'], pdf.apply(get_period_grid, axis=1)))

  # optimize payments using hyperopts
  best = optimize_payments_pandas(pdf, space)

  # append our original dataframe with suggested shift
  pdf['shift'] = pdf['merchant'].apply(lambda x: int(best[x]))
  
  # return optimized payment dataframe
  return pdf[['customer', 'merchant', 'shift']]

# COMMAND ----------

optimized_periodicity_df = periodicity_df.withColumnRenamed('shift', 'old_shift').join(
  periodicity_df.groupBy('customer').apply(optimize_payments), 
  ['customer', 'merchant']
).cache()

# COMMAND ----------

display(optimized_periodicity_df.select('customer', 'merchant', 'date', 'old_shift', 'shift'))

# COMMAND ----------

optimized_signals_df = recompose_signal(optimized_periodicity_df, 365)
optimized_signals_agg_df = aggregate_signal(optimized_signals_df)

# COMMAND ----------

# MAGIC %md
# MAGIC As represented below, our original regular payments data was optimized in a way to reduces financial vulnerabilities by better spreading regular payments over time (all within their allowed billing cycles). In this example representing Mrs. Perritt's regular transactions, we still observe the same average balance of $700, but exhibits much less swings and a transactional behavior that becomes more defensive towards unexpected events. 

# COMMAND ----------

from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)

# given a customer, we return all aggregated regular payments
xs = optimized_signals_agg_df.filter(F.col('customer') == customer_id).limit(1).toPandas().iloc[0]
ys = recomposed_signal_agg_df.filter(F.col('customer') == customer_id).limit(1).toPandas().iloc[0]

# modeled regular payments
xs_agg_df = [[xs['date'] + timedelta(days=i), x] for i, x in enumerate(xs['recomposed'])]
xs_agg_df = pd.DataFrame(xs_agg_df, columns=['date', 'transaction'])

ys_agg_df = [[ys['date'] + timedelta(days=i), x] for i, x in enumerate(ys['recomposed'])]
ys_agg_df = pd.DataFrame(ys_agg_df, columns=['date', 'transaction'])

ax[1].plot(xs_agg_df['date'], xs_agg_df['transaction'], lw=0.5, color='red', label='Optimized payments')
ax[1].plot(ys_agg_df['date'], ys_agg_df['transaction'], lw=0.4, color='orange', label='Original composite')
ax[1].axhline(np.average(xs['recomposed']), lw=0.2, ls='--', label='Average')
ax[1].set_ylabel("regularity of payments [$]")
ax[1].legend(loc='upper left')

# modeled individual payments
for i, r in optimized_signals_df.filter(F.col('customer') == customer_id).toPandas().iterrows():
  xs_ind_df = [[r['date'] + timedelta(days=i), x] for i, x in enumerate(r['recomposed'])]
  xs_ind_df = pd.DataFrame(xs_ind_df, columns=['date', 'transaction'])
  ax[0].plot(xs_ind_df['date'], xs_ind_df['transaction'], lw=0.5, label='Payments to {}'.format(r['merchant']))
ax[0].set_ylabel("regularity of individual payments [$]")
ax[0].set_xlabel("transaction date")
ax[0].legend(loc='upper left')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Self serve analytics
# MAGIC Finally, we can consolidate our findings in a delta table that we can push dowstream to different channels for financial advisory services or mobile banking applications for self serve customer insights. From an operation standpoint, retail banks can easily ingest card transactions data on Delta Live Table, monitor real time customer balances against regular pattern as early indicator to possible overdraft. 

# COMMAND ----------

optimized_struct = ArrayType(StructType(
  [
    StructField('payment_date_original', DateType(), True),
    StructField('payment_date_suggested', DateType(), True),
    StructField('payment_amount', DoubleType(), True),
    StructField('payment_frequency', StringType(), True),
  ]
))

@udf(optimized_struct)
def get_optimized_date(first_date, shift, old_shift, ffts):
  
  # let's ignore negative periodicity
  ffts = [fft for fft in ffts if fft.freq >= 0]
  
  # by sorting our array by frequency, we access frequency zero first
  ffts.sort(key=lambda x: x['freq'])
  
  # that gives us the average amount
  amount = ffts[0]['real']/ffts[0]['size']
  
  periods = []
  for fft in ffts[1:]:
    
    # retrieve payment days (original and suggested)
    period = int(fft.size / fft.freq)
    first_payment_orig = (first_date + timedelta(days=old_shift))
    first_payment = (first_payment_orig + timedelta(days=shift))
      
    # compute expected amount from fft coefficients
    # at t=0 (transaction happens), sine is null, cosine is 1
    # the amount is simply the average (frequency zero) plus normalized real number
    amount = round(amount + fft['real']/fft['size'], 2)
  
    # reporting that suggested payment
    periods.append([first_payment_orig, first_payment, amount, closest_enum(period)])
  
  return periods

# COMMAND ----------

display(
  optimized_periodicity_df.select(
    F.col('customer'),
    F.col('merchant'),
    F.explode(get_optimized_date(F.col('date'), F.col('shift'), F.col('old_shift'), F.col('fft'))).alias('payment')
  ).select(
    F.col('customer'),
    F.col('merchant'),
    F.col('payment.payment_date_original'),
    F.col('payment.payment_date_suggested'),
    F.col('payment.payment_frequency'),
    F.col('payment.payment_amount')
  )
  .filter(F.col('customer') == customer_id)
  .filter(F.col('payment_frequency') != '')
)

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we demonstrated a novel approach to consumer analytics. Using physics theory coupled with scalable computing and AI best practices on a same platform, retail banks can better model customer transactional behaviors in real time, detect payment regularity and provide each of their customers with financial advices and personalized insights, resulting in a more inclusive and human approach to retail banking, all powered by the Lakehouse for Financial Services
