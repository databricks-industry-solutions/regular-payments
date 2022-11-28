# Databricks notebook source
# MAGIC %md You may find this accelerator at https://github.com/databricks-industry-solutions/regular-payments. Please clone this repo to run, instead of using the downloaded DBC file.

# COMMAND ----------

# MAGIC %md
# MAGIC # The Theory
# MAGIC Before delving into the actual solution, let's get back to our physics fundamentals. In this notebook, we will explore signal processing theory as it may apply to retail banking and card transaction data. Let's select an hypothetical user for a subscription service we know would exhibit strong seasonality (such as Netflix). Easy to eye ball on a graph, such a payment regularity would be difficult to programmatically extract from billions of card transactions, especially when the regularity of payments may not be of exact match (bank holidays or missed payments). 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://upload.wikimedia.org/wikipedia/commons/d/df/Fourier2_-_restoration1.jpg' width=200>

# COMMAND ----------

from utils.regpay_utils import *

# COMMAND ----------

import pandas as pd
transactions = pd.read_json('''
{
  "amount":
    {
      "1547683200000":9.99,
      "1550361600000":9.99,
      "1552780800000":9.99,
      "1555459200000":9.99,
      "1558051200000":9.99,
      "1560729600000":9.99,
      "1563321600000":9.99,
      "1566000000000":9.99,
      "1568678400000":9.99,
      "1571270400000":9.99,
      "1573948800000":9.99,
      "1576540800000":9.99
    }
}
''')

# COMMAND ----------

# MAGIC %md
# MAGIC Our non continuous signal can be expressed in the form of pulse functions. 

# COMMAND ----------

import matplotlib.pyplot as plt
fig=plt.figure(2, figsize=(15,6))
plt.clf()
plt.stem(transactions.index, transactions.amount)
plt.title("Payments over time")
plt.ylabel("transaction amount [$]")
plt.xlabel("transaction date")

# COMMAND ----------

# MAGIC %md
# MAGIC We can compute fourier transform for our selected time series. Before doing so, let's first create an actual timeseries by merging our data with all available dates.

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np

min_date = np.min(transactions.index)
max_date = np.max(transactions.index)

dates_df = create_calendar_df(min_date, max_date)
transactions_ts = pd.merge(transactions, dates_df, left_index=True, right_index=True, how='right').fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC As represented below, we can extract all frequencies composing our signal. As expected, a series of pulse functions will only be explained by the superposition of an infinite number of frequencies. We know our signal is highly regular, but the superposition of all harmonics makes it hard to extract a dominant frequency

# COMMAND ----------

fft_ts = fourier_transform(transactions_ts.amount.to_list())

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(2, figsize=(15,6))

plt.clf()
plt.plot(fft_ts.frequencies, fft_ts.amplitudes, lw=0.5)
plt.stem(fft_ts.frequencies, fft_ts.amplitudes)
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude [db]")
plt.title("Fourier transform")
plt.xlim([0 , np.max(fft_ts.frequencies)])

# COMMAND ----------

# MAGIC %md
# MAGIC Our assumption is confirmed by litterature, pulse signal in the time domain is only explained by an infinite number of harmonics. 
# MAGIC Using simple utility functions, we can easily convert our pulse signals into a triangular form, maximizing the effectiveness of a fourier transform whilst preserving our signal latent value at most.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://i.stack.imgur.com/qH5Rv.gif'>

# COMMAND ----------

from datetime import timedelta
min_date = np.min(transactions_ts.index)
xs = to_triangle_signal(transactions_ts.amount)
transactions_ts_tri = pd.DataFrame([[min_date + timedelta(days=i), x] for i, x in enumerate(xs)], columns=['date', 'amount'])
transactions_ts_tri = transactions_ts_tri.set_index('date')

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)

ax[0].stem(transactions.index, transactions.amount)
ax[0].set_title("Payments over time")
ax[0].set_ylabel("transaction amount [$]")

ax[1].plot(transactions_ts_tri.index, transactions_ts_tri.amount, lw=0.5, color='red')
ax[1].set_ylabel("triangular representation [$]")
ax[1].set_xlabel("transaction date")

# COMMAND ----------

# MAGIC %md
# MAGIC As represented below, by transforming our signal into a triangular function, fourier transform yields more significant insights, peaking at a dominent frequency at around 10Hz. Thinking in term of Hertz does not offer much practicality. By converting frequencies to periods (in days), we observe our seasonality peak at ~ 30 days, as expected. Payments to Netflix are done every 30 days in average

# COMMAND ----------

# our timeseries contains n records, 1 record per day
# our sampling rate is therefore simply 1/n
n = transactions_ts_tri.shape[0]
sampling_rate = 1/n

# extract fourier frequencies
fft_ts_tri = fourier_transform(transactions_ts_tri.amount.to_list())

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# convert frequencies back to periods, in days
periods = [n / f for f in fft_ts_tri.frequencies[1:]]

fig=plt.figure(2, figsize=(15,6))
plt.clf()
plt.plot(periods, fft_ts_tri.amplitudes[1:], lw=0.5)
plt.stem(periods, fft_ts_tri.amplitudes[1:])
plt.xlabel("period [days]")
plt.ylabel("amplitude [db]")
plt.title("Fourier transform")
plt.xlim([0,np.max(periods)])

# COMMAND ----------

# MAGIC %md
# MAGIC We can extract dominant frequencies through kernel density, just like one would extract anomalies from a given set of data points. Whilst we could leverage anomaly detection techniques like Isolation Forests, it would be wrong to assume each signal exhibit regularity. By specifying a "contamination" parameter, we would be forcing our framework to return anomalies. Instead, we can find out how "packed" our data points are and define a kernel threshold to contains 99% of our possible distances

# COMMAND ----------

threshold = detect_anomalies(fft_ts_tri.amplitudes)
fft_ts_tri_df = fft_ts_tri.filter(threshold).to_df()
display(fft_ts_tri_df[fft_ts_tri_df['freq'] > 0])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's just focus on dominant frequencies. We can easily rebuild our signal as a series of sine waves. 

# COMMAND ----------

def recompose_signal(df, n):
  import numpy as np
  sampling_rate = 1/n
  x = np.arange(0, n, 1)
  transactions_ts_recombined = np.zeros((len(x),))
  for i, f in df.iterrows():
    cos = f['real']*np.cos(f['freq']*2*np.pi*x*sampling_rate)
    sin = f['imag']*np.sin(f['freq']*2*np.pi*x*sampling_rate)
    transactions_ts_recombined += 1/(n)*(cos-sin)
  return transactions_ts_recombined

# COMMAND ----------

transactions_ts_recombined = recompose_signal(fft_ts_tri_df, n)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)

ax[0].stem(transactions.index, transactions.amount)
ax[0].set_title("Payments over time")
ax[0].set_ylabel("transaction amount [$]")

x1 = transactions_ts_tri.index
y1 = transactions_ts_recombined

ax[1].plot(transactions_ts_tri.index, transactions_ts_recombined, lw=0.5, color='green')
ax[1].set_ylabel("modelled transactions [$]")
ax[1].set_xlabel("transaction date")

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we have been able to programmatically extract dominant frequencies for our payments. Summarizing that particular payment as a series of sine waves will allow us to overlay multiple signals that could model someone's financial behaviors. 
