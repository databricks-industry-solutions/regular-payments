# Databricks notebook source
config = {
  'timeseries_minimum'        :  5,
  'periodicity_max'           :  60,
  'optimization_runs'         :  500,
}

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import mlflow
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/regpay_experiment"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# Where we might stored temporary data on local disk
from pathlib import Path
temp_directory = "/tmp/{}/regpay".format(useremail)
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

def tear_down():
  import shutil
  shutil.rmtree(temp_directory)
