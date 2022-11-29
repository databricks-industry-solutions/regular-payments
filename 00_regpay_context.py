# Databricks notebook source
# MAGIC %md You may find this accelerator at https://github.com/databricks-industry-solutions/regular-payments. Please clone this repo to run, instead of using the downloaded DBC file.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">
# MAGIC 
# MAGIC [![DBR](https://img.shields.io/badge/DBR-11.3ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/11.3ml.html)
# MAGIC [![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC [![POC](https://img.shields.io/badge/POC-5_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC 
# MAGIC *Using signal processing theory and Fourier transforms, we extract regular payment informations from a large dataset of card transactions data. Although it is easy to eye ball regularity in payments when looking at specific transactions, doing so at scale across billions of card transactions requires a scientific (and programmatic) approach to a business problem. In this solution accelerator, we demonstrate a novel approach to consumer analytics by combining core mathematical concepts with engineering best practices and state of the art optimizations techniques to better model customers' behaviors and provide millions of customers with personalized insights. With 40% of americans struggling to come up with $400 for an unexpected expense [[source](https://www.cnbc.com/2019/07/20/heres-why-so-many-americans-cant-handle-a-400-unexpected-expense.html)], such a framework could be used to suggest financial goals and provide customers with recommended actions to better spread regular payments over billing cycles, minimize periods of financial vulnerability and better plan for unexpected events.*
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://raw.githubusercontent.com/databricks-industry-solutions/reg-reporting/main/images/reference_architecture.png width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
