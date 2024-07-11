# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:04:47 2024

@author: DELL LYF
"""

import numpy as np 
import pandas as pd
from scipy import stats
data = pd.read_csv("D:\Dataset\Salary_Data.csv")
salary = data['Salary']
SD = np.std(salary)
SD
mean = np.mean(salary)
mean
n=len(salary)

z_score = stats.norm.ppf((1+0.90)/2)
margin_err = z_score*(SD/np.sqrt(n))
confidence_interval_90 = (mean - margin_err, mean+margin_err)
print("confidence interval for 90%:",confidence_interval_90)



z_score = stats.norm.ppf((1+0.95)/2)
margin_err = z_score*(SD/np.sqrt(n))
confidence_interval_95 = (mean - margin_err, mean+margin_err)
print("confidence interval for 95%:",confidence_interval_95)


z_score = stats.norm.ppf((1+0.99)/2)
margin_err = z_score*(SD/np.sqrt(n))
confidence_interval_99 = (mean - margin_err, mean+margin_err)
print("confidence interval for 99%:",confidence_interval_99)