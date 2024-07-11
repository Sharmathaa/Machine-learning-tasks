# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:03:13 2024

@author: DELL LYF
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import binom
#1
exp = 15
pmf=poisson.pmf(exp,10)
print(pmf)
plt.vlines(exp,0,pmf,colors='green')
plt.xlabel('values')
plt.ylabel('probablity')
plt.show()

#2
pmf_poisson = poisson.pmf(5,10)
pmf_poisson
cdf_poisson = poisson.cdf(5,mu=10)
cdf_poisson
perc_emp = pmf_poisson+cdf_poisson
print("percentage of employee having 5 or less than 5 years of experience is:",perc_emp*100 )





#3 

exp = 200
pmf = poisson.pmf(exp,100)
print(pmf)
plt.vlines(exp,0,pmf,colors='k')
plt.ylabel('probablity')
plt.xlabel('intervals')


#4 
report_card = pd.DataFrame({'subject':['maths','sci','socio','eng','french'],
                            'Marks':[93,87,98,89,91]})
report_card

z = np.array(report_card['Marks'])
z
z.sort()
print('Mean:',np.mean(z))
print('Meadian:',np.median(z))
print('Standard deviation:',np.std(z))
print('Variance:',np.var(z))

from scipy import stats
mode= stats.mode(z,keepdims=False)
print(mode)











