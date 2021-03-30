# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:11:46 2021

@author: mbeni
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.random.randn(1000)

#plt.scatter(x, np.ones_like(x))


hist, _ = np.histogram(x, 100)


plt.plot(np.linspace(x.min(), x.max(), 100), hist)
plt.show()

#%%

from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d


f = gaussian_filter1d(hist, 10)
#%%
plt.plot(f)

