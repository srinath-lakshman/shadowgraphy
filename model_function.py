import numpy as np
import os
import glob
import cv2
import sys
import re
import math
from matplotlib import pyplot as plt

epsilon = 0.1
M = np.arange(0.01,1,0.01)
p = np.power(np.pi/np.log(epsilon),2)

F = (1/2.0)*np.sqrt((p*np.power(M+1,2))+np.power(M-1,2))

D = M/F;

plt.plot(1/F,D,'.')
plt.xlabel('1/F')
plt.ylabel('D')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**(-1),10**(+1))
plt.ylim(10**(-2),10**(0))
plt.show()
