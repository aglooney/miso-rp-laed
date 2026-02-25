import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt


fname="toy_data.dat"
data = DataPortal()
data.load(filename=fname)

print(data.data()['N_g'][None])

