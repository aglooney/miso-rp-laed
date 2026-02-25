import matplotlib.pyplot as plt
from pyomo.environ import DataPortal

case_name = 'toy_data.dat'
data = DataPortal()
data.load(filename=case_name)
plt.figure()
original_demand = list(data.data()["Load"].values())
print(original_demand)
plt.plot(range(len(original_demand)), original_demand)
plt.show()