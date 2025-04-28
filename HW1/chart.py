import pickle
import matplotlib.pyplot as plt

with open('perps.pickle', 'rb') as f:
    data = pickle.load(f)

plt.plot(data)
plt.title('Graph of List Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
