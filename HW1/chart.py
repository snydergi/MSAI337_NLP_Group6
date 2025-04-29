import pickle
import matplotlib.pyplot as plt

with open('perps.pickle', 'rb') as f:
    data = pickle.load(f)

with open('perpsTest.pickle', 'rb') as f:
    dataTest = pickle.load(f)

# Determine the maximum length
max_len = max(len(data), len(dataTest))

# Create a range of x values starting at 1
x_values = list(range(1, max_len + 1))

plt.plot(x_values[: len(data)], data, label='perps')
plt.plot(x_values[: len(dataTest)], dataTest, label='perpsTest')
plt.title('Perplexities')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.legend()

# Set x-ticks at every whole number starting from 1
plt.xticks(x_values)

# Remove grid
# plt.grid(False) is default, so just don’t call plt.grid(True)

plt.show()
