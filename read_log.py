import matplotlib.pyplot as plt


input_file = r".\weights.txt"

with open(input_file) as f:
    f = f.readlines()

trn_losses = []
weights = []
for line in f:
    if "TEST_LOSS" in line:
        trn_losses.append(float(line.split(':')[-1]))
    if "tensor" in line:
        weights.append(float(line.split(',')[2].replace(']', '')))

fig, ax = plt.subplots()
ax.plot(trn_losses)
plt.show()