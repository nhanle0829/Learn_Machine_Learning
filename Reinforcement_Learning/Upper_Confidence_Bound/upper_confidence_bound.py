import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

import math
N = 10_000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_reward = [0] * d
total_reward = 0
for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_reward[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad + 1)
    numbers_of_selections[ad] += 1
    reward = dataset.iloc[n, ad]
    sums_of_reward[ad] += reward
    total_reward += reward

print(total_reward)
plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel("Ads")
plt.ylabel("Number of Ads Selections")
plt.show()
