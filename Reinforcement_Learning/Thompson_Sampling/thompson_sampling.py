import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

import random
N = 10_000
d = 10
ads_selected = []
numbers_of_reward_1 = [0] * d
numbers_of_reward_0 = [0] * d
total_reward = 0

for n in range(N):
    ad = 0
    max_random = 0
    for i in range(d):
        beta_random = random.betavariate(numbers_of_reward_1[i] + 1, numbers_of_reward_0[i] + 1)
        if beta_random > max_random:
            ad = i
            max_random = beta_random
    ads_selected.append(ad + 1)
    if dataset.iloc[n, ad] == 0:
        numbers_of_reward_0[ad] += 1
    else:
        numbers_of_reward_1[ad] += 1
        total_reward += 1

plt.hist(ads_selected)
plt.title("Histogram of Ads Selections")
plt.xlabel("Ads")
plt.ylabel("Numbers of Ads Selections")
plt.show()
