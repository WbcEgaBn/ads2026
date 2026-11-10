import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('WineQT.csv')

# for i in range(len(df)):
#     plt.title(df.columns[i])
#     plt.hist(df.iloc[i])
#     plt.xlabel("amount")
#     plt.ylabel("frequency")
#     plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot= True,
    cmap = 'coolwarm'
)
plt.show()
#aaaaaa
