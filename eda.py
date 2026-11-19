import pandas as pd
from pathlib import Path
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(Path(__file__).resolve().with_name('wine2.csv'))

# for i in range(len(df)):
#     plt.title(df.columns[i])
#     plt.hist(df.iloc[i])
#     plt.xlabel("amount")
#     plt.ylabel("frequency")
#     plt.show()



corr_matrix = df.drop(columns=['quality', 'wine_type', 'quality_class'], axis=1).corr()

#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot= True,
    cmap = 'coolwarm',
#    mask=mask
)
plt.show()

