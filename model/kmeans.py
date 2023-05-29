import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


class DistributionModifier:

    def __init__(self, path:str, k:int = 5) -> None:
        self.path = path
        self.num_team = k

    def dataloader(self):

        data = pd.read_csv(self.path)
        y = data.iloc[:, 0].values
        x = data.iloc[:, 1].values
        self.samples = np.column_stack((x, y))
        return x , y, self.samples

    def making_plot_from_raw_data(self):

        x, y, samples = self.dataloader()
        plt.scatter(x, y)
        plt.xlabel('Feature 2')
        plt.ylabel('Feature 1')
        plt.title('Distribution of Samples')
        plt.show()

    def initialize_centroids(self):
        self.centroids = self.samples[np.random.choice(range(len(self.samples)), self.k, replace=False)]