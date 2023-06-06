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
    

    def initialize_centroids(self):
        self.centroids = self.samples[np.random.choice(range(len(self.samples)), self.num_team, replace=False)]

    def assign_classes(self):
        self.classes = np.zeros(len(self.samples))
        for i in range(len(self.samples)):
            distances = np.linalg.norm(self.samples[i] - self.centroids , axis=1)
            self.classes[i] = np.argmin(distances)

    def update_centroids(self):
        for i in range(len(self.centroids)):
            self.centroids[i] = np.mean(self.samples[self.classes== i], axis= 0)
        return self.classes
            
    
    def final_data(self):
        new_file = pd.DataFrame(self.samples)
        new_file['class'] = self.classes.astype(int)
        new_file.columns = ['X' , 'Y' , 'Class']
        with open('dataset/Completed_data.csv', 'w',) as file:
            new_file.to_csv(file , index=False)
            

    
    def making_plot_from_raw_data(self):
        self.dataloader()
        self.initialize_centroids()
        for i in range(40):
            self.assign_classes()
            self.update_centroids()

        self.final_data()
        colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
        for i in range(self.num_team):
            plt.scatter(self.samples[self.classes == i][:, 0], self.samples[self.classes == i][:, 1], color=colors[i])
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='black', s=100)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Distribution of Samples')
        plt.savefig('output/results_imgs.png')
        plt.show()