# To see the result you should run this code with proper args
from model.kmeans import *

if __name__ == "__main__":
    model = DistributionModifier(path='dataset/rc_task_2.csv')
    model.making_plot_from_raw_data()
    print('result will be appeared in this module')
