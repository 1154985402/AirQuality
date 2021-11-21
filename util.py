import matplotlib
import  matplotlib.pyplot as plt
from scipy import  stats


def plot_data(x, y1, y2):
    plt.figure(figsize=(10,10))
    plt.title("epoch-loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x , y1 , color = 'r', linewidth = 3, label = "train_loss" )
    plt.plot(x,  y2 , color = 'b', linewidth = 3, label = 'test_loss' )
    plt.legend()
    plt.show()


def plot_relation(y1, y2  ):
    plt.figure(figsize=(10,10))
    print(y1)
    print(y2)
    plt.title("relation between predict and real")
    r , p = stats.pearsonr(y1,y2)
    plt.xlabel("y_real\n correlation coefficient is  %0.3f ,  pearsonr num is %0.3f "%(r , p))
    plt.ylabel("y_hat")
    plt.scatter(y1, y2, c = 'red')
    plt.plot(y1, y1 ,color = 'b')
    plt.show()


def plot_data_fit(x, y):
    plt.figure(figsize=(10,10))
    plt.title("GA epoch-fit")
    plt.xlabel('epoch')
    plt.ylabel('minfit')
    plt.plot(x , y , color = 'r', linewidth = 3, label = "train_loss" )
    plt.show()
