
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setting up markers an pallete
    markers = ('.', '.', '.')
    colors = ('red', 'blue', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1 # min and max of the first traits in X
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1 # same for the second trait
    
    # create arrays of values between min and max of traits in X
    # with size of len(A) x len(B)
    # xx -- X coords, yy -- Y coords
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), # A
        np.arange(x2_min, x2_max, resolution), # B
    )

    # predict class of each value from 1d array of pairs of trait values
    # z -- activation function values after prediction, correspond to coords from xx and yy
    z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    z = z.reshape(xx.shape) # reshapre z into len(A) x len(B) array

    plt.contourf(xx, yy, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # display
    for idx, cl in enumerate(np.unique(y)): # cl ~ class label?
        plt.scatter(
            x = X[y==cl, 0], 
            y = X[y==cl, 1],
            alpha = 0.8,
            c = cmap(idx),
            marker = markers[idx],
            label = cl,
        )