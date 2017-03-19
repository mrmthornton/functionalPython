'''
Created on Mar 19, 2017

@author: mike
'''

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np


def animationFrame(nframe):
    point = [(x[0],x[1]) for x in np.random.random((1,2))]
    plt.plot( point[0], point[1], color='r', marker='*')
    plt.show()
    
#points = [(x[0],x[1]) for x in np.random.random((100,2))]

fig = plt.figure('test plots', figsize=(5,4))
#[plt.plot( point[0], point[1], color='r', marker='*') for point in points]
#plt.show()
anim = animation.FuncAnimation(fig, animationFrame, frames=100)

