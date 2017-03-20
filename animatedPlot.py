'''
Created on Mar 12, 2017
modified: MAR 15 2017

@author: mike
'''

#
#Joel Grus: Learning Data Science using functional python
#github.com/joelgrus/stupid-itertools-tricks-pydata
#
#curl -u 'USER' https://api.github.com/user/repos -d '{"name":"REPO"}'
#

from functools import reduce, partial #aka currying
from operator import iadd, mul
from itertools import count, islice, tee, repeat, cycle, chain
from itertools import accumulate #(p, [func=add])
import random
import numpy as np

from functionalLIB import iterate, take

from sklearn.cluster.k_means_ import KMeans
from sklearn.linear_model.tests.test_ridge import y_iris


#import time
#start_time = time.time()
#print()
#print("--- %s seconds ---" % (time.time() - start_time))


def within_tolerance(tol, prev, curr):
    if abs(prev - curr) < tol:
        raise StopIteration
    else:
        return curr
    
def until_nearly_converges(it, tolerance=0.001):
    return accumulate(it, partial(within_tolerance, tolerance))

def no_repeat(prev, curr):
    if prev == curr:
        raise StopIteration
    else:
        return curr

def until_convergence(it):
    return accumulate(it, no_repeat)

def squared_distances(p, q):
    return sum((p_i - q_i)**2 for p_i, q_i in zip(p,q))

def cluster_mean(points):
    dimensions = len(points[0]) if points else 0
    sum_each_dim = [ sum(point[j] for point in points) for j in range(dimensions)]
    return  [each_dim/len(points) for each_dim in sum_each_dim]

def closest_index(point, means):
    distances = [squared_distances(point, mean) for mean in means]
    return min(enumerate(distances), key=lambda pair: pair[1])[0]
    
def new_means(points, old_means):
    k = len(old_means)
    assignments = [closest_index(point, old_means) for point in points]
    clusters = [[point for point, c in zip(points, assignments) if c==j] for j in range(k)]
    return [cluster_mean(cluster) for cluster in clusters]

def k_meanses(points, k):
    initial_means = random.sample(points, k)
    return iterate(partial(new_means, points), initial_means)

points = [(x[0],x[1]) for x in np.random.random((100,2))]
#model = KMeans(5)
#model.fit(points)
#assignments = [ model.predict(point) for point in points ]

meanses = take(10, k_meanses(points, 5))
#meanses = until_convergence(k_meanses(points, 5))
 
# MATPLOT ANIMATION
from matplotlib import animation
from matplotlib import pyplot as plt


def animationFrame(nframe):
    colors = ['r','g','b','c','m']
    k = len(meanses)
    means = meanses[nframe]
    plt.cla()
    assignments = [closest_index(point, means) for point in points]
    clusters = [[point for point,c in zip(points, assignments) if c==j] for j in range(k)]
    
    for cluster, color, mean in zip(clusters, colors, means):
        x,y = zip(*cluster)
        plt.scatter(x,y,color=color)
        plt.plot(*mean, color=color, marker='*', markesize=10)

numFrames = len(meanses)
#debug#fig = plt.figure(figsize=(5,4))
#debug#anim = animation.FuncAnimation(fig, animationFrame, frames=numFrames)
#debug#anim.save("animation.gif", write='imagemagick', fps=4)


# GRADIENT DESCENT
def gradientStep(df, alpha, e_i):
    return [e_ij + alpha * df_j for e_ij, df_j in zip(e_i, df(e_i))]

# can curry df and alpha 
def gradientDescent(df, initial_element, alpha=0.1):
    return iterate( partial(gradientStep, df, -alpha), initial_element)

#sum of squares example
def f(e_i): 
    return sum(e_ij**2 for e_ij in e_i)
def df(e_i):
    return [2*e_ij for e_ij in e_i]

steps = take(100, gradientDescent(df, [random.random(), random.random()]))[::20]
#[print(step) for step in steps]


# f(x,y) = -exp(-x^3 /3 + x - y^2)
# min at (1,0), saddle point at (-1,0)
from math import exp
def f1(e):
    return -exp(e[0]**3/-3 + e[0] -e[1]**2)
def df1(e):
    return ((1 - e[0]**2) * f1(e), -2*e[1] * f1(e))

steps = take(100, gradientDescent(df1, [2 * random.random() -1, 2 * random.random() -1 ]))[::20]
#[print(step) for step in steps]


# STOCHASTIC Gradient Descent
def sgd_step(df, alpha, previous_beta, xy_i):
    # df is a function of x_i, y_i, beta
    x_i, y_i = xy_i
    return [beta_j + alpha * df_j for beta_j, df_j in zip(previous_beta, df(x_i, y_i, previous_beta))]

def sgd(df, x, y, beta_0, alpha=0.1):
    xys = chain([beta_0], cycle(zip(x, y)))
    return accumulate(xys, partial(sgd_step, df, -alpha))


# example using linear regeression model y = x * Beta + epsilon

def predict(x_i, beta):
    return x_i[0] * beta[0] + x_i[1] + beta[1]

def error(x_i, y_i, beta): 
    return predict(x_i, beta) - y_i

def sqerror(x_i, y_i, beta): 
    return error(x_i, y_i, beta)**2

def sqerror_gradient(x_i, y_i, beta):
    return (2 * x_i[0] * error(x_i, y_i, beta),
            2 * x_i[1] * error(x_i, y_i, beta))


x = [(1, random.randrange(100)) for _ in range(100)]
y = [ (-5 * x_i[0] + 10 * x_i[1] + random.random()) for x_i in x]
beta_0 = (random.random(), random.random())
steps = 20
show_every = 5
results = [x for x in take(steps, sgd(sqerror_gradient, x, y, beta_0, 0.01))]
subresults = results[::show_every]

from matplotlib import animation
from matplotlib import pyplot as plt
nframes = len(subresults)

def animation_frame(nframe):
    a,b = subresults[nframe]
    # regression line goes through (0,a) and (1,a+b)
    plt.plot([0,1],[a, a+b])
    plt.show()

fig = plt.figure("Stochastic", figsize=(5,4))
plt.plot([x[0] for x[0] in x],[x[1] for x[1] in x])
plt.show()
#anim = animation.FuncAnimation(fig, animationFrame, frames=numFrames)

print('DONE')

