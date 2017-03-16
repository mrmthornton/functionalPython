'''
Created on Mar 12, 2017
modified: MAR 15 2017

@author: mike
'''

#
#Joel Grus: Learning Data Science using functional python
#

from functools import reduce, partial #aka currying
from operator import iadd
from itertools import count, islice, tee, repeat, cycle, chain
from itertools import accumulate #(p, [func=add])  # python 3 ?


# iteration with next
lst = [1,2,3] ; it = iter(lst) ; nxt = next(it)


def iterate(f, x):
    "iterate  - results in sequence x f(x) f(f(x)) ..."
    #yield x
    #yield from iterate(f, f(x)) #python 3 syntax
    return accumulate(repeat(x), lambda fx, _: f(fx) )

#generator, a function with a yield
def lazy_integers_old(n=0):
    while True:
        yield n
        n +=1
#x = lazy_integers_old()
#[next(x) for _ in range(10)]
add1 = partial(iadd, 1)
print(add1(2))
def lazy_intergers(n=0):
    return iterate(add1, n)

# generator comprehension
squares = (x*x for x in lazy_intergers()) # use () not [] 
print(next(squares))
print(next(squares))

# generators and piplines   
# the equivilent of    cat filename | grep -i someword | wc -l 
with open("xmlFromNotice", "r") as f:
    lazyLines = (line for line in f)
    matchedLines = filter(lambda line: "date" in line.lower(), lazyLines ) # with lambda functins, 
                                                                    # if predicate then value (not bool)
    lineCount = len(list(matchedLines))  # all the evaluation happens here.                                                                
    print(lineCount)


def take(n, it):
    return [x for x in islice(it, n)]

def drop(n, it):
    return islice(it, n, None)


#force the first value of a sequence
head = next
# new sequence with all but the first value of a sequence
tail = partial(drop,1)


    
def fibs():
    def next_pair(pair):
        x,y = pair
        return (y, x+y)
    return (y for x,y in iterate (next_pair, (0,1) ) )
    
    
#gradient descent 
def f(x_i):
    return sum(x_ij**2 for x_ij in x_i)
def df(x_i):
    return [2*x_ij for x_ij in x_i]
def gradientStep(df, alpha, x_i):
    return [x_ij + alpha *df_j for x_ij, df_j in zip(x_i, df(x_i))]
# can curry df and alpha 
def gradientDescent(df, x_0, alpha=0.1):
    return iterate( partial(gradientStep, df, -alpha), x_0) 

#matplotlib animation 
##from matplotlib import animation
##def animationFrame(nframe):
##    plt.cla()
##    x,y = getdataFor(nframe)
##    plt.plot(x,y)
##fig = plt.figure(figsize=(5,4))
##amim = animation.FuncAnimation(fig, animationFrame, frames=numFrames)
##anim.save("animation.gif", write='imagemagick', fps=4)

# Stochastic Gradeint DESCENT