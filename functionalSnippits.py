'''
Created on Mar 12, 2017
modified: MAR 15 2017

@author: mike
'''

#
#Joel Grus: Learning Data Science using functional python
#
#
#curl -u 'USER' https://api.github.com/user/repos -d '{"name":"REPO"}'
#

from functools import reduce, partial  #aka currying
from operator import iadd, mul
from itertools import count, islice, tee, repeat, cycle, chain
from itertools import accumulate #(p, [func=add])


# iteration with next
lst = [1,2,3] ; it = iter(lst) ; print(next(it),next(it))

# accumulate
#accumulate(seq, [func])
print([a for a in accumulate([1,2,3])])
print( [a for a in accumulate([1,2,3], func=mul)] )
print 

def iterate(f, x):
    "iterate  - results in sequence x f(x) f(f(x)) ..."
    #yield x
    #yield from iterate(f, f(x)) #python 3 syntax blows up stack
    return accumulate(repeat(x), lambda fx, _: f(fx) )

#generator, a function with a yield
def lazy_integers_old(n=0):
    while True:
        yield n
        n +=1
ni = lazy_integers_old()
print([next(ni) for _ in range(10)])  # use of underscore as dummy variable

# partial functions
add1 = partial(iadd, 1)
print(add1(2))

# 
def lazy_intergers():
    return iterate(add1, 0)

# generator comprehension
squares = (x**2 for x in lazy_intergers()) # use () not [] 
print(next(squares), next(squares), next(squares))

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

seq = [10,9,8,7,6,5,4,3,2,1,0]
print( take(5,lazy_intergers()) )

def drop(n, it):
    return [x for x in islice(it, n, None)]

print( drop(4,seq) )
tail = partial(drop, 1)
print( tail(seq) )

#force the first value of a sequence
head = next
# new sequence with all but the first value of a sequence
tail = partial(drop,1)


    
def fibs():
    def next_pair(pair):
        a,b = pair
        return (b, a+b)
    return (y for x,y in iterate(next_pair, (0,1)))

import time
start_time = time.time()
print( take(30,fibs()) )
#take(30,fibs())
print("--- %s seconds ---" % (time.time() - start_time))
    

def filter_primes(it):
    '''will blow the stack'''
    p = next(it)
    yield p
    yield from filter_primes(filter(lambda x: x%p > 0, it))
def all_primes():
    return filter_primes(count(2))

print(take(100,all_primes()))
    
    
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


# Matplotlib Animation 
#from matplotlib import animation
#from matplotlib import pyplot as plt
#def animationFrame(nframe):
#    plt.cla()
#    x,y = getdataFor(nframe)
#    plt.plot(x,y)
#fig = plt.figure(figsize=(5,4))
#amim = animation.FuncAnimation(fig, animationFrame, frames=numFrames)
#anim.save("animation.gif", write='imagemagick', fps=4)

# Stochastic Gradeint DESCENT