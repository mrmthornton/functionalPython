'''
Created on Mar 18, 2017
modified: MAR 18 2017

@author: mike
'''

#
#Joel Grus: Learning Data Science using functional python
#
#

from functools import partial 
from operator import iadd
from itertools import count, islice, repeat, accumulate #(p, [func=add])


def iterate(f, x):
    "iterate  - results in sequence x f(x) f(f(x)) ..."
    return accumulate(repeat(x), lambda fx, _: f(fx) )


#generator, a function with a yield
def lazy_integers_old(n=0):
    while True:
        yield n
        n +=1


def lazy_intergers():
    add1 = partial(iadd, 1)
    return iterate(add1, 0)


def take(n, it):
    return [x for x in islice(it, n)]


def drop(n, it):
    return [x for x in islice(it, n, None)]


def fibs():
    def next_pair(pair):
        a,b = pair
        return (b, a+b)
    return (y for x,y in iterate(next_pair, (0,1)))


def all_primes():
    def filter_primes(it):
        '''will blow the stack'''
        p = next(it)
        yield p
        yield from filter_primes(filter(lambda x: x%p > 0, it))
    return filter_primes(count(2))
