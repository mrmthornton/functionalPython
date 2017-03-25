'''
Created on Mar 24, 2017

@author: mike
'''

from functools import partial
from itertools import accumulate, count
from operator import iadd


from functionalLIB import iterate


def countsUpFrom(n=0):
    return iterate(partial(iadd, 1), n)

def test_countsUpFrom():
    counter = countsUpFrom(-4)
    print(next(counter), next(counter),next(counter))
    [print(n) for n in counter if n < 10]


if __name__ == '__main__':
    
    test_countsUpFrom()
    

        