#-------------------------------------------------------------------------------
# Name:      UTIL_LIB.py 
# style:     functional
# Purpose:   A library for common behaviors, extracted from VPS_LIB
# Author:    mthornton
#
# Created:   2015 AUG 01
# Updates:   2017 FEB 20
# Copyright: (c) michael thornton 2015, 2016, 2017
#-------------------------------------------------------------------------------


from functools import partial
from itertools import accumulate
from tkinter import *
from toolz import compose, partial
import re 
import time
#import tkMessageBox

from functionalLIB import take, drop, iterate


def cleanUpStringImperitive(messyString):
    cleanString = messyString.replace(' ' , '') # remove any spaces
    cleanString = cleanString.replace('"' , '') # remove any double quotes
    cleanString = cleanString.replace('\t' , '') # remove any tabs
    cleanString = cleanString.replace(',' , '\n') # replace comma with \n
    for n in range(10):            # replace multiple newlines with a single \n
        cleanString = cleanString.replace('\n\n' , '\n')
    return cleanString

def noSpaces(s):
    return s.replace(' ' , '') # remove any spaces

def noDoubleQuotes(s):
    return s.replace('"' , '') # remove any double quotes

def noTabs(s):
    return s.replace('\t' , '') # remove any tabs

def commaToNewline(s):
    return s.replace(',' , '\n') # replace comma with \n


def noDoubleNewlines(s):
    ''' finds any sequence of newline chars, and replaces the group with single newline '''
    def singleNewline(inString):
        found = multipleNewlinePattern.search(s)
        result = inString[:found.start()] + '\n' + inString[found.end():]
        return result
    multipleNewlinePattern =re.compile('\n{2,}', flags=re.MULTILINE)
    n = len(multipleNewlinePattern.findall(s))
    if n == 0: 
        return s
    else: 
        return take(n+1, iterate(singleNewline, s))
    
def cleanUpStringFunctional(messyString):
    ''' a functional form of 'cleanUpString' '''
    return compose(noDoubleNewlines, commaToNewline, noTabs, noDoubleQuotes, noSpaces)(messyString)

def test_CleanUpStrings():
    mess1 = "01234 56789"
    mess2 = '"air quotes"'
    mess3 = "tab\ttab"
    mess4 = "comma,comma"
    mess5 = "multiple\n\n new\n\nlines"
    
    print(mess1, mess2, mess3, mess4, mess5)
    print('IMPERITIVE')
    print(cleanUpStringImperitive(mess1),
          cleanUpStringImperitive(mess2),
          cleanUpStringImperitive(mess3),
          cleanUpStringImperitive(mess4),
          cleanUpStringImperitive(mess5))
    print('FUNCTIONAL')
    print(cleanUpStringFunctional(mess1),
          cleanUpStringFunctional(mess2),
          cleanUpStringFunctional(mess3),
          cleanUpStringFunctional(mess4),
          cleanUpStringFunctional(mess5))
    

def loadRegExPatterns():
    return dict(
        linePattern=re.compile('^.+'),
        wordPattern=re.compile('\w+'),
        numCommaPattern=re.compile('[0-9,]+'),
        csvPattern=re.compile('[A-Z0-9 .#&]*,'),
        commaToEOLpattern=re.compile(',[A-Z0-9 .#&]+$'),
        LICpattern=re.compile('^LIC '),
        issuedPattern=re.compile('ISSUED '),
        reg_dtPattern=re.compile('REG DT '),
        datePattern=re.compile('[0-9]{2,2}/[0-9]{2,2}/[0-9]{4,4}'),
        dateYearFirstPattern=re.compile(r'\d{4,4}/\d{2,2}/\d{2,2}')
    )


def parseString(inputString,indexPattern, targetPattern, segment="all"): # segment may be start, end, or all
    # the iterator is used to search for all possible target pattern instances
    found = indexPattern.search(inputString)
    if found != None:
        indexStart = found.start()
        indexEnd = found.end()
        #print "parseString: found start", indexStart #debug statement
        iterator = targetPattern.finditer(inputString)
        for found in iterator:
            if found.start() > indexStart and found != None:
                targetStart = found.start()
                targetEnd = found.end()
                #print "parseString: found end", targetStart #debug statement
                return inputString[indexEnd:targetEnd:]
    return None


def timeout(msg="Took too much time!"):
    print(msg)


def waitForUser(msg="enter login credentials"):
    #Wait for user input
    root = Tk()
    tkMessageBox.askokcancel(message=msg)
    root.destroy()


if __name__ == '__main__':
    
    test_CleanUpStrings()