"""
Class containing utilities.
"""

# -----------------------------------------------------------------------

import time
from math import sqrt

# -----------------------------------------------------------------------

TIMING_INFO = True

# -----------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# timing functions
#-----------------------------------------------------------------------------------------

TIMINGS = {}

def timethis(method):
    """
    Use as a decorator on methods to print timing information for that method.
    """
    def timed(*args, **kw):
        if TIMING_INFO:
            ts = time.time()
        result = method(*args, **kw)
        if TIMING_INFO:
            te = time.time()
            if method.__name__ not in TIMINGS.keys():
                TIMINGS[method.__qualname__] = [te-ts]
            else:
                TIMINGS[method.__qualname__].append(te-ts)
        return result
    return timed

def showTiming():
    """
    Print all timing info stored in the TIMINGS dict.
    """
    if TIMING_INFO:
        print("Showing average timing info for method instances")
        for k, v in TIMINGS.items():
            print("{3}: {0:.2f} (sigma={1:.2f}, total={2:.2f})".format(mean(v), stdEstimate(v), sum(v), k))

#-----------------------------------------------------------------------------------------
# maths functions
#-----------------------------------------------------------------------------------------

def mean(x):
    """
    Return standard mean.
    """
    return sum(x) / (len(x)+0.0)

def stdEstimate(x):
    """
    Return standard deviation.
    """
    meanx = mean(x)
    norm = 1./(len(x)+0.0)
    y = []
    for v in x:
        y.append( (v - meanx)*(v - meanx) )
    return sqrt(norm * sum(y))
