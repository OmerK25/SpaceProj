import math

import numpy as np

def BFalgorithm(f, bsc):
    """
    Find match between stars captured in frame and a priori star database.
    :param Frame(f)
    :param starDatabase(bsc)
    :return: The 3 matching stars
    """
    # let <p1, p2, p3> triplet of stars from the frame(p=<x,y>)
    # let <si, sj, st> triplet of stars from the database(p=star)
    # let <dp1, dp2, dp3> be the distances from each two stars(distance in pixels)
    bscDistances = []
    for a in bsc:
        for b in bsc:
            for c in bsc:
                if a == b or a == c or b == c:
                    continue
                S = 0
                bscDistances.append([S*AD(a, b), S*AD(a, c), S*AD(b, c)])

    # let bscDistances <dsi, dsj, dst> be the distances from each two stars(angular)
    starMatching = []
    for a in bscDistances:
        for b in f:
            starMatching.append(RMS(a, b.getDistances()))

    return np.min(starMatching)

def RMS(catalogTripletDistances: list, frameTripletDistances: list):
    return np.sqrt(np.sum(np.square(catalogTripletDistances-frameTripletDistances))/len(catalogTripletDistances))

def AD(star1, star2):
    return math.sin(star1.dec)*math.sin(star2.dec) + math.cos(star1.dec)*math.cos(star2.dec)*math.cos(star1.ra-star2.ra)
