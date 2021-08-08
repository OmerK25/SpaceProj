import numpy as np
import cv2
import math
import pandas as pd
from PyAstronomy import pyasl

class PixStar:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
       
    def __str__(self):
        return (print("X : ", self.x, " | Y :", self.y, " | R: ", self.r))

class Star:
    def __init__(self, name, mag,ra,dec):
        self.name = name
        self.mag = mag
        self.ra = ra    
        self.dec = dec

class PixTriangle:
    def __init__(self, stars):
        self.stars = stars

    def getLoc(self):
        Locs = []
        Locs.append(" X :"+str(self.stars[0].x)+", Y :"+str(self.stars[0].y))
        Locs.append(" X :"+str(self.stars[1].x)+", Y :"+str(self.stars[1].y))
        Locs.append(" X :"+str(self.stars[2].x)+", Y :"+str(self.stars[2].y))
        return Locs

    def getDistances(self):
        distances = []
        for i in range(len(self.stars)):
            for j in range(i + 1, len(self.stars)):
                x = float(self.stars[i].x) - float(self.stars[j].x)
                y = float(self.stars[i].y) - float(self.stars[j].y)
                d = math.sqrt(x * x + y * y)
                distances.append(d)
        return distances


    def __str__(self):
        return (str(self.getLoc()))

class CatalogTriangle:
    def __init__(self, stars):
        self.stars = stars

    def getNames(self):
        names = []
        names.append(str(self.stars[0].name))
        names.append(str(self.stars[1].name))
        names.append(str(self.stars[2].name))
        return names

    def __str__(self):
        return (str(self.getNames()))


df = pd.read_csv("/content/cassipioa.csv")
df[['proper']] = df[['proper']].fillna(value="unknown")
starsFromCatalog = []
for i in range(len(df)):
    mag = df.iloc[i].loc['mag']

    if mag < 3:
        name = df.iloc[i].loc['proper']
        ra = df.iloc[i].loc['ra']
        dec = df.iloc[i].loc['dec']
        starsFromCatalog.append(Star(name,mag, ra, dec))

# splitting a vector of stars into all possible triples.

def find_all_triplets(starsVec,type):
    allTriples = []
    for i in range(0, len(starsVec) - 2):
        for j in range(i + 1, len(starsVec) - 1):
            for k in range(j + 1, len(starsVec)):

                sta = [starsVec[i], starsVec[j], starsVec[k]]
                if(type == "Pix"):
                    T = PixTriangle(sta)
                else:
                    T = CatalogTriangle(sta)
                allTriples.append(T)

    return allTriples


def takeRadius(elem):
    return float(elem[2])


def takeX(elem):
    return float(elem[0])


TriplesFromCatalog = find_all_triplets(starsFromCatalog,"Cat")

file_name = '/content/star5.jpeg'
image = cv2.imread(file_name)
orig = image.copy()
imgheight, imgwidth = image.shape[:2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image, 70, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []

res = []
dupes = []
for contour in contours:

    area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)

    center = (int(x), int(y))
    radius = int(radius)
    xint = center[0]
    yint = center[1]
    x = str(x)
    y = str(y)
    r = str(radius)
    if (radius > 0.7 and radius < 3):
        if(not center in dupes):
            dupes.append(center)
            cv2.circle(orig, center, radius + 4, (0, 255, 255), 5)
            cv2.putText(orig, ("X :"+str(center[0])+" Y: "+str(center[1])), (center[0]-50,center[1]+40), cv2.FONT_ITALIC, 0.5, (255, 255, 255))
            res.append([x, y, radius])
connectivity = 4

#cv2.imshow("Naive", orig)
cv2.imwrite("%s_processed.jpg" % file_name, orig)

res.sort(key=takeX, reverse=False)

# remove duplicates
final_res = []
for i in range(len(res)):
    dup = False
    for j in range(i + 1, len(res)):
        if (abs(float(res[i][0]) - float(res[j][0])) <= 0.2 and abs(float(res[i][1]) - float(res[j][1])) <= 0.2):
            dup = True
    if (not dup):
        final_res.append(res[i])

# Creating an array of all the stars in the picture
allStarsFromFrame = []
for i in range(0, len(final_res)):
    s = PixStar(final_res[i][0], final_res[i][1], final_res[i][2])
    allStarsFromFrame.append(s)

allTriplesFromFrame = find_all_triplets(allStarsFromFrame,"Pix")

def RMS(frameTripletDistances, catalogTripletDistances):
    return np.sqrt(
        np.sum(np.square(catalogTripletDistances[:3] - frameTripletDistances)) / len(catalogTripletDistances[:3]))

def AD(star1, star2):
    return pyasl.getAngDist(star1.ra,star1.dec,star2.ra,star2.dec)

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
    S = 1200/80
    for a in bsc:
        bscDistances.append(
            np.sort([S * AD((a.stars)[0], (a.stars)[1]), S * AD((a.stars)[1], (a.stars)[2]), S * AD((a.stars)[0], (a.stars)[2])]))
    # np.sort(np.asarray(bscDistances))
    # print(bscDistances)
    # let bscDistances <dsi, dsj, dst> be the distances from each two stars(angular)

    starMatch = []
    for a in f:
        triangle = []
        minD = 9999
        pos = 0
        for b in bscDistances:
            if minD > RMS(np.sort(np.asarray(a.getDistances())), np.asarray(b)):
                triangle.clear()
                triangle.append(bsc[pos])
                minD = RMS(np.sort(np.asarray(a.getDistances())), np.asarray(b))
            pos = pos+1
        match = "Triangle From Frame: "+str(a)+" \n Match Triangle From Catalog: "+str(triangle[0])
        if not match in starMatch:
            starMatch.append(match)

    return starMatch
ans = BFalgorithm(allTriplesFromFrame, TriplesFromCatalog)
for i in range(len(ans)):
    T = ans[i]
    print(T)
