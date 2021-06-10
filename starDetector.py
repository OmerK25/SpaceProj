import numpy as np
import cv2
import math
import pandas as pd


class Star:
    def __init__(self, x, y, r, name, ra, dec):
        self.x = x
        self.y = y
        self.r = r
        self.name = name
        self.ra = ra
        self.dec = dec

    def __str__(self):
        return "X : ", self.x, " | Y :", self.y, " | R: ", self.r, " | NAME :", self.name, " | RA : ", self.ra, " | dec : ", self.dec


class Triangle:

    def __init__(self, stars):
        self.stars = stars

    def getDistances(self):
        distances = []
        for i in range(len(self.stars)):
            for j in range(i + 1, len(self.stars)):
                x = float(self.stars[i].x) - float(self.stars[j].x)
                y = float(self.stars[i].y) - float(self.stars[j].y)
                d = math.sqrt(x * x + y * y)
                distances.append(d)
        return distances

    def getAngles(self):
        angles = []
        dis = self.getDistances()  # AB, AC, BC
        a = (math.pow(dis[2], 2))
        b = (math.pow(dis[1], 2))
        c = (math.pow(dis[0], 2))
        alpha = math.acos((b + c - a) / (2 * math.sqrt(b) * math.sqrt(c)))
        betta = math.acos((a + c - b) / (2 * math.sqrt(a) * math.sqrt(c)))
        gamma = math.acos((a + b - c) / (2 * math.sqrt(a) * math.sqrt(b)))

        # Converting to degree
        alpha = alpha * 180 / math.pi
        betta = betta * 180 / math.pi
        gamma = gamma * 180 / math.pi

        angles.append(alpha)
        angles.append(betta)
        angles.append(gamma)
        return angles

    def __str__(self):
        return str(self.getDistances())


df = pd.read_csv("hygdata_v3.csv")
# df.head(20) # show table
# df.isnull().sum() # count null in each cols for tables
df[['proper']] = df[['proper']].fillna(value="unknown")
starsFromCatalog = []
print(len(df))
for i in range(len(df)):
    radius = df.iloc[i].loc['mag']

    if radius < 3:
        name = df.iloc[i].loc['proper']
        x = df.iloc[i].loc['x']
        y = df.iloc[i].loc['y']
        ra = df.iloc[i].loc['ra']
        dec = df.iloc[i].loc['dec']
        starsFromCatalog.append(Star(x, y, radius, name, ra, dec))
print(len(starsFromCatalog))

def toString(x, y, radius, name):
    print("location: (", x, ",", y, ")", " radius: ", radius, " Starname: ", name)


# for i in range(len(starsFromCatalog)):
#   x = starsFromCatalog[i].x
#   y = starsFromCatalog[i].y
#   radius = starsFromCatalog[i].r
#   name = starsFromCatalog[i].name
#   toString(x, y, radius, name)

# splitting a vector of stars into all possible triples.

def find_all_triplets(starsVec):
    allTriples = []
    for i in range(0, len(starsVec) - 2):
        for j in range(i + 1, len(starsVec) - 1):
            for k in range(j + 1, len(starsVec)):

                st1 = [starsVec[i], starsVec[j], starsVec[k]]
                T1 = Triangle(st1)
                allTriples.append(T1)

    return allTriples


def takeRadius(elem):
    return float(elem[2])


def takeX(elem):
    return float(elem[0])


TriplesFromCatalog = find_all_triplets(starsFromCatalog)
print("catalog end")
# for i in range(len(TriplesFromCatalog)):
#     print(i)

file_name = 'new.jpg'
image = cv2.imread(file_name)
orig = image.copy()
imgheight, imgwidth = image.shape[:2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image, 70, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []

res = []
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

    if (radius > 1 and radius < 5):
        cv2.circle(orig, center, radius + 4, (0, 255, 255), 5)
        res.append([x, y, radius])

connectivity = 4

# cv2.imshow("Naive", orig)
# cv2.imwrite("%s_processed.jpg" % file_name, orig)

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

# print(final_res)  # list of all the star locations.

# Creating an array of all the stars in the picture
allStars = []
for i in range(0, len(final_res)):
    s = Star(final_res[i][0], final_res[i][1], final_res[i][2], "unknown", "noRA", "noDec")
    allStars.append(s)

allTriplesFromFrame = find_all_triplets(allStars)
print("end from frame")


# for i in range(0, len(allTriples)):
#     print(i)

# print(.TgetDistances())
# print(T.getAngles(), " : ANGLES")

# TrianglesFromCatalog
# allTriplesFromFrame

def RMS(frameTripletDistances, catalogTripletDistances):
    return np.sqrt(
        np.sum(np.square(catalogTripletDistances[:3] - frameTripletDistances)) / len(catalogTripletDistances[:3]))


def AD(star1, star2):
    return math.sin(star1.dec) * math.sin(star2.dec) + math.cos(star1.dec) * math.cos(star2.dec) * math.cos(
        star1.ra - star2.ra)


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
    S = 12000000
    for a in bsc:
        bscDistances.append(
            [S * AD((a.stars)[0], (a.stars)[1]), S * AD((a.stars)[1], (a.stars)[2]), S * AD((a.stars)[0], (a.stars)[2])])
    np.sort(np.asarray(bscDistances))
    print("end sort")
    # let bscDistances <dsi, dsj, dst> be the distances from each two stars(angular)

    starMatch = []
    for a in f:
        triangle = []
        minD = 999999999
        i = 0
        for b in bscDistances:
            if minD > RMS(np.sort(np.asarray(a.getDistances())), np.asarray(b)):
                triangle.append(bsc[i])
                minD = RMS(np.sort(np.asarray(a.getDistances())), np.asarray(b))
            i += 1
            print(i)
        starMatch.append(triangle[len(triangle)-1])

    return starMatch


print(BFalgorithm(allTriplesFromFrame, TriplesFromCatalog))
