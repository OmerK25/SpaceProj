import numpy as np
import cv2
import math


class Star:
    def __init__(self, x, y, r, name):
        self.x = x
        self.y = y
        self.r = r
        self.name = name

    def printStar(self):
        print("X : ", self.x, "| Y :", self.y,
              "| R : ", self.r, "| NAME : ", self.name)

    def __str__(self):
        return "X : ", self.x, " | Y :", self.y, " | R: ", self.r, " | NAME :", self.name


class Triangle:

    def __init__(self, stars):
        self.stars = stars

    def getDistances(self):
        distances = []
        for i in range(len(self.stars)):
            for j in range(i+1, len(self.stars)):
                x = float(self.stars[i].x)-float(self.stars[j].x)
                y = float(self.stars[i].y)-float(self.stars[j].y)
                d = math.sqrt(x*x+y*y)
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

#splitting a vector of stars into all possible triples.
def find_all_triplets(starsVec):
    allTriples = []
    for i in range(0, len(starsVec)-2):
        for j in range(i+1, len(starsVec)-1):
            for k in range(j + 1, len(starsVec)):
                st = [starsVec[i], starsVec[j], starsVec[k]]
                T = Triangle(st)
                allTriples.append(T)
    return allTriples

def takeRadius(elem):
    return float(elem[2])


def takeX(elem):
    return float(elem[0])


file_name = 'stars1.jpg'
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

    if (radius > 1 and radius < 8):
        cv2.circle(orig, center, radius+4, (0, 255, 255), 5)
        res.append([x, y, radius])

connectivity = 4

# cv2.imshow("Naive", orig)
# cv2.imwrite("%s_processed.jpg" % file_name, orig)

res.sort(key=takeX, reverse=False)

# remove duplicates
final_res = []
for i in range(len(res)):
    dup = False
    for j in range(i+1, len(res)):
        if(abs(float(res[i][0]) - float(res[j][0])) <= 0.2 and abs(float(res[i][1]) - float(res[j][1])) <= 0.2):
            dup = True
    if(not dup):
        final_res.append(res[i])

# print(final_res)  # list of all the star locations.

#Creating an array of all the stars in the picture
allStars = []
for i in range(0,len(final_res)-1):
    s = Star(final_res[i][0], final_res[i][1], final_res[i][2], "unknown")
    allStars.append(s)

allTriples = find_all_triplets(allStars)
# for i in range(0, len(allTriples)):
    # print(i)
# print(T.getDistances())
# print(T.getAngles(), " : ANGLES")
