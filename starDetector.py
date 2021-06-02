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


# s1 = Star(1, 26, 3, "HI")
# s2 = Star(10, 2, 3, "HA")
# s3 = Star(15, 22, 3, "HO")
# stars = [s1, s2, s3]
# T = Triangle(stars)
# print(T.getAngles())

def takeRadius(elem):
    return float(elem[2])


def takeX(elem):
    return float(elem[0])


file_name = '90DEG.jpg'
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

cv2.imshow("Naive", orig)
cv2.imwrite("%s_processed.jpg" % file_name, orig)

res.sort(key=takeX, reverse=False)

abs(float() - float()) <= 0.5

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
s1 = Star(final_res[0][0], final_res[1][0], final_res[2][0], "1")
s2 = Star(final_res[0][1], final_res[1][1], final_res[2][1], "1")
s3 = Star(final_res[0][2], final_res[1][2], final_res[2][2], "1")
s = [s1, s2, s3]
T = Triangle(s)
print(T.getDistances())
# print(T.getAngles(), " : ANGLES")
