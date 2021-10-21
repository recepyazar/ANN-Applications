#Building and plotting the convex hull of the given data sets.

from matplotlib import pyplot as plt
from random import randint
from math import atan2

set1={'vectors1':[[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1],[-1,-1],[-1,0],[-1,1]],'class1':1}
set2={'vectors2':[[-3,3],[-3,1],[-3,0],[-3,-1],[-3,-3],[-1,3],[-1,-3],[0,3],[0,-3],[1,3],[1,-3],[3,3],[3,1],[3,0],[3,-1],[3,-3],[-2,3],[-3,2],[-3,-2],[-2,-3],[2,3],[3,2],[3,-2],[2,-3]],'class2':-1}
vecs1=set1.get('vectors1')
vecs2=set2.get('vectors2')
def scatterplot(coords1, coords2, convex_hull1=None, convex_hull2=None):
    xs1, ys1 = zip(*coords1)
    xs2, ys2 = zip(*coords2)
    plt.scatter(xs1, ys1, label='class 1')
    plt.scatter(xs2, ys2, label='class -1')
    plt.legend()
    if convex_hull1 != None:
        for i in range(1,len(convex_hull1)+1):
            if i == len(convex_hull1): i=0
            c0=convex_hull1[i-1]
            c1=convex_hull1[i]
            plt.plot((c0[0],c1[0]),(c0[1],c1[1]),'r')
    if convex_hull2 != None:
        for i in range(1, len(convex_hull2) + 1):
            if i == len(convex_hull2): i = 0
            c0 = convex_hull2[i - 1]
            c1 = convex_hull2[i]
            plt.plot((c0[0], c1[0]), (c0[1], c1[1]), 'k')
    plt.show()

def polar_angle(p0,p1=None):
    if p1 == None: p1 = lock
    yspan = p0[1] - p1[1]
    xspan = p0[0] - p1[0]
    return atan2(yspan, xspan)

def distance(p0,p1=None):
    if p1 == None: p1=lock
    yspan = p0[1] - p1[1]
    xspan = p0[0] - p1[0]
    return yspan ** 2 + xspan ** 2

def det(p1,p2,p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1])\
           -(p2[1]-p1[1])*(p3[0]-p1[0])

def qs(a):
    if len(a)<=1: return a
    smaller,equal,larger=[],[],[]
    piv_ang=polar_angle(a[randint(0,len(a)-1)])
    for pt in a:
        pt_ang=polar_angle(pt)
        if pt_ang<piv_ang: smaller.append(pt)
        elif pt_ang==piv_ang: equal.append(pt)
        else:                 larger.append(pt)
    return qs(smaller) \
           + sorted(equal, key=distance) \
           + qs(larger)

def convexhull_test(points,show_progress=False):
    global lock

    min_idx=None
    for i,(x,y) in enumerate(points):
        if min_idx == None or y<points[min_idx][1]:
            min_idx=i
        if y==points[min_idx][1] and x<points[min_idx][0]:
            min_idx=i
    lock=points[min_idx]
    sorted_pts= qs(points)
    del sorted_pts[sorted_pts.index(lock)]

    hull=[lock, sorted_pts[0]]
    for s in sorted_pts[1:]:
        while det(hull[-2],hull[-1],s)<=0:
            del hull[-1]
            if len(hull)<2:
                break
        hull.append(s)
        if show_progress: scatterplot(points, hull)
    return hull

hull1=convexhull_test(vecs1, False)
hull2= convexhull_test(vecs2, False)
scatterplot(vecs1, vecs2, hull1, hull2)






