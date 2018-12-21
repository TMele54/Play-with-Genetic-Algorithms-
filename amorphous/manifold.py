from __future__ import division
import pandas as pd
from collections import namedtuple
import operator
import matplotlib.patches as patches
import numpy as np
from random import randint
import random as rand
import math,time,pickle
from collections import namedtuple
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.special import binom
import matplotlib.pyplot as plt

circCreature = namedtuple('circCreature', 'X Y R l')
########################################################################################################################
##################################################### Functions ########################################################
########################################################################################################################
#makes random manifold
def get_new_manifold(scale):
    def new_space():
        # GET manifold
        bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)

        rad = 0.2
        edgy = 0.05
        points = list()

        def bezier(points, num=200):
            N = len(points)
            t = np.linspace(0, 1, num=num)
            curve = np.zeros((num, 2))
            for i in range(N):
                curve += np.outer(bernstein(N - 1, i, t), points[i])
            return curve
        class Segment():
            def __init__(self, p1, p2, angle1, angle2, **kw):
                self.p1 = p1;
                self.p2 = p2
                self.angle1 = angle1;
                self.angle2 = angle2
                self.numpoints = kw.get("numpoints", 100)
                r = kw.get("r", 0.3)
                d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
                self.r = r * d
                self.p = np.zeros((4, 2))
                self.p[0, :] = self.p1[:]
                self.p[3, :] = self.p2[:]
                self.calc_intermediate_points(self.r)
            def calc_intermediate_points(self, r):
                self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                                   self.r * np.sin(self.angle1)])
                self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                                   self.r * np.sin(self.angle2 + np.pi)])
                self.curve = bezier(self.p, self.numpoints)
        def get_curve(points, **kw):
            segments = []
            for i in range(len(points) - 1):
                seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
                segments.append(seg)
            curve = np.concatenate([s.curve for s in segments])
            return segments, curve
        def ccw_sort(p):
            d = p - np.mean(p, axis=0)
            s = np.arctan2(d[:, 0], d[:, 1])
            return p[np.argsort(s), :]
        def get_bezier_curve(a, rad=0.2, edgy=0):
            """ given an array of points *a*, create a curve through
            those points.
            *rad* is a number between 0 and 1 to steer the distance of
                  control points.
            *edgy* is a parameter which controls how "edgy" the curve is,
                   edgy=0 is smoothest."""
            p = np.arctan(edgy) / np.pi + .5
            a = ccw_sort(a)
            a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
            d = np.diff(a, axis=0)
            ang = np.arctan2(d[:, 1], d[:, 0])
            f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
            ang = f(ang)
            ang1 = ang
            ang2 = np.roll(ang, 1)
            ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
            ang = np.append(ang, [ang[0]])
            a = np.append(a, np.atleast_2d(ang).T, axis=1)
            s, c = get_curve(a, r=rad, method="var")
            x, y = c.T
            return x, y, a
        def get_random_points(n=5, scale=scale, mindst=None, rec=0):
            """ create n random points in the unit square, which are *mindst*
            apart, then scale them."""
            mindst = mindst or .7 / n
            a = np.random.rand(n, 2)
            d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
            if np.all(d >= mindst) or rec >= 200:
                return a * scale
            else:
                return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)
        a = get_random_points(n=10, scale=scale)
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
        for i in range(0, len(x)):
            points.append((x[i], y[i]))

        points2 = x, y

        # plt.plot(x,y)
        # plt.show()
        space = Polygon(points)
        return space, points, points2  # polygon, (x,y), [[x],[y]]
    def save_space(space):
        with open('space.pickle', 'wb') as f:
            pickle.dump(space, f)
        return space
    def on_manifold(point, space):
        point = Point(point[0], point[1])  # point id in form   0.5,0.5   no tup or list
        return space.contains(point)

    manifold, points, XYpoints = save_space(new_space())
    return manifold, points, XYpoints

#loads previous manifold
def load_manifold():
    with open('space.pickle', 'rb') as f:
        space, points, points2 = pickle.load(f)
    return space, points, points2

# is point on manifold
def on_manifold(point, space):
    point = Point(point[0], point[1])  # point id in form   0.5,0.5   no tup or list
    return space.contains(point)

#creates n cirles
def make_circles(popSize,scale):
    population = []

    for i in range(0, popSize):
        R = 5
        X = rand.randint(0,scale)
        Y = rand.randint(0,scale)
        l = 1

        totalY = Y + R
        totalX = X + R

        lessX = X - R
        lessY = Y - R

        while totalX > dX:
            X = X - (l + 1)
            totalX = X + R
        while totalY > dY:
            Y = Y - (l + 1)
            totalY = Y + R
        while lessX < R:
            X = X + (l + 1)
            lessX = X - R
        while lessY < R:
            Y = Y + (l + 1)
            lessY = Y - R

        population.append(circCreature(X, Y, R, l))

    return population

# creates first set of circles
def initialPopulation(popSize, scale, manifold, XYpoints):

    population = make_circles(popSize,scale)

    # display initial population
    plot_circles_manifold(points=XYpoints, polygon=manifold, creatures=population)

    return population

# distance from circle center to target circle center
def distance(p0, p1):  # (X1,Y1), (X2,Y2)
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

# area in units of intersection between a circle and the target circl
def intersectionArea(d, a, b):
    R = a.R
    r = b["R"]

    if d <= abs(R - r): # Concentric
        return np.pi * min(R, r) ** 2

    if d >= r + R: # too far apart
        return 0

    r2, R2, d2 = r ** 2, R ** 2, d ** 2
    alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
    beta = np.arccos((d2 + R2 - r2) / (2 * d * R))

    return (r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta)))

# performs distance function
def centerDistanceMethod(a, b):
    aCenter= [a.X, a.Y]
    bCenter = [b["X"], b["Y"]]
    centerDistance = distance(aCenter, bCenter)

    return [centerDistance]

# rank order the distance
def rankProperties(generation, manifold):
    ranks = list()
    # either you are in or out of the landing field, later to add, not overlapping another circle
    for i in range(0, len(generation)-1):
        om = on_manifold(generation[i], manifold)
        #adj=''
        ranks.append((i,om))
    ranks.sort(key=lambda tup: tup[1])
    return ranks

# select those to continue in the next generation
def artificialSelection(genRanked, eliteSize):
    print "*" * 100
    selectionResults = []
    df = pd.DataFrame(np.array(genRanked), columns=["Index", "Bool"])
    df['cum_sum'] = df.Bool.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Bool.sum()

    # get all elite (need adjustment here...)
    for i in range(0, eliteSize):
        selectionResults.append(genRanked[i][0])

    # randomly get creatures
    for i in range(0, len(genRanked) - eliteSize):
        pick = 100 * rand.random()
        for i in range(0, len(genRanked)):

            if pick <= df.iat[i, 3]:
                selectionResults.append(genRanked[i][0])
                break

    return selectionResults

# group of those whose genes will be passed on
def matingPool(generation, selectResults):
    pool = []

    # get selected creatures by their index..
    for i in range(0, len(selectResults)):
        index = selectResults[i]
        pool.append(generation[index])

    return pool

# keep circles form over lapping
def no_overlapping(creature_of_interest, pool):

    for creature in pool:
        if distance(creature_of_interest, creature) <= 10:
            creature_of_interest = make_circles(1,scale=scale)[0]
        else:
            pass

    return creature_of_interest




# special editing space for child circles
def CRISPR(genes, pool):
    dX, dY = genes[0], genes[1]
    X, Y, R, l = genes[2][0], genes[2][1], 5, 3

    totalY = Y + R
    totalX = X + R
    lessX = X - R
    lessY = Y - R
    while totalX > dX:
        X = X - (l + 1)
        totalX = X + R
    while totalY > dY:
        Y = Y - (l + 1)
        totalY = Y + R
    while lessX < R:
        X = X + (l + 1)
        lessX = X - R
    while lessY < R:
        Y = Y + (l + 1)
        lessY = Y - R

    eG = X, Y, R, l
    print len(pool)

    child = circCreature(eG[0], eG[1], eG[2], eG[3])
    # stop over lapping
    #child = no_overlapping(child, pool)


    return child

# breed, exchange of genes
def breed(parent1, parent2, dX, dY, pool):
    # just being explicit
    aveGenes = [(parent1[0]+parent2[0])/2,(parent1[1]+parent2[1])/2]

    genes = aveGenes ## this needs to be something meaningful...

    # keep circles from going off the page..
    editedGenes = CRISPR((dX, dY, genes), pool) ################################################dRdRdR

    eG = editedGenes
    child = circCreature(eG[0], eG[1], eG[2], eG[3])

    return child

# calls breed function
def breedPopulation(matingpool, eliteSize, dX, dY, dR):
    children = []

    # the elite are passed onto the next generation
    length = len(matingpool) - eliteSize

    # randomize positions..
    pool = rand.sample(matingpool, len(matingpool))

    # keep the elite
    for i in range(0, eliteSize):
        children.append(matingpool[i])

    # do breeding
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1], dX, dY, pool)
        children.append(child)

    return children

# gives a random mutation (new circle dims, simple change)
def mutate(individual, mutationRate, scale):
    r = rand.random()
    if (r < mutationRate):
        individual = make_circles(1, scale=scale)[0]

    return individual

# performs mutations
def mutatePopulation(children, mutationRate, scale):
    mutatedPop = []

    for individual in range(0, len(children)):
        mutatedIndividual = mutate(children[individual], mutationRate, scale)
        mutatedPop.append(mutatedIndividual)

    return mutatedPop

# plot circles on manifold
def plot_circles_manifold(points, polygon, creatures):
    fig = plt.figure(figsize=(16, 16), dpi=200)
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    m = 0,

    def draw_circle(x, y, radius, color):
        from matplotlib.patches import Circle
        from matplotlib.patheffects import withStroke
        circle = Circle((x, y),radius,lw=5,color=color, zorder=2, facecolor=color)
        ax.add_patch(circle)

    m=0
    on=0
    off=0
    for cre in creatures:

        m += 1
        # this works to plot the Path
        ax.plot(points[1],points[0])

        if on_manifold((cre[1], cre[0]), polygon) == True:
            draw_circle(cre[0], cre[1], 1, 'blue')
            on+=1
        else:
            draw_circle(cre[0], cre[1], 1, 'red')
            off+=1

    ax.set_title("Points on Manifold: "+str((on/1000)*100)+"% of, "+str(len(creatures))+" Creatures", fontsize=20, verticalalignment='bottom')
    plt.show()
    return None

# progress as %
def progress(polygon, creatures):
    on = 0
    off = 0
    for cre in creatures:
        if on_manifold((cre[1], cre[0]), polygon) == True:
            on += 1
        else:
            off += 1

    print("Points on Manifold: " + str((on / 1000) * 100) + "% of, " + str(len(creatures)) + " Creatures")

# creates new generation
def newGeneration(pop, mutationRate, eliteSize, dX, dY, dR, manifold, gen):
    genRanked = rankProperties(generation=pop, manifold=manifold)

    print "Current Population Size:", len(genRanked)

    # execute artificial process of natural selection
    chosenOnes = artificialSelection(genRanked=genRanked, eliteSize=eliteSize)

    # Gather parents for breeding
    pool = matingPool(generation=pop, selectResults=chosenOnes)

    # breed the mating pool to obtain children.. children will take on X,Y and H,W genes from parents
    children = breedPopulation(matingpool=pool, eliteSize=eliteSize, dX=dX, dY=dY, dR=dR)

    # mutations
    nextGeneration = mutatePopulation(children=children, mutationRate=mutationRate, scale=scale)
    return nextGeneration

# runs Genetic algorithm
def geneticAlgorithm(popSize, eliteSize, mutationRate, generations, dX, dY, dR, scale, manifold, XYpoints):
    # creates a population of circle points
    population = initialPopulation(popSize=popSize, scale=scale, manifold=manifold, XYpoints=XYpoints)
    creatures = list()
    print "Initial Population Size:", len(population)

    for i in range(0, generations):
        print "Computing Generation: {-", i, "-} of", generations, " generations.."
        population = newGeneration(pop=population, mutationRate=mutationRate, eliteSize=eliteSize, dX=dX, dY=dY, dR=dR, manifold=manifold, gen=i)


        #plot_circles_manifold(points=XYpoints, polygon=manifold, creatures=population)
        progress(polygon=manifold, creatures=population)
        if i % 5 == 0:
            plot_circles_manifold(points=XYpoints, polygon=manifold, creatures=population)

        #bestcircIndex = rankProperties(pop,)[0][0]
        #bestcirc = pop[bestcircIndex]
        #creatures.append(bestcirc)

    plot_circles_manifold(points=XYpoints, polygon=manifold, creatures=population)

    return None

########################################################################################################################
##################################################### Init Vars ########################################################
########################################################################################################################

# coordinate space (page size)
scale=1000
MAX_RADIUS=scale/500
dX=scale
dY=scale
dR=MAX_RADIUS
coorSpace=[dY, dX]
popSize=1000
eliteSize=int(popSize/10)
generations=1000
mutationRate=0.05

manifold, points, XYpoints = load_manifold()  # get_new_manifold(scale)
########################################################################################################################
##################################################### Execute ##########################################################
########################################################################################################################
# execute GA function..

geneticAlgorithm(popSize=popSize,eliteSize=eliteSize,mutationRate=mutationRate,generations=generations,
                 dX=dX,dY=dY,dR=dR,scale=scale,manifold=manifold,XYpoints=XYpoints)

