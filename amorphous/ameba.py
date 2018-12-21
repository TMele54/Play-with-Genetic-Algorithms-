########################################################################################################################
##################################################### Modules ##########################################################
########################################################################################################################
from __future__ import division
import pandas as pd
from collections import namedtuple
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from random import randint
import random as rand
import math

########################################################################################################################
##################################################### Notes ############################################################
########################################################################################################################
''' 

    Notes:

        Added CRISPR function to keep parents from having offspring whose circle 
        goes off the coordinate space.     

        Adjusting population to live in cord space needed a -4 dx/dy

        Total intersections are given a one point boost to position them at the top of rank list

        12/16/18 - need to improve fitness metric.. (this was updated early 17th to use distance and not area..)

'''

########################################################################################################################
##################################################### Creatures ########################################################
########################################################################################################################
# circle interface/builder
circCreature = namedtuple('circCreature', 'X Y R l')


########################################################################################################################
##################################################### Functions ########################################################
########################################################################################################################

# creates first set of circles
def initialPopulation(popSize):
    population = []

    for i in range(0, popSize):
        R = randint(1, dR)
        X = randint(1, dX)
        Y = randint(1, dY)
        l = 3

        totalY = Y + R
        totalX = X + R

        lessX = X - R
        lessY = Y - R

        while totalX > dX:
            X = X - (l+1)
            totalX = X + R
        while totalY > dY:
            Y = Y - (l+1)
            totalY = Y + R
        while lessX < R:
            X = X + (l + 1)
            lessX = X - R
        while lessY < R:
            Y = Y + (l + 1)
            lessY = Y - R


        population.append(circCreature(X, Y, R, l))

    # display initial population
    # plotcircles(coorSpace=coorSpace, tBB=tBB, creatures=population)

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
def rankProperties(generation, tBB):
    ranks = list()

    for i in range(0, len(generation)):
        d = centerDistanceMethod(generation[i], tBB)[0]
        a = intersectionArea(d, generation[i], tBB)
        ranks.append((i, a, d))

    ranks.sort(key=lambda tup: tup[1])  # intersection is less important than distance..
    ranks.sort(key=lambda tup: tup[2])

    return ranks

# select those to continue in the next generation
def artificialSelection(genRanked, eliteSize):
    print "*" * 100
    selectionResults = []
    df = pd.DataFrame(np.array(genRanked), columns=["Index", "Area", "Distance"])
    df['cum_sum'] = df.Area.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Area.sum()

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

# special editing space for child circles (ie if a child's circle would extend beyond the cord space)
def CRISPR(genes):
    dX, dY = genes[0], genes[1]
    X, Y, R, l = genes[2][0], genes[2][1], genes[2][2], genes[2][3]

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

    editedGenes = X, Y, R, l

    return editedGenes

# breed, exchange of genes
def breed(parent1, parent2, dX, dY, dR):
    # just being explicit
    geneA = [parent1[0], parent1[1]]
    geneB = [parent2[2]]
    genes = geneA[0], geneA[1], geneB[0], 5
    # keep circles from going off the page..
    editedGenes = CRISPR((dX, dY, genes)) ################################################dRdRdR

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
        child = breed(pool[i], pool[len(matingpool) - i - 1], dX, dY, dR)
        children.append(child)

    return children

# gives a random mutation (new circle dims, simple change)
def mutate(individual, mutationRate):
    if (rand.random() < mutationRate):
        # circCreature(X=11, Y=48, H=1152, W=889, l=3)

        R = randint(1, dR)
        X = randint(1, dX)
        Y = randint(1, dY)
        l = 3

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

        individual = circCreature(X, Y, R, l)
    return individual

# performs mutations
def mutatePopulation(children, mutationRate):
    mutatedPop = []

    for individual in range(0, len(children)):
        mutatedIndividual = mutate(children[individual], mutationRate)
        mutatedPop.append(mutatedIndividual)

    return mutatedPop

# plots circles
def plotcircles(coorSpace, tBB, creatures):
    corSpace = np.array(np.zeros(coorSpace), dtype=np.uint16)
    fig, ax = plt.subplots(1, figsize=(16, 16))
    print len(creatures)
    m = 0
    for cre in creatures:
        m += 1

        ax.imshow(corSpace)
        circ = patches.Circle((tBB["X"], tBB["Y"]), tBB["R"], linewidth=tBB["l"], edgecolor='b', facecolor='none')
        ax.add_patch(circ)

        node_patch = patches.Circle((cre[0], cre[1]),cre[2],lw=cre[3],color="r",zorder=2, facecolor='none')
        ax.add_patch(node_patch)

        plt.pause(0.3)
        ax.cla()

    plt.show()

# creates new generation
def newGeneration(pop, mutationRate, eliteSize, dX, dY, dR):

    genRanked = rankProperties(generation=pop, tBB=tBB)

    print "Current Population Size:", len(genRanked)
    # execute artificial process of natural selection
    chosenOnes = artificialSelection(genRanked=genRanked, eliteSize=eliteSize)

    print "Chosen Ones:", len(chosenOnes)
    # Gather parents for breeding
    pool = matingPool(generation=pop, selectResults=chosenOnes)

    # print "Mating pool", len(pool)
    # breed the mating pool to obtain children.. children will take on X,Y and H,W genes from parents
    children = breedPopulation(matingpool=pool, eliteSize=eliteSize, dX=dX, dY=dY, dR=dR)

    # mutations
    nextGeneration = mutatePopulation(children=children, mutationRate=mutationRate)

    return nextGeneration

# runs Genetic algorithm
def geneticAlgorithm(popSize, eliteSize, mutationRate, generations, dX, dY, dR):
    # get a first population or circles and their intersection areas
    pop = initialPopulation(popSize=popSize)

    creatures = list()
    print "Initial Population", len(pop)

    for i in range(0, generations):
        print "Computing Generation: {-", i, "-} of", generations, " generations.."
        pop = newGeneration(pop=pop, mutationRate=mutationRate, eliteSize=eliteSize, dX=dX, dY=dY, dR=dR)

        bestcircIndex = rankProperties(pop, tBB)[0][0]
        bestcirc = pop[bestcircIndex]
        creatures.append(bestcirc)

        ''' 
            # val = intersectionPercentage(bestcirc, tBB)
            # with open("../xtr/example.txt", "a") as f:
            #    f.write(str(val)+ "," + str(i) + '\n')
            #    f.close()
        '''

    print "Total Top Creatures:", len(creatures)
    creatures = set(creatures)
    print "Unique Creatures:", len(set(creatures))

    # plotBestCircles(coorSpace=coorSpace, tBB=tBB, creatures=bestcirc)
    plotcircles(coorSpace=coorSpace, tBB=tBB, creatures=creatures)

    return None


########################################################################################################################
##################################################### Init Vars ########################################################
########################################################################################################################

# coordinate space (page size)
MAX_RADIUS = 150
dX, dY, dR = 900, 1200, MAX_RADIUS
coorSpace = [dY, dX]

# target bounding circle (C info)
tBB = dict()                             #450  #600
tBB["X"], tBB["Y"], tBB["R"], tBB["l"] = dX/2, dY/2, MAX_RADIUS, 3


from matplotlib.path import Path

n = 8 # Number of possibly sharp edges
r = .7 # magnitude of the perturbation from the unit circle,
# should be between 0 and 1
N = n*3+1 # number of points in the Path
# There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

angles = np.linspace(0,2*np.pi,N)
codes = np.full(N,Path.CURVE4)
codes[0] = Path.MOVETO

verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
path = Path(verts, codes)

fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax.add_patch(patch)

ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
ax.axis('off') # removes the axis to leave only the shape


'''
# size of a population
popSize, eliteSize, generations, mutationRate = 1000, 10, 1000, 0.01

########################################################################################################################
##################################################### Execute ##########################################################
########################################################################################################################

# execute function..


geneticAlgorithm(
    popSize=popSize,
    eliteSize=eliteSize,
    mutationRate=mutationRate,
    generations=generations,
    dX=dX,
    dY=dY,
    dR=dR
)'''