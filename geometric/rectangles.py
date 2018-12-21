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

        Added CRISPR function to keep parents from having offspring whose rectangle 
        goes off the coordinate space.     
        
        Adjusting population to live in cord space needed a -4 dx/dy
        
        Total intersections are given a one point boost to position them at the top of rank list
        
        12/16/18 - need to improve fitness metric.. (this was updated early 17th to use distance and not area..)
    
'''

########################################################################################################################
##################################################### Creatures ########################################################
########################################################################################################################
# rectangle interface/builder
rectCreature = namedtuple('rectCreature', 'X Y H W l')

########################################################################################################################
##################################################### Functions ########################################################
########################################################################################################################
# creates first set of boxes
def initialPopulation(popSize):
    population=[]

    for i in range(0,popSize):
        H = randint(1, dY)
        W = randint(1, dX)
        X = randint(1, dX)
        Y = randint(1, dY)
        l = 3
        totalY = Y + H
        totalX = X + W

        while totalX > dX:
            X = X - 4 ## subtracting 1 left the outside of the rect to be displayed outside of the cord space
            totalX = X + W
        while totalY > dY:
            Y = Y - 4
            totalY = Y + H

        population.append(rectCreature(X,Y,H,W,l))

    # display initial population
    # plotRectangles(coorSpace=coorSpace, tBB=tBB, creatures=population)


    return population

# distance from box corners to target box
def distance(p0, p1): #(X1,Y1), (X2,Y2)
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

# area in units of intersection between a box and the target box
def intersectionArea(a, b):  # returns False if rectangles don't intersect

    aTopLeft = [a.X, a.Y]
    aBottomRight = [a.X + a.W, a.Y + a.H]
    bTopLeft = [b["X"], b["Y"]]
    bBottomRight = [b["X"] + b["W"], b["Y"] + b["H"]]

    dist = distance(aTopLeft,bTopLeft)
    xO = min(aBottomRight[0],bBottomRight[0])-max(aTopLeft[0],bTopLeft[0])
    yO = min(aBottomRight[1],bBottomRight[1])-max(aTopLeft[1],bTopLeft[1])

    # calculated areas
    targetArea = b["W"] * b["H"]
    interArea = xO * yO

    if interArea < 0:
        interArea = 0

    #print targetArea, interArea, xO, yO
    #plotBestRectangles(coorSpace=coorSpace, tBB=tBB, creatures=a)

    # does intersect?
    if (xO >= 0) and (yO >= 0):
        if (interArea == targetArea):
            return [True, interArea+1, dist]
        else:
            return [True, interArea, dist]
    else:
        return [False, 0, dist]

# ^^ as a %
def intersectionPercentage(a, b):  # returns None if rectangles don't intersect
    aTopLeft = [a.X, a.Y]
    aBottomRight = [a.X + a.W, a.Y + a.H]
    bTopLeft = [b["X"], b["Y"]]
    bBottomRight = [b["X"] + b["W"], b["Y"] + b["H"]]

    dist = distance(aTopLeft, bTopLeft)
    xO = min(aBottomRight[0], bBottomRight[0]) - max(aTopLeft[0], bTopLeft[0])
    yO = min(aBottomRight[1], bBottomRight[1]) - max(aTopLeft[1], bTopLeft[1])

    # calculated areas
    targetArea = b["W"] * b["H"]
    interArea = xO * yO

    if interArea < 0:
        interArea = 0


    if (xO >= 0) and (yO >= 0):
        print "Percent Coverage: {0:.0f}%".format(interArea / targetArea * 100)
        print "Absolute Distance: ", round(dist,2)
        return interArea / targetArea * 100
    else:
        return 0.0

# performs distance function
def cornerDistanceMethod(a, b):
    aTopLeft = [a.X, a.Y]
    aBottomRight = [a.X + a.W, a.Y + a.H]
    bTopLeft = [b["X"], b["Y"]]
    bBottomRight = [b["X"] + b["W"], b["Y"] + b["H"]]

    upperLeftDistance = distance(aTopLeft,bTopLeft)
    lowerRightDistance = distance(aBottomRight,bBottomRight)


    return [upperLeftDistance,lowerRightDistance]

# rank order the distance
def rankDistances(generation, tBB):
    ranks = list()

    for i in range(0, len(generation)):
        a,b = cornerDistanceMethod(generation[i], tBB)
        c = a+b
        ranks.append((i,a,b,c))


    ranks.sort(key=lambda tup: tup[3])

    return ranks

# rank order the areas
def rankAreas(generation, tBB):
    fitnessResults={}
    ranks={}

    for i in range(0,len(generation)):
        # returns [bool T/F, area-int, distance-float]
        fitnessResults[i] = intersectionArea(generation[i], tBB)

    # sort descending
    ranks = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

    # tuple structure
    for i in range(0,len(ranks)):
        ranks[i] = (ranks[i][0], ranks[i][1][1]*1.0, ranks[i][1][2]*1.0)

    ranks.sort(key=lambda tup: tup[2])

    return ranks

# select those to continue in the next generation
def artificialSelection(genRanked, eliteSize):

    print "*"*100
    selectionResults=[]
    df = pd.DataFrame(np.array(genRanked), columns=["Index","Area","Distance","Total"])
    df['cum_sum'] = df.Area.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Area.sum()
    df.head()

    # get all elite (need adjustment here...)
    for i in range(0,eliteSize):
        selectionResults.append(genRanked[i][0])

    # randomly get creatures
    for i in range(0, len(genRanked)-eliteSize):
        pick = 100*rand.random()
        for i in range(0,len(genRanked)):

            if pick <= df.iat[i,3]:
                selectionResults.append(genRanked[i][0])
                break

    return selectionResults

# group of those whose genes will be passed on
def matingPool(generation, selectResults):
    pool=[]

    # get selected creatures by their index..
    for i in range(0,len(selectResults)):
        index = selectResults[i]
        pool.append(generation[index])

    return pool

# special editing space for child boxes (ie if a child's box would extend beyond the cord space)
def CRISPR(genes):
    dX, dY = genes[0],genes[1]
    X, Y, H, W, l = genes[2][0],genes[2][1],genes[2][2],genes[2][3],genes[2][4]

    totalY = Y + H
    totalX = X + W

    while totalX > dX:
        X = X - 4
        totalX = X + W

    while totalY > dY:
        Y = Y - 4
        totalY = Y + H

    editedGenes = X, Y, H, W, l

    return editedGenes

# breed, exchange of genes
def breed(parent1, parent2, dX, dY):

    #just being explicit
    geneA = parent1[0], parent1[1]
    geneB = parent2[4], parent2[3]

    genes = geneA[0], geneA[1], geneB[0], geneB[1], 5

    # keep boxes from going off the page..
    editedGenes = CRISPR((dX, dY, genes))

    eG = editedGenes
    child = rectCreature(eG[0],eG[1],eG[2],eG[3],eG[4])

    return child

# calls breed function
def breedPopulation(matingpool, eliteSize, dX, dY):
    children = []

    # the elite are passed onto the next generation
    length = len(matingpool) - eliteSize

    # randomize positions..
    pool = rand.sample(matingpool, len(matingpool))

    # keep the elite
    for i in range(0,eliteSize):
        children.append(matingpool[i])

    # do breeding
    for i in range(0, length):
        child = breed(pool[i],pool[len(matingpool)-i-1],dX,dY)
        children.append(child)

    return children

# gives a random mutation (new box dims, simple change)
def mutate(individual, mutationRate):

    if(rand.random() < mutationRate):
        # rectCreature(X=11, Y=48, H=1152, W=889, l=3)

        X = randint(1, dX)
        Y = randint(1, dY)
        H = randint(1, dY)
        W = randint(1, dX)
        l = individual[4]
        totalY = Y + H
        totalX = X + W

        while totalX > dX:
            X = X - 4
            totalX = X + W

        while totalY > dY:
            Y = Y - 4
            totalY = Y + H

        individual = rectCreature(X,Y,H,W,l)
    return individual

# performs mutations
def mutatePopulation(children, mutationRate):
    mutatedPop = []

    for individual in range(0,len(children)):
        mutatedIndividual = mutate(children[individual], mutationRate)
        mutatedPop.append(mutatedIndividual)

    return mutatedPop

# plots rectangles
def plotRectangles(coorSpace, tBB, creatures):

    corSpace = np.array(np.zeros(coorSpace), dtype=np.uint16)
    fig, ax = plt.subplots(1,figsize=(16, 16))

    m=0
    for cre in creatures:
        m+=1

        ax.imshow(corSpace)

        rect = patches.Rectangle((tBB["X"], tBB["Y"]), tBB["W"], tBB["H"], linewidth=tBB["l"], edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((cre[0], cre[1]), cre[3], cre[2], linewidth=cre[4], edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.pause(0.3)
        ax.cla()

    plt.show()

# plots best boxes
def plotBestRectangles(coorSpace, tBB, creatures):

    cre = creatures
    corSpace = np.array(np.zeros(coorSpace), dtype=np.uint16)
    fig, ax = plt.subplots(1)

    rect = patches.Rectangle((tBB["X"], tBB["Y"]), tBB["W"], tBB["H"], linewidth=tBB["l"], edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    ax.imshow(corSpace)
    rect = patches.Rectangle((cre[0], cre[1]), cre[3], cre[2], linewidth=cre[4], edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.pause(0.2)

    plt.show()

# creates new generation
def newGeneration(pop, mutationRate, eliteSize, dX, dY):
    # rank intersected areas
    # genRanked = rankAreas(generation=pop, tBB=tBB)
    genRanked = rankDistances(generation=pop, tBB=tBB)
    print "Current Population Size:", len(genRanked)
    # execute artificial process of natural selection
    chosenOnes = artificialSelection(genRanked=genRanked, eliteSize=eliteSize)

    #print "Chosen Ones", len(chosenOnes)
    # Gather parents for breeding
    pool = matingPool(generation=pop, selectResults=chosenOnes)

    #print "Mating pool", len(pool)
    # breed the mating pool to obtain children.. children will take on X,Y and H,W genes from parents
    children = breedPopulation(matingpool=pool, eliteSize=eliteSize, dX=dX, dY=dY)

    #print "Children", len(children)
    # mutations
    nextGeneration = mutatePopulation(children=children, mutationRate=mutationRate)

    return nextGeneration

# runs Genetic algorithm
def geneticAlgorithm(popSize, eliteSize, mutationRate, generations, dX, dY):

    # get a first population or rectangles and their intersection areas
    pop = initialPopulation(popSize=popSize)

    creatures = list()
    print "Initial Population", len(pop)

    for i in range(0, generations):
        print "Computing Generation: {-", i, "-} of", generations, " generations.."
        pop = newGeneration(pop=pop, mutationRate=mutationRate, eliteSize=eliteSize, dX=dX, dY=dY)

        bestRectIndex = rankDistances(pop, tBB)[0][0]
        bestRect = pop[bestRectIndex]
        #print bestRectIndex
        creatures.append(bestRect)
        val = intersectionPercentage(bestRect, tBB)
        #with open("../xtr/example.txt", "a") as f:
        #    f.write(str(val)+ "," + str(i) + '\n')
        #    f.close()

    print "Total Top Creatures:", len(creatures)
    creatures = set(creatures)
    print "Unique Creatures:", len(set(creatures))


    #plotBestRectangles(coorSpace=coorSpace, tBB=tBB, creatures=bestRect)
    plotRectangles(coorSpace=coorSpace, tBB=tBB, creatures=creatures)

    return None

########################################################################################################################
##################################################### Init Vars ########################################################
########################################################################################################################

# coordinate space (page size)
dX, dY = 900, 1200
coorSpace = [dY,dX]

# target bounding box (C info)
tBB=dict()
tBB["X"],tBB["Y"],tBB["H"],tBB["W"],tBB["l"] = 50,650,300,800,10

# size of a population
popSize,eliteSize,generations,mutationRate = 1000,10,500,0.01

########################################################################################################################
##################################################### Execute ##########################################################
########################################################################################################################
# execute function..
# run chart.py after running this, and also first empty example.txt
geneticAlgorithm(
    popSize=popSize,
    eliteSize=eliteSize,
    mutationRate=mutationRate,
    generations=generations,
    dX=dX,
    dY=dY
    )

print '''
All this is doing is trying to create a similar rectangle to the blue rectangle. 
The interesting part is that this is being accomplished by generating random rectangles 
and breeding generations of offspring that get better and better at drawing the ideal rectangle.
At completeion the top rectangles from several generations are plotted against the key rectangle.'''