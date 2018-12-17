import numpy as np
import random, operator
import pandas as pd
import matplotlib.pyplot as plt

# 2 Classes City & Fitness

# Create instance of a city
class City:
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

# Create instance of "Fitness"
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]

                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList,i):
    route = random.sample(cityList, len(cityList))
    for rt in route:
        print "Route ("+str(i)+") :",rt
    return route

def initialPopulation(popSize, cityList):
    population=[]
    for i in range(0,popSize):
        population.append(createRoute(cityList,i))
    return population

def rankRoutes(population):
    fitnessResults = dict()
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    ranks = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse = True)

    return ranks

def naturalSelection(popRanked, eliteSize):
    selectionResults=[]
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0,eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked)-eliteSize):
        pick = 100*random.random()

        for i in range(0,len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults

def matingPool(population, selectResults):
    matingPool=[]
    for i in range(0,len(selectResults)):
        index = selectResults[i]
        matingPool.append(population[index])
    return matingPool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA,geneB)
    endGene = max(geneA,geneB)

    for i in range(startGene,endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1+childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i],pool[len(matingpool)-i-1])
        children.append(child)

    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0,len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):

    # take current population and rank them
    popRanked = rankRoutes(currentGen)

    # select good parents
    selectionResults = naturalSelection(popRanked, eliteSize)

    # define those to breed
    matingpool = matingPool(currentGen, selectionResults)

    # breed children
    children = breedPopulation(matingpool, eliteSize)

    # add a mutation
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial Distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0,generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final Distance: " + str(1 / rankRoutes(pop)[0][1] ))

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):

    pop = initialPopulation(popSize, population)

    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

cityList = []
citiesCount = 26

for i in range(0,citiesCount):
    city = City(
                    x=int(random.random()*200),
                    y=int(random.random()*200)
                )
    print City
    cityList.append(city)

    print("Created City: ", i)

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


