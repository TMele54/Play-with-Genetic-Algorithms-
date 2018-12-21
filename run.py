''' Anthony Mele 12/15/18, TonyMele.Rutgers@gmail.com

    Below are a series of implementations for the Genetic Algorithm.

    Written from scratch, without any Python Genetic Algorithm Frameworks

    Loosely based on Research here: (this is basically a typical design for GA)

    https://www.researchgate.net/publication/309770246_A_Study_on_Genetic_Algorithm_and_its_Applications

'''

# -----------------------Rectangles-----------------------
# Concept:: fit random rectangles to a key rectangle using the GA
# Fitness:: minimize distance OR maximize area of intersection

#from geometric.rectangles import geneticAlgorithm as Rectangles
#Rectangles()

# ------------------------Salesman------------------------
# Concept:: solve traveling salesman problem
# Fitness:: this also optimizes distance but in a different way
#from logistics.tsp import geneticAlgorithm as Salesman
#Salesman()

# -----------------------Targeting----------------------
# Concept:: develope circles with the ability to live within a closed path
# Fitness: optimizes distance and kin overlap
from amorphous.manifold import geneticAlgorithm as tar
#tar()


