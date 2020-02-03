import classifip

# We start by creating an instance of the base classifier we want to use
print("Example of Imprecise Linear Discriminant Analyse for Classification - Data set IRIS \n")
model = classifip.models.knn.IPKNN()
dataArff = classifip.dataset.arff.ArffFile()
dataArff.load("iris.arff")

# Learning
model.learn(dataArff)

# Evaluation : we can set the method for minimize convex problem with quadratic
test = model.evaluate([dataArff.data[80][:-1]])

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using interval dominance criterion with 0/1 costs\n")
print(test[0])

