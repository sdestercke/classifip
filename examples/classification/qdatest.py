import classifip

# We start by creating an instance of the base classifier we want to use
print("Example of Imprecise Linear Discriminant Analyse for Classification - Data set IRIS \n")
model = classifip.models.qda.LinearDiscriminant()
data = classifip.dataset.uci_data_set.export_data_set('iris.data')

# Learning
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].tolist()
model.learn(X, y, ell=5)

# Evaluation : we can set the method for minimize convex problem with quadratic
test, _ = model.evaluate(query=X[2], method="quadratic")

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using interval dominance criterion with 0/1 costs + quadratic method\n")
print(test)

# Evaluation : we can set the method for minimize convex problem with non-linear
test, _= model.evaluate(query=X[2], method="nonlinear")

# The output is a list of probability intervals, we can print each instance :
print("\nPrediction using interval dominance criterion with 0/1 costs + nonlinear method\n")
print(test)