from example import dataset, model
from deep_eknn import EKNN
import numpy as np

# Load CIFAR10 dataset
trainloader, testloader = dataset.load_CIFAR10()

# Load pre-trained ResNet18
ResNet18 = model.load_ResNet18()

# Feedforward the network and extract new feature space
X_train, y_train, X_test, y_test = model.encode_input(trainloader, testloader, ResNet18)

# Number of Nearest Neighbors
K_NEIGHBORS = 8

# Fit Deep EK-NN(nb_classes, nb_neighbors)
estimator = EKNN(10, K_NEIGHBORS)
estimator.fit(X_train, y_train)

# Compute accuracy (if needed)
accuracy = estimator.score(X_test, y_test)

# Compute uncertainties
aleatoric, epistemic = estimator.get_uncertainties(X_test)

print("Model accuracy: %.4f" % accuracy)
print("Mean Epistemic uncertainty: %.4f\n\rMean Aleatoric uncertainty: %.4f" 
      % (np.mean(epistemic), np.mean(aleatoric)))