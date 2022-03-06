import numpy as np


class Environment:
    def __init__(self, numOfArms, probabilities):
        self.numOfArms = numOfArms
        self.probabilities = probabilities  # list of probabilities of success for each arm

    def round(self, pulledArm):
        return np.random.binomial(1, self.probabilities[pulledArm])
