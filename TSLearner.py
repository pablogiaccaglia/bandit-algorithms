from typing import Union

from numpy import long
from numpy import signedinteger
from Learner import *


class TSLearner(Learner):
    def __init__(self, numOfArms: int) -> None:
        super().__init__(numOfArms = numOfArms)
        self.betaParameters = np.ones([numOfArms, 2])  # matrix of numOfArms x 2 values

    def pullArm(self) -> Union[signedinteger, long]:
        # the TS algorithm samples a value for each arm from a beta distribution and then
        # selects the arm corresponding to the beta distribution of maximum sample's value

        # self.betaParameters[:, 0] selects whole first column of the matrix
        # self.betaParameters[:, 1] selects whole second column of the matrix

        index = np.argmax(np.random.beta(self.betaParameters[:, 0], self.betaParameters[:, 1]))
        return index

    def update(self, pulledArm: int, reward: Union[int, float]) -> None:
        super().update(pulledArm = pulledArm, reward = reward)
        self.betaParameters[pulledArm, 0] += reward
        self.betaParameters[pulledArm, 1] += 1.0 - reward

    def reset(self, numOfArms: Optional[int] = None) -> None:

        if not numOfArms:
            numOfArms = self.numOfArms

        self.betaParameters = np.ones([numOfArms, 2])  # matrix of numOfArms x 2 values
        super(TSLearner, self).reset(numOfArms = numOfArms)

    def executeIteration(self, env:Environment) -> None:
        pulledArm = self.pullArm()
        reward = env.round(pulledArm = pulledArm)
        self.update(pulledArm = pulledArm, reward = reward)




