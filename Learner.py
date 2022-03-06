from abc import abstractmethod
from typing import Optional

import numpy as np

from abc import ABC

from Environment import Environment


class Learner(ABC):

    def __init__(self, numOfArms) -> None:
        self.numOfArms = numOfArms
        self.t = 0

        # the size of an internal list is given by the number of times the arm is pulled
        self.rewardsPerArm = [[] for _ in range(numOfArms)]
        self.collectedRewards = np.array([])

    @abstractmethod
    def update(self, pulledArm, reward) -> None:
        self.t += 1
        self.rewardsPerArm[pulledArm].append(reward)
        self.collectedRewards = np.append(self.collectedRewards, reward)

    def getRewards(self):
        return self.collectedRewards

    def getCumulativeReward(self):
        return self.collectedRewards.sum()

    def reset(self, numOfArms: Optional[int] = None) -> None:
        if not numOfArms:
            self.numOfArms = numOfArms

        self.rewardsPerArm = [[] for _ in range(numOfArms)]
        self.collectedRewards = np.array([])
        self.t = 0

    @abstractmethod
    def executeIteration(self, env: Environment) -> None:
        pass
