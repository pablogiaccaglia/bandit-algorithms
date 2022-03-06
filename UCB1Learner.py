from Learner import *
import math


class UCB1Learner(Learner):

    def __init__(self, numOfArms) -> None:
        super().__init__(numOfArms = numOfArms)
        self.numOfActions = np.zeros(numOfArms)
        self.empiricalMeans = np.zeros(numOfArms)
        self.ucbs = np.zeros(numOfArms)

    def update(self, pulledArm, reward) -> None:
        super().update(pulledArm = pulledArm, reward = reward)
        self.updateMean(arm = pulledArm, reward = reward)
        self.numOfActions[pulledArm] += 1

    def updateMean(self, arm, reward) -> None:
        self.empiricalMeans[arm] = (self.empiricalMeans[arm] * (self.t - 1) + reward) / self.t

    def computeUCB(self, arm) -> float:
        return self.empiricalMeans[arm] + math.sqrt((2 * math.log10(self.t)) / self.numOfActions[arm])

    # the function selects the arm to pull at each time t according to the one with the maximum upper confidence bound
    def pullArm(self) -> int:
        # We want to pull each arm at least once, so at round 0 we pull arm 0 etc..
        if self.t < self.numOfArms:
            return self.t

        # if all the arms have been pulled at least once, we select the arm which maximizes the expectedRewards array
        # we are taking the indexes of the arm with the maximum expected reward, but since we could have multiple arms returned
        # we have to pick one randomly to pull
        idxs = np.argmax([self.computeUCB(i) for i in range(0, self.numOfArms)]).reshape(-1)
       # print(idxs)
        return np.random.choice(idxs)

    def executeIteration(self, env: Environment) -> None:
        pulledArm = self.pullArm()
        reward = env.round(pulledArm = pulledArm)
        self.update(pulledArm = pulledArm, reward = reward)

    def reset(self, numOfArms: Optional[int] = None) -> None:
        if not numOfArms:
            numOfArms = self.numOfArms

        self.numOfActions = np.zeros(numOfArms)
        self.empiricalMeans = np.zeros(numOfArms)
        self.ucbs = np.zeros(numOfArms)

        super(UCB1Learner, self).reset(numOfArms = numOfArms)
