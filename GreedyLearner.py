from Learner import *


class GreedyLearner(Learner):

    def __init__(self, numOfArms):
        super().__init__(numOfArms = numOfArms)
        self.expectedRewards = np.zeros(numOfArms)

    # the function selects the arm to pull at each time t by maximizing the
    # expectedRewards array.
    def pullArm(self) -> int:

        # We want to pull each arm at least once, so at round 0 we pull arm 0 etc..
        if self.t < self.numOfArms:
            return self.t

        # if all the arms have been pulled at least once, we select the arm which maximizes the expectedRewards array
        # we are taking the indexes of the arm with the maximum expected reward, but since we could have multiple arms returned
        # we have to pick one randomly to pull
        idxs = np.argwhere(self.expectedRewards == self.expectedRewards.max()).reshape(-1)
        return np.random.choice(idxs)

    def update(self, pulledArm, reward) -> None:
        super().update(pulledArm = pulledArm, reward = reward)

        # the expected reward is computed by updating the mean
        self.expectedRewards[pulledArm] = (self.expectedRewards[pulledArm]*(self.t-1) + reward)/self.t

    def reset(self, numOfArms: Optional[int] = None) -> None:

        if not numOfArms:
            numOfArms = self.numOfArms

        self.expectedRewards = np.zeros(numOfArms)
        super(GreedyLearner, self).reset(numOfArms = numOfArms)

    def executeIteration(self, env: Environment) -> None:
        pulledArm = self.pullArm()
        reward = env.round(pulledArm = pulledArm)
        self.update(pulledArm = pulledArm, reward = reward)




