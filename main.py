import matplotlib.pyplot as plt
from TSLearner import *
from GreedyLearner import *
from UCB1Learner import *

numOfArms = 4

# Bernoulli distribution probabilities
probabilities = np.array([0.15, 0.1, 0.1, 0.35])
optimalArmProb = probabilities[3]

opt = 0

for i in range(0, 300):
    opt += np.random.binomial(1, optimalArmProb)

timeHorizon = 300

# since the reward functions are stochastic, to better visualize the results and remove the noise, we have to perform
# at least 1000 or 10000 experiments
numOfExperiments = 10000

tsRewardsPerExperiment = []
greedyRewardsPerExperiment = []
ucb1RewardsPerExperiment = []

cumulativeTsReward = 0
cumulativeGreedyReward = 0
cumulativeUCB1Reward = 0

env = Environment(numOfArms = numOfArms, probabilities = probabilities)
tsLearner = TSLearner(numOfArms = numOfArms)
greedyLearner = GreedyLearner(numOfArms = numOfArms)
ucb1Learner = UCB1Learner(numOfArms = numOfArms)

for experiment in range(0, numOfExperiments):
    tsLearner.reset()
    greedyLearner.reset()
    ucb1Learner.reset()

    for t in range(0, timeHorizon):
        # Thompson Sampling Learner
        tsLearner.executeIteration(env = env)

        # Greedy Learner
        greedyLearner.executeIteration(env = env)

        # UCB1 Learner
        ucb1Learner.executeIteration(env = env)

    tsRewardsPerExperiment.append(tsLearner.getRewards())
    greedyRewardsPerExperiment.append(greedyLearner.getRewards())
    ucb1RewardsPerExperiment.append(ucb1Learner.getRewards())

    cumulativeTsReward += tsLearner.getCumulativeReward()
    cumulativeGreedyReward += greedyLearner.getCumulativeReward()
    cumulativeUCB1Reward += ucb1Learner.getCumulativeReward()

cumulativeTsReward = cumulativeTsReward / numOfExperiments
cumulativeGreedyReward = cumulativeGreedyReward / numOfExperiments
cumulativeUCB1Reward = cumulativeUCB1Reward / numOfExperiments

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")

# the regret is the cumulative sum of the difference from the optimum and the reward collected by the agent
plt.plot(np.cumsum(np.mean(optimalArmProb - tsRewardsPerExperiment, axis = 0)), 'r')
plt.plot(np.cumsum(np.mean(optimalArmProb - greedyRewardsPerExperiment, axis = 0)), 'g')
plt.plot(np.cumsum(np.mean(optimalArmProb - ucb1RewardsPerExperiment, axis = 0)), 'b')

plt.legend(["TS", "Greedy", "UCB1"])
plt.show()

print("Clairvoyant algorithm cumulative expected reward: " + str(opt))
print("Expected cumulative reward of Thompson Sampling: " + str(cumulativeTsReward))
print("Expected cumulative reward of Greedy algorithm: " + str(cumulativeGreedyReward))
print("Expected cumulative reward of UCB1 algorithm: " + str(cumulativeUCB1Reward))
