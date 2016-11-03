import random


class EnsembleUtil:

    def createBootstrapSamples(self, trainingSet, numBags):
        bootstrap_samples = []
        sample_size = len(trainingSet)

        for index in range(numBags):
            print("Creating bootstrap sample #" + str(index))
            bootstrap_samples.append(
                [random.choice(trainingSet) for _ in range(sample_size)]
            )

        return bootstrap_samples