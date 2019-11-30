import numpy as np
import pandas as pd

features = np.loadtxt('C:\\Users\\nmkor\\Desktop\\Project-of-data-mining-CS235-UCR\\Nicholas Kory\\sampling_train_preprocessed.csv', delimiter=',')
classes = np.loadtxt('C:\\Users\\nmkor\\Desktop\\Project-of-data-mining-CS235-UCR\\Nicholas Kory\\sampling_train_preprocessed_classes.csv', delimiter=',')
test = np.loadtxt('C:\\Users\\nmkor\\Desktop\\Project-of-data-mining-CS235-UCR\\Nicholas Kory\\sampling_test_preprocessed.csv', delimiter=',')
ids = np.loadtxt('C:\\Users\\nmkor\\Desktop\\Project-of-data-mining-CS235-UCR\\Nicholas Kory\\sampling_test_preprocessed_ids.csv', delimiter=',')

def train(features, classes):
    feature_max = features.max().max()

    prior_prob = np.array([len(classes[classes==0]) / len(classes), len(classes[classes==1]) / len(classes)])
    cond_prob = np.zeros((len(features[0]),int(feature_max + 1),2))

    # for each feature, loop through the data points and add counts to cond_prob
    for j in range(int(len(features[0]))):
        for i in range (len(features)):
            cond_prob[j][int(features[i][j])][int(classes[i])] += 1

    # Laplace smoothing
    cond_prob += 1

    # find ^p(x | y)
    for feature in range(len(features[0])):
        cond_prob[feature,:,] /= (cond_prob.sum(axis=1)[feature,:,])

    # model is complete
    return (prior_prob,cond_prob)



def predict(test,model):
    (prior_prob,cond_prob) = model

    # initialize return values
    prediction = np.zeros(len(test))

    # initialize loop to check each of the examples in test
    index = 0
    for datapoint in test:
        #set ^p(y)
        p0 = prior_prob[0]
        p1 = prior_prob[1]

        # find ^p(y | x)
        for i in range(len(datapoint)):
            p0 *= cond_prob[i,int(datapoint[i]),0]
            p1 *= cond_prob[i,int(datapoint[i]),1]

        # compare the two possibilities and report rule
        if p1 > p0:
            prediction[index] = 1

        #increment index for next data point
        index += 1

    # predictions of test are complete
    return prediction


model = train(features, classes)
prediction = predict(test,model)

print(prediction)
