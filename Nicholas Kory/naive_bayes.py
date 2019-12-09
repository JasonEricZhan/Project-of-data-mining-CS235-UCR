import numpy as np
import pandas as pd


# function train generates the naive Bayes model
# params: features - a 2D array with rows as datapoints and col as features
#         classes - a col array with the corresponding y value to the features
# returns: a list object with the prior probability and the conditional probability
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


# function predict uses the naive Bayes model to make predictions about test data
# params: test - a 2D array with rows as datapoints and col as features
#         model - a list object with the prior probability and the conditional
#                 probability of the naive Bayes model
# returns: a col array with the predictions for the test set
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

# uncomment to use toy data set
# features = np.loadtxt('toy_train.csv', delimiter=',')
# classes = np.loadtxt('toy_train_classes.csv', delimiter=',')
# test = np.loadtxt('toy_test.csv', delimiter=',')
# ids = np.loadtxt('toy_test_ids.csv', delimiter=',')

# uncomment to use data sample
features = np.loadtxt('sampling_train_preprocessed.csv', delimiter=',')
classes = np.loadtxt('sampling_train_preprocessed_classes.csv', delimiter=',')
test = np.loadtxt('sampling_test_2_preprocessed.csv', delimiter=',')
ids = np.loadtxt('sampling_test_2_preprocessed_ids.csv', delimiter=',')

model = train(features, classes)
prediction = predict(test,model)
submission = np.column_stack((ids,prediction))

print(submission)

pd.DataFrame(submission).to_csv('nb_submission_2.csv', header=['id','target'], index=False)
