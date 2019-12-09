# Naïve Bayes

Nicholas Kory's model.

## Code

Run naive_bayes.py to generate the naïve Bayes output from the preprocessed data samples.

## CSV Files
I've included all of the csv files used to make the model work, as well as anything used when preprocessing the data. An explaination of these files is below:
1. nb_submission - the original nb output using live testing data (no ground truth to check data)
2. nb_submission_2 - the new nb output using sample testing data based off the large training set
3. sampling_codes - codes applied to labels in the data set
4. sampling_labels - the feature labels
5. sampling_test - original test sample set, unprocessed
6. sampling_test_2_preprocessed - new test sample set, processed
7. sampling_test_2_preprocessed_ids - the target values for the test set
8. sampling_test_2_source_truth - the target values for the test set, processed
9. sampling_test_preprocessed - original test sample set, processed
10. sampling_test_preprocessed_ids - id's for original test set
11. sampling_train - features, training data, unprocessed
12. sampling_train_preprocessed - features, training data, processed
13. sampling_train_preprocessed_classes - target values for the training set
14. toy_* - the csv files for the toy example

## Toy example
Uncomment the following on lines 64-68 to run the toy example based off of the popular weather/tennis example:
```python
# features = np.loadtxt('toy_train.csv', delimiter=',')
# classes = np.loadtxt('toy_train_classes.csv', delimiter=',')
# test = np.loadtxt('toy_test.csv', delimiter=',')
# ids = np.loadtxt('toy_test_ids.csv', delimiter=',')
```
