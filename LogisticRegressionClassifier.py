from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import FileOperations as fo

training_data, training_data_class_labels = fo.load_twitter_data_from_file(fo.TRAINING_DATA_FILE_NAME)
testing_data, testing_data_class_labels = fo.load_twitter_data_from_file(fo.TESTING_DATA_FILE_NAME)

classifier = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
classifier.fit(training_data, training_data_class_labels)
predicted_labels = classifier.predict(testing_data)
print("Confusion Matrix")
print(confusion_matrix(testing_data_class_labels, predicted_labels))
print("Precision")
print(precision_score(testing_data_class_labels, predicted_labels, average=None))
print("Recall")
print(recall_score(testing_data_class_labels, predicted_labels, average=None))
print("F1 score")
print(f1_score(testing_data_class_labels, predicted_labels, average=None))
