from sklearn import svm
from sklearn.metrics import confusion_matrix
import FileOperations as fo

training_data, training_data_class_labels = fo.get_twitter_profile_details(is_training_data = True)
testing_data, testing_data_class_labels = fo.get_twitter_profile_details(is_training_data = False)

SVM_classifier = svm.SVC()
print(SVM_classifier)
SVM_classifier.fit(training_data, training_data_class_labels)
predicted_labels = SVM_classifier.predict(testing_data)
print(confusion_matrix(testing_data_class_labels, predicted_labels))

