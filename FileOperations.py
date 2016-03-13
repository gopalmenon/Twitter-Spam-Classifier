from sklearn import svm
import FileOperations as fo
import numpy as np

TESTING_DATA_FOLDER = "../Testing_data/"
TRAINING_DATA_FOLDER = "../Training_data/"
HAM_USER_PROFILE_INFO_FILE = "legitimate_users.txt"
SPAM_USER_PROFILE_INFO_FILE = "spammers.txt"

# Load data from a text file containing Twitter profile information into an array and return the array contents 
# along with class label corresponding to spam or ham
def load_twitter_profile_text_file(file_path, file_name, is_spam):

 try:

  # Array containing NumberOfFollowings, NumberOfFollowers, NumberOfTweets, LengthOfScreenName, LengthOfDescriptionInUserProfile
  profile_data = np.genfromtxt(file_path.strip() + file_name.strip(), delimiter="\t")[:, [3,4,5,6,7]].astype(int)

  # Class label as a column vector or 1's or 0's corresponding to spam or ham
  class_label = np.ones(shape=(profile_data.shape[0], 1)) if is_spam else np.zeros(shape=(profile_data.shape[0], 1))

  return profile_data, class_label

 except IOError as e:
  print(e)

# Return twitter profile training or testing data along with class labels
def get_twitter_profile_details(training_data):
 # Get ham profile info and class label
 ham_profile_data, ham_class_label = load_twitter_profile_text_file(TRAINING_DATA_FOLDER if training_data else TESTING_DATA_FOLDER, HAM_USER_PROFILE_INFO_FILE, is_spam = False)

 # Get spam profile info and class label
 spam_profile_data, spam_class_label = load_twitter_profile_text_file(TRAINING_DATA_FOLDER if training_data else TESTING_DATA_FOLDER, SPAM_USER_PROFILE_INFO_FILE, is_spam = True)

 # Merge the spam and ham profile info and class labels to get data
 training_data = np.concatenate((ham_profile_data, spam_profile_data))
 training_data_class_labels = np.concatenate((ham_class_label, spam_class_label))

 return training_data, training_data_class_labels
