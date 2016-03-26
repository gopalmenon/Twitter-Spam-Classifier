import FileOperations as fo
import numpy as np

TRAINING_DATA_FILE_NAME = "Training Data Tab.txt"
TESTING_DATA_FILE_NAME = "Testing Data Tab.txt"

# Load twitter data with class labels
def load_twitter_data_from_file(file_name):

 try:
  # Get tweet data for Number Of Followings, Number Of Followers, Number Of Tweets, NumberOfUrlTweets, Change in Following
  array_of_tweets = np.genfromtxt(file_name.strip(), delimiter="\t")[:, [1, 2, 3, 7, 8]].astype(int)
  array_of_data_labels = np.genfromtxt(file_name.strip(), delimiter="\t")[:, [9]].astype(int)[:, 0]

  return array_of_tweets.tolist(), array_of_data_labels.tolist()

 except IOError as e:
  print(e)
