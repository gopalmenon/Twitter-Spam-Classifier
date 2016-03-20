import FileOperations as fo
import math
import numpy as np

TESTING_DATA_FOLDER = "../Testing_data/"
TRAINING_DATA_FOLDER = "../Training_data/"
HAM_USER_PROFILE_INFO_FILE = "legitimate_users.txt"
SPAM_USER_PROFILE_INFO_FILE = "spammers.txt"
HAM_TWEETS_FILE = "legitimate_users_tweets.txt"
SPAM_TWEETS_FILE = "spammers_tweets.txt"
DATA_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# User profile related constants
NUMBER_OF_FOLLOWINGS = "Followings"
NUMBER_OF_FOLLOWERS = "Followers"
NUMBER_OF_TWEETS = "Tweets"
LENGTH_OF_SCREEN_NAME = "NameLength"
LENGTH_OF_DESCRIPTION_IN_USER_PROFILE = "DescLength"

# Load data from a text file containing Twitter profile information into an array and return the array contents 
# along with class label corresponding to spam or ham
def load_twitter_profile_text_file(file_path, file_name, is_spam):

 try:

  # Array containing NumberOfFollowings, NumberOfFollowers, NumberOfTweets, LengthOfScreenName, LengthOfDescriptionInUserProfile
  profile_data = np.genfromtxt(file_path.strip() + file_name.strip(), delimiter="\t")[:, [3,4,5,6,7]].astype(int)

  # Class label as a column vector or 1's or 0's corresponding to spam or ham
  class_label = np.ones(profile_data.shape[0], dtype=np.int) if is_spam else np.zeros(profile_data.shape[0], dtype=np.int)

  return profile_data, class_label

 except IOError as e:
  print(e)

# Load data from a text file containing Tweets into an array and return the array contents 
# along with class label corresponding to spam or ham
def load_tweets_text_file(file_path, file_name, is_spam):

 try:
  list_of_tweets = []
  with open(file_path.strip() + file_name.strip()) as tweets_file:
   for line in tweets_file:
    tweet = [text_column.strip() for text_column in line.split('\t')][2]
    list_of_tweets.append(tweet)
  
  # Class label as a column vector or 1's or 0's corresponding to spam or ham
  class_label = [1 for i in range(len(list_of_tweets))] if is_spam else [0 for i in range(len(list_of_tweets))]
  
  return list_of_tweets, class_label

 except IOError as e:
  print(e)

# Convert profile numeric data to categories
def profile_data_to_categories(profile_data):

 number_of_records = len(profile_data)
 data_in_categories = []
 for count in range(number_of_records):
  current_profile = profile_data[count]
  data_in_categories.append([NUMBER_OF_FOLLOWINGS + DATA_CATEGORIES[int(math.log(current_profile[0] if current_profile[0] > 0  else 1, 2))], NUMBER_OF_FOLLOWERS + DATA_CATEGORIES[int(math.log(current_profile[1] if current_profile[1] > 0  else 1, 2))], NUMBER_OF_TWEETS + DATA_CATEGORIES[int(math.log(current_profile[2] if current_profile[2] > 0  else 1, 2))], LENGTH_OF_SCREEN_NAME + DATA_CATEGORIES[int(math.log(current_profile[3] if current_profile[3] > 0 else 1, 2))], LENGTH_OF_DESCRIPTION_IN_USER_PROFILE + DATA_CATEGORIES[int(math.log(current_profile[4] if current_profile[4] > 0 else 1, 2))]])

 return data_in_categories

# Return twitter profile training or testing data along with class labels
def get_twitter_profile_details(is_training_data, categorize_data):
 # Get ham profile info and class label
 ham_profile_data, ham_class_label = load_twitter_profile_text_file(TRAINING_DATA_FOLDER if is_training_data else TESTING_DATA_FOLDER, HAM_USER_PROFILE_INFO_FILE, is_spam = False)

 # Get spam profile info and class label
 spam_profile_data, spam_class_label = load_twitter_profile_text_file(TRAINING_DATA_FOLDER if is_training_data else TESTING_DATA_FOLDER, SPAM_USER_PROFILE_INFO_FILE, is_spam = True)

 # Merge the spam and ham profile info and class labels to get data
 profile_data = np.concatenate((ham_profile_data, spam_profile_data))
 profile_data_class_labels = np.concatenate((ham_class_label, spam_class_label))

 return profile_data_to_categories(profile_data.tolist()) if categorize_data else profile_data.tolist(), profile_data_class_labels.tolist()

# Return tweets training or testing data along with class labels
def get_tweet_details(is_training_data):
 # Get ham profile info and class label
 ham_tweets_data, ham_class_label = load_tweets_text_file(TRAINING_DATA_FOLDER if is_training_data else TESTING_DATA_FOLDER, HAM_TWEETS_FILE, is_spam = False)

 # Get spam profile info and class label
 spam_tweets_data, spam_class_label = load_tweets_text_file(TRAINING_DATA_FOLDER if is_training_data else TESTING_DATA_FOLDER, SPAM_TWEETS_FILE, is_spam = True)

 # Merge the spam and ham profile info and class labels to get data
 tweets_data = np.concatenate((ham_tweets_data, spam_tweets_data))
 tweets_data_class_labels = np.concatenate((ham_class_label, spam_class_label))

 return tweets_data, tweets_data_class_labels

