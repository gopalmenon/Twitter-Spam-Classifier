import numpy as np

# Load data from a text file containing Twitter profile information into an array and return the array contents 
# along with class label corresponding to spam or ham
def load_twitter_profile_text_file(file_path, file_name, is_spam):

 try:

  # Array containing NumberOfFollowings, NumberOfFollowers, NumberOfTweets, LengthOfScreenName, LengthOfDescriptionInUserProfile
  text_data = np.genfromtxt(file_path.strip() + file_name.strip(), delimiter="\t")[:, [3,4,5,6,7]].astype(int)

  # Class label as a column vector or 1's or 0's corresponding to spam or ham
  class_label = np.ones(shape=(text_data.shape[0], 1)) if is_spam else np.zeros(shape=(text_data.shape[0], 1))

  return text_data, class_label

 except IOError as e:
  print(e)

