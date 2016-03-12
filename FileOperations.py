import numpy as np

# Load data from a text file into an array and return the array contents
def load_text_file(file_path, file_name):
 try:
  text_data = np.genfromtxt(file_path.strip() + file_name.strip(), delimiter="\t")
  return text_data
 except IOError as e:
  print(e)

