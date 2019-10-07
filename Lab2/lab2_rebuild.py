import pods
import zipfile
import sys
import pandas as pd
import numpy as np

'''
@Copyright Knowledge and dataset
    F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5 (4):1-19 https://doi.org/10.1145/2827872
'''
# Get the dataset
# https://grouplens.org/ You can access the data from this link

pods.util.download_url("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
zip_console = zipfile.ZipFile("./ml-latest-small.zip",'r')
for name in zip_console.namelist():
    zip_console.extract(name,'./')

# Random number generator
# You can change the value of YourStudentID to any other three digits of number

YourStudentID = 746
nUsersInExample = 10

ratings = pd.read_csv("./ml-latest-small/ratings.csv")

# userId,movieId,rating and tags in rating.csv
indexs_unique_users = ratings['userId'].unique()  # get the unique identification ID
n_users = indexs_unique_users.shape[0]   # show the number of column of file

np.random.seed(YourStudentID)  # To make the random number can be predicted and same
index_users = np.random.permutation(n_users)
# Based on the YourStudentID to disturb the DataArrange and even though you run this program in different time you can
# get the same answer as before.
my_batch_users = index_users[0:nUsersInExample]  # build the array to store the data, by using the

# list the array of movies that that these users gave watched
list_movies_each_user = [[] for _ in range(nUsersInExample)]
list_ratings_each_user = [[] for _ in range(nUsersInExample)]

# list the movie just for one
list_movies = ratings['movieId'][ratings['userId'] == my_batch_users[0]].values
list_movies_each_user[0] = list_movies

# list the rating
list_ratings = ratings['rating'][ratings['userId'] == my_batch_users[0]].values
list_ratings_each_user[0] = list_ratings

# users list
n_each_user = list_movies.shape[0]  # pass the number of movie to n_each-user
list_users = my_batch_users[0] * np.ones((1, n_each_user))

for i in range(1, nUsersInExample):
    # Movies
    local_list_per_user_movies = ratings['movieId'][ratings['userId'] == my_batch_users[i]].values
    list_movies_each_user[i] = local_list_per_user_movies
    list_movies = np.append(list_movies,local_list_per_user_movies)
    # Ratings
    local_list_per_user_ratings = ratings['rating'][ratings['userId'] == my_batch_users[i]].values
    list_ratings_each_user[i] = local_list_per_user_ratings
    list_ratings = np.append(list_ratings, local_list_per_user_ratings)
    # Users
    n_each_user = local_list_per_user_movies.shape[0]
    local_rep_user =  my_batch_users[i]*np.ones((1, n_each_user))
    list_users = np.append(list_users, local_rep_user)

# show the result of the array
indexes_unique_movies = np.unique(list_movies)
n_movies = indexes_unique_movies.shape[0]

# build a matrix Y and fill according to the data for each user
temp = np.empty((n_movies, nUsersInExample))
temp[:] = np.nan
Y_with_NaNs = pd.DataFrame(temp)
for i in range(nUsersInExample):
 local_movies = list_movies_each_user[i]
 ixs = np.in1d(indexes_unique_movies, local_movies)
 Y_with_NaNs.loc[ixs, i] = list_ratings_each_user[i]

Y_with_NaNs.index = indexes_unique_movies.tolist()
Y_with_NaNs.columns = my_batch_users.tolist()

# convert the form into a form that is appropriate for processing
p_list_ratings = np.concatenate(list_ratings_each_user).ravel()  # to link all the data from the dataset and convert to a single array with a couple of brackets
p_list_ratings_original = p_list_ratings.tolist() # convert to list variable
mean_ratings_train = np.mean(p_list_ratings) # calculate the mean of rating
p_list_ratings =  p_list_ratings - mean_ratings_train # remove the mean
p_list_movies = np.concatenate(list_movies_each_user).ravel().tolist()
p_list_users = list_users.tolist()
Y = pd.DataFrame({'users': p_list_users, 'movies': p_list_movies, 'ratingsorig': p_list_ratings_original,'ratings':p_list_ratings.tolist()})

# Display all the materials
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('max_colwidth',100)

#





