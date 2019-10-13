import pods
import zipfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

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

# Steepest Descent Algorithm
# To start with though, we need initial values for the matrix  𝐔  and the matrix  𝐕 . Let's create them as pandas data
# frames and initialise them randomly with small values.
#
#
q = 2  #
learn_rate = 0.01
U = pd.DataFrame(np.random.normal(size=(nUsersInExample, q))*0.001, index=my_batch_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.001, index=indexes_unique_movies)

# now we set the objective function and finish the gradient function
def objective_gradient(Y, U, V):
    gU = pd.DataFrame(np.zeros((U.shape)), index=U.index)
    gV = pd.DataFrame(np.zeros((V.shape)), index=V.index)
    obj = 0.
    nrows = Y.shape[0]
    for i in range(nrows): # circulate the read and store function
        row = Y.iloc[i]    # read and store all the information
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        prediction = np.dot(U.loc[user], V.loc[film]) # vTu for inner product the dot calculate the corresponding multiply result
        diff = prediction - rating # vTu - y
        obj += diff*diff
        gU.loc[user] += 2*diff*V.loc[film]   # corresponding gradient result
        gV.loc[film] += 2*diff*U.loc[user]
    return obj, gU, gV
iterations = 20
for i in range(iterations):
    obj, gU, gV = objective_gradient(Y, U, V)
    print("Iteration", i+1 , "Objective function: ", obj)
    U -= learn_rate*gU   # update the result when you finish a gradient you have to update ot once
    V -= learn_rate*gV

# TODO: Create a function that provides the prediction of the ratings for the users in the dataset. Is the quality of
#  the predictions affected by the number of iterations or the learning rate? The function should receive Y, U and V
#  and return the predictions and the absolute error between the predictions and the actual rating given by the users.
#  The predictions and the absolute error should be added as additional columns to the dataframe Y.
def prediction(Y, U, V):
    pred_df = pd.DataFrame(index=Y.index, columns=['prediction'])
    abs_error_df = pd.DataFrame(index=Y.index, columns=['absolute error'])
    for i in Y.index:
        row = Y.iloc[i]
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        pred_df.loc[i] = np.dot(U.loc[user], V.loc[film])  # vTu
        abs_error_df.loc[i] = abs(pred_df.iloc[i, 0] - rating)

    return pred_df, abs_error_df


pred_df, abs_error_df = prediction(Y, U, V)
Y['prediction'] = pred_df
Y['absolute error'] = abs_error_df

# TODO: Stochastic gradient descent involves updating separating each gradient update according to each separate
#   observation, rather than summing over them all. It is an approximate optimization method, but it has proven
#   convergenceunder certain conditions and can be much faster in practice. It is used widely by internet companies
#   for doing machine learning in practice. For example, Facebook's ad ranking algorithm uses stochastic gradient
#   descent.
'''
Create a stochastic gradient descent version of the algorithm. Monitor the objective function after every 1000 updates 
to ensure that it is decreasing. When you have finished, plot the movie map and the user map in two dimensions (you can 
use the columns of the matrices  𝐔  for the user map and the columns of  𝐕  for the movie map). Provide three 
observations about these maps.
'''
# calculate the obj function (first you should make your prediction function and initial it when )
def create_obj(Y,U,V):
    obj = 0.0
    num_inter = Y.shape[0]
    for i in range(num_inter):
        each_row = Y.iloc[i]
        user = each_row['users']
        movie = each_row['movies']
        rating = each_row['ratings']
        prediction = np.dot(U.loc[user], V.loc[movie])  # to calculate the
        diff = prediction - rating
        obj = diff * diff
    return obj

# the gradient function to use the rating and u, v to gradient
# we just indicate the function of gradient.
def obj_gradient(rating, u, v):
    prediction_value = np.dot(u,v)
    diff = prediction_value - rating
    gU = 2*diff*v
    gV = 2*diff*u
    return gU,gV

def check_converge(new_obj, pre_obj):

    if pre_obj == None or new_obj < pre_obj:
        return False
    elif new_obj > pre_obj:
        return True


def map_function(Y, U, V, learn_rate = 0.001, period_iteration = 1000, loop_time = 100):
    index_interation = 0
    obj_func = None
    converge_label = False
# create list to storage the obj and iteration process
    obj_value = []
    loopnum_update = []
    currentlabel = 0

    index_list = Y.index.values
    for i in range(loop_time):
        # start to iteration
        np.random.shuffle(index_list)
        # converge_label is to check if the gradient is convergence
        if converge_label:
            break
        # now we start the gradient 1000 times and shuffle the vector to update the inner product value
        #  we set the number of update each row of value is 1000 times, when update the random the matrix we check if the
        # gradient function is becoming the convergence. if not ,we continue the gradient process
        for j in index_list:
            row = Y.iloc[j]
            user = row['users']
            movie = row['movies']
            rating = row['ratings']
            gU, gV = obj_gradient(rating, U.loc[user], V.loc[movie])
            U.loc[user] -= learn_rate*gU
            V.loc[movie] -= learn_rate * gV
            index_interation += 1
            # if finish the each the 1000
            if index_interation % period_iteration == 0:
                # generate the new obj_function value
                obj_newvalue = create_obj(Y, U, V)
                loopnum_update.append(index_interation)
                obj_value.append(obj_newvalue)
                print("After the %n times iteration, the value of objective function is %0.8f",(index_interation, obj_newvalue))
                label = check_converge(obj_newvalue, obj_value[currentlabel])
                if label :
                    break
                else:
                    currentlabel += 1
        plt.plot(loopnum_update, obj_value, 'b-.')
        plt.title('Objectives over updates')
        plt.save('./objective_gradient.png')
        return U,V     # update the value

def map_vector(U,V):
    plt.plot(U[0],U[1],'bx')
    plt.title('Users Map')
    plt.save('./usermap.png')

    plt.plot(V[0], V[1], 'bx')
    plt.title('Movie Map')
    plt.save('./moviemap.png')

# we start all three function to
q = 2
U = pd.DataFrame(np.random.normal(size=(nUsersInExample, q)) * 0.001, index=my_batch_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q)) * 0.001, index=indexes_unique_movies)

U,V = map_function(Y, U, V)
map_vector(U,V)



# TODO: Use stochastic gradient descent to make a movie map for the MovieLens 100k data. Plot the map of the movies
#  when you are finished.

# use the panda to store the new data for stochastic gradient
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
Y_all = pd.DataFrame({'users': ratings['userId'], 'movies': ratings['movieId'], 'ratingsorig': ratings['rating']})
# set a new value for ratings and calculate the Mean Deviation
Y_all['rating'] = Y_all['ratingsorig'] - np.mean(Y_all['ratingsorig'])
# select all the distinctive users
user_dis = Y_all['users'].unique()
num_user = user_dis.shape[0]
movie_dis = Y_all['movies'].unique()
num_movie = movie_dis.shape[0]

q = 2
U = pd.DataFrame(np.random.normal(size=(n_users, q))*0.001, index=indexes_unique_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.001, index=indexes_unique_movies)

U, V = map_function(Y_all, U, V)













