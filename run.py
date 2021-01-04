# imports
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import csv
import pandas as pd
import numpy as np
import math
import datetime
from numpy import linalg
from surprise import KNNBaseline, BaselineOnly
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import Reader
import matplotlib.pyplot as plt
import pickle

# Import for neural nets
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
from IPython.display import SVG
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from keras.regularizers import l2
from keras.models import model_from_json


# Plot error progression
def plot(errors, names=[]):
    
    for error in errors:
        x, y = zip(*error)
        plt.plot(np.array(x), np.array(y), 'o-')
    
    plt.legend(names, loc='upper left')
    plt.ylabel('RMSE')
    plt.xlabel('n factors')

plt.show()

def save(liste, name="errors"):
    with open(name, 'wb') as fp:
        pickle.dump(liste, fp)

def read(name="errors"):
    with open (name, 'rb') as fp:
        return pickle.load(fp)

# Create submission given an algo setup from surprise
def create_submission(algo):
    data = Dataset.load_from_df(df[['people', 'movies', 'Prediction']], reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    submission = toPredict[['people', 'movies', 'Prediction']].values.tolist()
    peoples, movies, _ = list(map(list, zip(*submission)))
    submission_pred = algo.test(submission)
    data = {'Id': ['r' + str(submission_pred[0].iid) + '_c' + str(submission_pred[0].uid)], 'Prediction': [str(submission_pred[0].est)]}
    for i in range(1, len(submission_pred)):
        data['Id'].append('r{0}_c{1}'.format(movies[i], peoples[i]))
        data['Prediction'].append(str(submission_pred[i].est))
    
    submission_df = pd.DataFrame(data).rename(columns={0 : 'Id', 1 : 'Prediction'})
    submission_df.Prediction = [round(float(val)) for val in submission_df.Prediction.values]
    submission_df.to_csv(relative_path + '/submission.csv')

# Create submission
def submission_nn(model):
    submission = toPredict[['people', 'movies', 'Prediction']].values.tolist()
    peoples, movies, _ = list(map(list, zip(*submission)))
    
    grades = [max(min(c[0], 5), 1) for c in np.round(model.predict([peoples, movies]),0)]
    
    data = {'Id': [], 'Prediction': []}
    for i in range(len(grades)):
        data['Id'].append('r{0}_c{1}'.format(movies[i], peoples[i]))
        data['Prediction'].append(str(grades[i]))
    
    submission_df = pd.DataFrame(data).rename(columns={0 : 'Id', 1 : 'Prediction'}).set_index("Id")
    submission_df.Prediction = [round(float(val)) for val in submission_df.Prediction.values]
    submission_df.to_csv(relative_path + '/submission.csv')
    return submission_df

# Simple network
def dot_product_network(dropout=0.5):
    # Matrix factorisation in Keras
    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
    
    features = int(60*(1 + dropout))
    
    # Creating movie input
    movie_input = keras.layers.Input(shape=[1],name='Item')
    
    movie_embedding = keras.layers.Embedding(n_movies + 1, features, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
    movie_drop = keras.layers.Dropout(dropout)(movie_vec)
    
    # Creating user input
    user_input = keras.layers.Input(shape=[1],name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, features, name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
    user_drop = keras.layers.Dropout(dropout)(user_vec)
    
    # Merging
    prod = keras.layers.Dot(axes=1)([movie_drop, user_drop])
    
    model = keras.Model([user_input, movie_input], prod)
    model.compile('adam', 'mean_squared_error')
    
    return model

# More complex network (allowing non-linear combining of features vectors)
def concat_network(dropout=0.3):
    n_latent_factors_user = int(60*(1+dropout))
    n_latent_factors_movie = int(60*(1+dropout))
    
    n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
    
    movie_input = keras.layers.Input(shape=[1],name='Item')
    movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
    movie_vec = keras.layers.Dropout(dropout)(movie_vec)
    
    
    user_input = keras.layers.Input(shape=[1],name='User')
    user_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
    user_vec = keras.layers.Dropout(dropout)(user_vec)
    
    concat = keras.layers.Concatenate()([movie_vec, user_vec])
    dense = keras.layers.Dense(200,name='FullyConnected')(concat)
    dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
    dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dropout_1)
    
    
    result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
    adam = Adam(lr=0.005)
    model = keras.Model([user_input, movie_input], result)
    model.compile(optimizer=adam,loss= 'mean_absolute_error')
    
    return model


# Plot learning curves
def plot_history(name):
    history = read(name)
    pd.Series(history['loss']).plot(logy=True)
    ax = pd.Series(history['val_loss']).plot(logy=True)
    ax.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Train Error")
    plt.plot()

def test_model(df, model):
    y_hat = [max(min(c[0], 5), 1) for c in np.round(model.predict([df.user_id, df.item_id]),0)]
    y_true = df.rating
    print("MAE: {}".format(mean_absolute_error(y_true, y_hat)))

def save_model(model, name):
    model_json = model.to_json()
    with open("{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("{}.h5".format(name))

def read_model(name):
    json_file = open('{}.json'.format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("{}.h5".format(name))
    return loaded_model


##### LOADING DATA FOR SUPRISE ALGORITHMS

relative_path = os.getcwd() + "/data"

df = pd.read_csv(relative_path + '/data_train.csv').set_index("Id")
movies, people = zip(*[tuple(int(x[1:]) for x in x.split("_")) for x in df.index.values])
df["movies"] = movies
df["people"] = people
df = df.reset_index().drop(columns=["Id"])

predictions = pd.read_csv(relative_path + '/sample_submission.csv').set_index("Id")
toPredict = predictions.copy()
movies, people = zip(*[tuple(int(x[1:]) for x in x.split("_")) for x in toPredict.index.values])
toPredict["movies"] = movies
toPredict["people"] = people
toPredict = toPredict.reset_index().drop(columns=["Id"])

# Initializing Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['people', 'movies', 'Prediction']], reader)

#Parameters testing
factors = np.linspace(50, 175, num = 6)
learning_rates = np.linspace(0.05 , 0.2, num = 7)
regularization_rates = np.linspace(0.0025, 0.01, num = 7)

# Nombre moyen de movies par people
tmp = df.groupby(df.people).count()
res = sum(tmp.movies.values)/tmp.count()[0]
print("En moyenne, on a {} movies per people".format(res))

# Nombre moyen de movies par people
tmp = df.groupby(df.movies).count()
res = sum(tmp.people.values)/tmp.count()[0]
print("En moyenne, on a {} people per movie".format(res))

print("{0} movies and {1} users".format(len(df.movies.unique()), len(df.people.unique())))


##### LOADING DATA FOR NEURAL NETWORK

relative_path = os.getcwd() + "/data"

dataset = pd.read_csv(relative_path + '/data_train.csv').set_index("Id")
movies, people = zip(*[tuple(int(x[1:]) for x in x.split("_")) for x in dataset.index.values])
dataset["item_id"] = movies
dataset["user_id"] = people
dataset = dataset.reset_index().drop(columns=["Id"]).rename(columns={"Prediction":"rating"})

def create_best_svd():
    algo = SVD(n_epochs=1, n_factors=60, lr_all=0.005, reg_all=0.1, verbose=True)
    create_submission(algo)

def create_best_nn():
    model = concat_network()
    model.fit([dataset.user_id, dataset.item_id], dataset.rating, epochs=30, verbose=1)
    submission_nn(model)

#### CREATE SUBMISSION FILE
create_best_svd()


