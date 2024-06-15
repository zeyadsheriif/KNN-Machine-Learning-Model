!pip install -U scikit-learn

#import needed libraries (pandas , numpy , seaborn, matplotlib , sklearn)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#target hna hwa hwa speecies lw astkhdmt aldata mn csv file mn kaggale mgbthash mn scikit learn 3alatol
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
print(type(iris))
iris

# Show the dataset infromation
iris_df.info()

#Show head of dataset
print(iris_df.head())

#Describe the dataset
iris_df.describe()

#check the samples for each class / is it balanced dataset
iris_df.target.value_counts().plot(kind= 'bar');

#value counts of each target (type):
iris_df.target.value_counts()

#check for missing data
print('missing values -> {}'.format (iris_df.isna().sum()))  # -> why ??

#check duplicates
print('dubblicate values -> {}'.format (iris_df.duplicated()))

#drop duplicates
iris_df.drop_duplicates(inplace = True)
#test after remove the duplicates
print(iris_df.duplicated().sum())

##select all rows and all columns except the last one.
X = iris_df.iloc[:,:-1]
#select all rows, but only the last column.
y = iris_df.iloc[:, -1]

#check the X head
X.head()

#check the X tail
X.tail()

#check the y head
y.head()

#check the y tail
y.tail()

#split the data into train and test sets (80,20):
#Shuffle=True, meaning the data will be shuffled before splitting.
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=True, random_state=0)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

#check the traing set size and test set size:
print("Training set size:", len(X_train), "samples")
print("Test set size:", len(X_test), "samples")

#check the traing set shape and test set shape:
print("Training set shape:", X_train.shape, "samples")
print("Test set shape:", X_test.shape, "samples")


def distance_ecu(x_train, x_test_point):
    """
    Calculate the Euclidean distance between a test point and each point in the training data.

    Parameters:
        - x_train: The training data (2D array or DataFrame).
        - x_test_point: The test point (1D array or list).

    Returns:
        - distances: The distances between the test point and each point in the training data (DataFrame).
    """
    distances = []

    # Loop over the rows of x_train
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = 0

        # Loop over the columns of the row
        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - x_test_point[col]) ** 2

        # Calculate the square root of the sum of squared differences
        current_distance = np.sqrt(current_distance)

        # Append the distance to the list of distances
        distances.append(current_distance)

    # Convert distances to a DataFrame
    distances = pd.DataFrame(data=distances, columns=['index'])

    return distances

def nearest_neighbors(distance_point , k):
    """
    Input:
        - distance_point : The distances between the the test point and each point in the training data.
        - K              : The number of neighbors

    Output:
        - df_nearest : The nearest K neighbors between the test point and the training data

    """
    # Sort distances using the sort_values function
    #df_nearest = distance_point.sort_values(by = ['index'], axis = 0 )
    df_nearest = sorted(distance_point)

    ## Take only the first K neighbors
    df_nearest = df_nearest[:k]
    return df_nearest


def voting(df_nearest , y_train):
    """
    Input:
        - df_nearest: Dataframe contains the nearest K neighbors between the Full training dataset and the test point
        - y_train : The labels of the training dataset

    Output:
        - y_pred : The prediction based on Majority Voting

    """
    ## Use the Counter Object to get the labels with K nearest neighbors
    # counter_vote  = Counter(y_train[df_nearest.index])
    counter_vote  = Counter(y_train[df_nearest])
    ## Majority Voting !
    y_pred = counter_vote.most_common(1)[0][0]
    return y_pred


def KNN_from_scratch(X_train, y_train, X_test, K):
    """
    Perform k-nearest neighbors classification from scratch.

    Inputs:
    - x_train: The full training dataset.
    - y_train: The labels of the training dataset.
    - x_test: The full test dataset.
    - k: The number of neighbors to consider.

    Output:
    - y_pred: The predictions for the whole test set based on majority voting.
    """
    y_pred = []
    for i in range(len(X_test)):
          # Loop over all the test set and perform the three steps
        distances = []
        for j in range(len(X_train)):
            distances.append([np.sqrt(np.sum(np.square(X_test[i] - X_train[j]))), j])
        distances.sort(key=lambda x: x[0])
        df_nearest = np.array(distances[:K])[:, 1].astype(int)
        y_pred.append(voting(df_nearest, y_train))
    return y_pred


K = 3
y_pred_scratch = KNN_from_scratch(X_train, y_train, X_test, K)
print(y_pred_scratch)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
print(f'The accuracy of our implementation is {accuracy_scratch*100} %')


K = 5
y_pred_scratch = KNN_from_scratch(X_train, y_train, X_test, K)
print(y_pred_scratch)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
print(f'The accuracy of our implementation is {accuracy_scratch*100} %')


K = 7
y_pred_scratch = KNN_from_scratch(X_train, y_train, X_test, K)
print(y_pred_scratch)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
print(f'The accuracy of our implementation is {accuracy_scratch*100} %')


#the scaler is fitted to the training set / the Normalizer calculates the normalization parameters based on the training set.
scale = Normalizer().fit(X_train)
#the scaler is applied to the training set / this step scales each feature in the training set independently.
x_train_normalized = scale.transform(X_train)
##the scaler is applied to the test set
x_test_normalized = scale.transform(X_test)

print("X train before Normalization")
print(X_train[0:5])
print("\nX train after Normalization")
print(x_train_normalized[0:5])

## Graph Before normalization
# view the relationships between variables; color code by species type
di = {0.0: "Setosa", 1.0: "Versicolor", 2.0 : "Virginica"}
before = sns.pairplot(iris_df.replace({"target": di}), hue='target')
before.fig.suptitle("Pair Plot of the dataset Before normalization", y=1.08)

## Graph after normalization
# view the relationships between variables; color code by species type
iris_df_2 = pd.DataFrame(data = np.c_[iris['data'],iris['target']],
                       columns = iris['feature_names'] + ['target'])
di2 = {0.0: "Setosa", 1.0: "Versicolor", 2.0 : "Virginica"}
after = sns.pairplot(iris_df_2.replace({"target": di}), hue='target')
after.fig.suptitle("Pair Plot of the dataset After normalization", y=1.08)

#check corroleation
correlation_matrix = iris_df.corr()

sns.heatmap(data=iris_df)
plt.title("Heatmap of Features Before Normalization")
plt.show()

sns.heatmap(data=iris_df_2)
plt.title("Heatmap of Features After Normalization")
plt.show()


print("\nAfter normalization:")
k = 3
y_pred_normalized = KNN_from_scratch(X_train, y_train, X_test, k)
accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
print(f"Accuracy for K={k}: {accuracy_normalized*100} %")


print("\nAfter normalization:")
k = 5
y_pred_normalized = KNN_from_scratch(X_train, y_train, X_test, k)
accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
print(f"Accuracy for K={k}: {accuracy_normalized *100} %")


print("\nAfter normalization:")
k = 7
y_pred_normalized = KNN_from_scratch(X_train, y_train, X_test, k)
accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
print(f"Accuracy for K={k}: {accuracy_normalized*100} %")


print("\nAfter normalization:")
for k in [3, 5, 7]:
    y_pred_normalized = KNN_from_scratch(X_train, y_train, X_test, k)
    accuracy_normalized = accuracy_score(y_test, y_pred_normalized)
    print(f"Accuracy for K={k}: {accuracy_normalized*100} %")

