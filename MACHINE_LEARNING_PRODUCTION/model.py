## importing the required libraries

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## load the data

def load_data(path: str = "/path/to/csv"):
    """
    This function takes a path string to a csv file and load it into the pandas dataframe
    param: path(optional): str,relative path of the csv file

    """
    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# create target variable and independent variables
def create_target_and_independent_variable(
        data: pd.DataFrame=None
        target: str = "estimated_stock_pct"
):
"""
This function takes the pandas dataframe and it splits it into train and test data,This data will be used
train the supervized regression model
param: data:pd.DataFrame, dataframe containing the data for the model
param: target: str(oprional), target variable that we want to predict
return: X: pd.DataFrame
        y: pd.Series
"""
# check to see whether the predictor variable is present in the dataset or not
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the dataset")
    X= data.drop(columns=[target])
    y=data[target]

    return X,y

# train the algorithm
def train_algorithm_with_cross_validation(
        X:pd.DataFrame = None,
        y:pd.Series = None
):
    """
    This function takes the independent variable and dependent variable and trains the random forest regressor
    model across k folds. using cross validation performance metrics will be output for each fold
    :param X:independent variables
    :param y:target variable
    :return:
    """
    #create the list to store the accuracies
    accuracy=[]

    #enter the loop to  run the k folds cross validation
    for fold in range(0,k):
        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

        # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")