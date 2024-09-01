import pandas as pd

def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    pd.Series
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target


#--------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import os

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally as CSV files."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

    files_written = []
    
    if X_train is not None:
        X_train.to_csv(os.path.join(path, 'X_train.csv'), index=False)
        files_written.append('X_train.csv')
    if X_val is not None:
        X_val.to_csv(os.path.join(path, 'X_val.csv'), index=False)
        files_written.append('X_val.csv')
    if X_test is not None:
        X_test.to_csv(os.path.join(path, 'X_test.csv'), index=False)
        files_written.append('X_test.csv')
    if y_train is not None:
        y_train.to_csv(os.path.join(path, 'y_train.csv'), index=False, header=False)
        files_written.append('y_train.csv')
    if y_val is not None:
        y_val.to_csv(os.path.join(path, 'y_val.csv'), index=False, header=False)
        files_written.append('y_val.csv')
    if y_test is not None:
        y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False, header=False)
        files_written.append('y_test.csv')

    print(f"Files written: {files_written}")







#--------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import os

def load_sets(path='../data/processed/'):
    """Load the datasets from CSV files."""
    data = {}
    for filename in ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv', 'X_test.csv', 'y_test.csv']:
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            if 'y_' in filename:
                data[filename.replace('.csv', '')] = pd.read_csv(file_path, header=None)
            else:
                data[filename.replace('.csv', '')] = pd.read_csv(file_path)
    return data



#--------------------------------------------------------------------------------------------------------------------------------


def subset_x_y(target, features, start_index:int, end_index:int):
    """Keep only the rows for X and y (optional) sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    features : int
        Index of the starting observation
    features : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """

    return features[start_index:end_index], target[start_index:end_index]

#--------------------------------------------------------------------------------------------------------------------------------

def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(df_copy))

    return X_train, y_train, X_val, y_val, X_test, y_test