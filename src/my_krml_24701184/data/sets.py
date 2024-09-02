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

import os
import pandas as pd

def save_sets(base_path, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
    """
    Saves multiple dataframes to CSV files in the specified directory.

    Parameters
    ----------
    base_path : str
        The directory path where all CSV files will be saved.
    X_train : pd.DataFrame, optional
        Training feature set, by default None.
    y_train : pd.Series, optional
        Training target set, by default None.
    X_val : pd.DataFrame, optional
        Validation feature set, by default None.
    y_val : pd.Series, optional
        Validation target set, by default None.
    X_test : pd.DataFrame, optional
        Test feature set, by default None.
    y_test : pd.Series, optional
        Test target set, by default None.
    """

    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

    for name, data in datasets.items():
        if data is not None:
            save_path = os.path.join(base_path, f"{name}.csv")
            data.to_csv(save_path, index=False)




#--------------------------------------------------------------------------------------------------------------------------------


import os
import pandas as pd

def load_sets(base_path):
    """
    Loads multiple dataframes from CSV files in the specified directory.

    Parameters
    ----------
    base_path : str
        The directory path from where all CSV files will be loaded.

    Returns
    -------
    dict
        A dictionary with dataframe names as keys and dataframes as values.
    """
    datasets = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    loaded_datasets = {}

    for name in datasets:
        file_path = os.path.join(base_path, f"{name}.csv")
        if os.path.exists(file_path):
            if "y_" in name:
                # Load as DataFrame and then convert to Series
                df = pd.read_csv(file_path)
                loaded_datasets[name] = pd.Series(df.iloc[:, 0], name=df.columns[0])
            else:
                # Load as DataFrame
                loaded_datasets[name] = pd.read_csv(file_path)

    return loaded_datasets







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