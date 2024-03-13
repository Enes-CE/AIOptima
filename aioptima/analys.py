import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import LabelEncoder


def load_file(path: str, extension: str) -> pd.DataFrame:
    """

     Loads the data from the specified file path based on the given extension.

    Parameters:
        path (str): A string representing the file path.
        extension (str): A string representing the file extension.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.

    Examples:
        >>> load_file("data", "csv")
           Column1  Column2  Column3
        0        1        2        3
        1        4        5        6
        2        7        8        9

        >>> load_file("data", "xlsx")
           Column1  Column2  Column3
        0        1        2        3
        1        4        5        6
        2        7        8        9

        >>> load_file("data", "json")
           Column1  Column2  Column3
        0        1        2        3
        1        4        5        6
        2        7        8        9

    """
    full_path = path + "." + extension
    try:
        if extension == "csv":
            data = pd.read_csv(full_path)
        elif extension == "xlsx":
            data = pd.read_excel(full_path)
        elif extension == "json":
            data = pd.read_json(full_path)
        else:
            print("Unsupported file extension:", extension)
            return None
        return data
    except FileNotFoundError:
        print("File not found:", full_path)
        return None


def set_dataview():
    """
    Sets the display options for pandas to enhance data viewing.

    This function sets the following display options:
    - 'display.max_columns': None, which displays all columns in DataFrames
    - 'display.width': 500, which sets the output width to 500 characters

    Parameters:
        None

    Returns:
        None

    Example:
        >>> set_dataview()
        # After calling this function, the display options are set
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)


def overview(dataframe: pd.DataFrame, definition: int = 5):
    """

    Displays a general overview of the given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): The DataFrame for which the overview will be displayed.
        - definition (int): The number of rows initially displayed. Default is 5.

    Outputs:
        The function prints the head, tail, info, statistics, missing values, columns,
        whether there are any empty data, and a sample row of the DataFrame.

    """
    try:
        print("##################### Head #####################")
        print(dataframe.head(definition))
        print("\n##################### Tail #####################")
        print(dataframe.tail(definition))
        print("\n##################### Info #####################")
        dataframe.info()
        print("\n##################### Statistics #####################")
        print(dataframe.describe().T)
        print("\n##################### NA #####################")
        print(dataframe.isnull().sum())
        print("\n##################### Columns #####################")
        print(dataframe.columns)
        print("\n##################### Empty Data #####################")
        print(dataframe.isnull().values.any())
        print("\n##################### Sample #####################")
        print(dataframe.sample())
    except FileNotFoundError:
        print("DataFrame not found:", dataframe)


def column_detection(dataframe: pd.DataFrame, cat_th: int = 10, car_th: int = 20):
    """

    Identifies the columns in the given pandas DataFrame based on their types and detects
    categorical, numerical, or special cases.

    Parameters:
        - dataframe (pd.DataFrame): The DataFrame where the column types will be identified.
        - cat_th (int): Maximum number of unique values for a column to be considered categorical.
          Default is 10.
        - car_th (int): Minimum number of unique values for a column to be considered categorical
          but cardinal. Default is 20.

    Outputs:
        The function prints the categorical columns, numerical columns, categorical but
        cardinal columns, and numerical but categorical columns in the DataFrame.
        It also returns an overview of the columns.

    """
    try:
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print("\n##################### Categoric #####################")
        print(cat_cols)
        print("\n##################### Numeric #####################")
        print(num_cols)
        print("\n##################### Categoric But Cardinal #####################")
        print(cat_but_car)
        print("\n##################### Numeric But Categoric #####################")
        print(num_but_cat)
        print("\n##################### Columns Overview #####################")
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car, num_but_cat
    except Exception as e:
        print("An error occurred:", e)
        return None, None


def outlier_thresholds(dataframe: pd.DataFrame, col_name: str, q1=0.25, q3=0.75):
    """

    Calculates the lower and upper limits of outliers for a specific column in the given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): The DataFrame where the lower and upper limits of outliers will be calculated.
        - col_name (str): The name of the column for which the lower and upper limits of outliers will be calculated.
        - q1 (float): The quartile value to be used for calculating the lower limit. Default is 0.25.
        - q3 (float): The quartile value to be used for calculating the upper limit. Default is 0.75.

    Outputs:
        The function calculates the lower and upper limits of outliers for the specified column
        and returns these limits.

    """
    try:
        if col_name not in dataframe.columns:
            raise ValueError(f"{col_name} is not a column in the DataFrame.")

        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit
    except Exception as e:
        print("An error occurred:", e)
        return None, None


def check_outlier(dataframe: pd.DataFrame, col_name: str):
    """

    Checks if there are outliers in a specific column of the given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): The DataFrame where the outliers will be checked.
        - col_name (str): The name of the column for which the outliers will be checked.

    Outputs:
        - Returns True if there are outliers. Returns False if there are no outliers.
          Returns None if an error occurs.

    """
    try:
        if col_name not in dataframe.columns:
            raise ValueError(f"{col_name} is not a column in the DataFrame.")

        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(
                axis=None):
            return True
        else:
            return False
    except Exception as e:
        print("An error occurred:", e)
        return None


def outliers_themselves(dataframe: pd.DataFrame, col_name: str, index: bool = True):
    """

    Returns or prints the outliers for a specific column in the given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): The DataFrame where the outliers will be returned or printed.
        - col_name (str): The name of the column for which the outliers will be checked.
        - index (bool): If True, the indices of the outliers will be returned. If False, they will not be returned.
                        Default is True.

    Outputs:
        - Returns the outliers themselves or their indices in the DataFrame. Returns None if there are no outliers.


    """
    try:
        if col_name not in dataframe.columns:
            raise ValueError(f"{col_name} is not a column in the DataFrame.")

        low, up = outlier_thresholds(dataframe, col_name)

        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
        else:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

        if index:
            outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
            return outlier_index
    except Exception as e:
        print("An error occurred:", e)
        return None


def missing_values_table(dataframe: pd.DataFrame, na_name: bool = False):
    """

    Analyzes missing values in a given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame in which missing values will be analyzed.
        - na_name (bool): If True, returns the column names with missing values. Default is False.

    Outputs:
        - Prints a DataFrame containing the number and percentage of missing values.
        - If na_name is True, returns the column names with missing values.

    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def convert_numeric_to_categorical(dataframe: pd.DataFrame, columns: list):
    """

    Converts specific columns in a given pandas DataFrame from numeric to categorical data.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame in which the columns will be converted.
        - columns (list): List of column names to be converted.

    Outputs:
        - DataFrame with the converted columns.

    """
    for column in columns:
        dataframe[column] = dataframe[column].astype('category')
    return dataframe


def feature_scaling(dataframe: pd.DataFrame, columns: list, method: str):
    """

    Scales specific columns in a given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame in which the columns will be scaled.
        - columns (list): List of column names to be scaled.
        - method (str): Scaling method. Can be 'standard', 'robust', 'minmax', or 'normalize'.

    Outputs:
        - DataFrame with the scaled columns.

    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'normalize':
        scaler = Normalizer()
    else:
        raise ValueError("Invalid scaling method. Must be 'standard', 'robust', 'minmax' or 'normalised'.")

    dataframe[columns] = scaler.fit_transform(dataframe[columns])
    return dataframe


def missing_vs_target(dataframe: pd.DataFrame, target: str, na_columns: list):
    """

    Analyzes the relationship between missing values in specific columns and the target variable in a given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the missing values and the target variable.
        - target (str): Name of the target variable.
        - na_columns (list): List of column names with missing values.

    Outputs:
        - Prints the results of the relationship between missing values in specific columns and the target variable.

    """
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1,
                                             0)  # eksik değere sahip olanlar -> 1 olmayanlar -> 0

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe: pd.DataFrame, binary_col: str):
    """

    Converts a binary categorical column in a given pandas DataFrame using label encoding.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame in which the column will be converted.
        - binary_col (str): Name of the binary categorical column to be converted.

    Outputs:
        - DataFrame with the column converted using label encoding.

    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe: pd.DataFrame, categorical_cols: list, drop_first: bool = True):
    """

    Converts categorical columns in a given pandas DataFrame using one-hot encoding.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame in which the columns will be converted.
        - categorical_cols (list): List of column names to be converted.
        - drop_first (bool): Determines whether to drop the first column. Default is True.

    Outputs:
        - DataFrame with the columns converted using one-hot encoding.

    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe: pd.DataFrame, target: str, cat_cols: list):
    """

    Analyzes rare classes in categorical columns of a given pandas DataFrame.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame in which the rare classes will be analyzed.
        - target (str): Name of the target variable.
        - cat_cols (list): List of column names to analyze the rare classes.

    Outputs:
        - Prints the results of the analysis of rare classes.

    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
