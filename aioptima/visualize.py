import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

matplotlib.use('TkAgg', force=True)


def single_categorical_var_visualize(dataframe:pd.DataFrame, cat_column:str):
    """

    Visualizes the frequency distribution of a single categorical variable in a bar chart.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing the categorical variable.
        cat_column (str): The name of the categorical column in the dataframe.

    Returns:
        None

    Example:
        data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B']})
        single_categorical_var_visualize(data, 'Category')

    """
    dataframe[cat_column].value_counts().plot(kind="bar")
    plt.show(block=True)


def single_numeric_var_visualize(dataframe: pd.DataFrame, num_column: str, chart_type: str):
    """

    Visualizes the distribution or box plot of a single numeric variable in a DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the numeric variable.
        num_column (str): The name of the numeric column in the DataFrame.
        chart_type (str): The type of chart to be plotted. It can be "hist" for a histogram or "boxplot" for a box plot.

    Returns:
        None

    Example:
        data = pd.DataFrame({'Value': [10, 20, 30, 40, 50, 60]})
        single_numeric_var_visualize(data, 'Value', 'hist')

    """
    if chart_type == "hist":
        plt.hist(dataframe[num_column])
        plt.xlabel(num_column)
        plt.title(num_column)
        plt.show(block=True)
    elif chart_type == "boxplot":
        plt.boxplot(dataframe[num_column])
        plt.hist(dataframe[num_column])
        plt.xlabel(num_column)
        plt.show(block=True)


def cat_summary(dataframe, cat_cols=None, plot=True):
    """

    Prints a summary of categorical variables in the given DataFrame and optionally plots them.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the categorical variables.
        cat_cols (list or str): The list of categorical columns to summarize or a single column name.
            If set to None (default), columns of type 'object' or 'category' in the DataFrame are automatically selected.
        plot (bool): A flag indicating whether to plot or not. Default is True.

    Returns:
        None

    Example:
        data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B', 'B'],
                             'Color': ['Red', 'Blue', 'Green', 'Blue', 'Green', 'Red']})
        cat_summary(data, cat_cols=['Category', 'Color'], plot=True)

    """
    if cat_cols is None:
        cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    elif isinstance(cat_cols, str):
        cat_cols = [cat_cols]

    for cat_col in cat_cols:
        print(pd.DataFrame({cat_col: dataframe[cat_col].value_counts(),
                            "Ratio": 100 * dataframe[cat_col].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[cat_col], data=dataframe)
            plt.show(block=True)
