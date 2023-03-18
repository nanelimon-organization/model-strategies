import re
import pandas as pd
from typing import List
from mintlemon import Normalizer

def normalize_numeric_text_in_dataframe_column(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Normalize numeric text in a pandas dataframe column.

    Replaces numeric text in a specified column of a pandas dataframe with their textual
    equivalents, using the Normalizer.convert_text_numbers() method from the mintlemon-turkish-nlp library.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The pandas dataframe to process.
    column_name : str
        The name of the column to process.
                
    Returns:
    --------
    dataframe : pd.DataFrame
        A new pandas dataframe with the specified column's numeric text replaced with their textual equivalents.

    Examples:
    ---------
    >>> import pandas as pd
    >>> from mintlemon import Normalizer
    >>> df = pd.DataFrame({'text': ['2 milyon suriyeli yerine bu 10 milyon siyahi insanların gelmesine razıyım',
    ...                             'Hiç sayı yok bu metinde']})
    >>> df = normalize_numeric_text_in_dataframe_column(df, "text")
    >>> print(df)
        
       text
    0  iki milyon suriyeli yerine bu on milyon siyahi insanların gelmesine razıyım
    1  Hiç sayı yok bu metinde
    """
    for index, row in dataframe.iterrows():
        cell_text = row[column_name]
        if any(char.isdigit() for char in cell_text):
            words = cell_text.split()
            revised_text = " ".join([Normalizer.convert_text_numbers(word) if word.isdigit() else word for word in words])
            dataframe.at[index, column_name] = revised_text
    return dataframe


def mintlemon_data_preprocessing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Apply various preprocessing steps to the specified column of the input pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    column : str
        The name of the column to preprocess.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame.

    Notes
    -----
    The preprocessing steps applied to the column are as follows:
    1. Remove all punctuations.
    2. Normalize Turkish characters.
    3. Deasciify the text.
    4. Convert all characters to lowercase.
    """
    df[column] = df[column].apply(Normalizer.remove_punctuations)
    df[column] =  df[column].apply(Normalizer.remove_accent_marks)
    #df[column] = df[column].apply(Normalizer.normalize_turkish_chars)
    #df[column] = df[column].apply(Normalizer.deasciify)
    df[column] = df[column].apply(Normalizer.lower_case)

    return df


def remove_short_text(df: pd.DataFrame, min_len: int = 5) -> pd.DataFrame:
    """
    Remove observations from the input DataFrame with short text values based on a minimum length threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with the text column
    min_len : int, optional (default=5)
        The minimum length threshold for text values to be considered valid

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with the short text values removed

    Notes
    -----
    This function removes observations from the input DataFrame where the length of the text value is less than the
    specified minimum length threshold. The function first identifies the indexes of the observations with short text
    values based on the minimum length threshold. Then, the function drops those observations from the input DataFrame.
    """

    result = [index for index, i in enumerate(df.text) if len(str(i)) < min_len]
    df.drop(df.index[result], inplace=True)
    return df


def replace_is_offensive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the value of 'is_offensive' from 1 to 0 for the observation units that meet the following criteria:

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame that contains the 'target' and 'is_offensive' columns

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with the 'is_offensive' values replaced

    Notes
    -----
    This function modifies the input DataFrame by replacing the 'is_offensive' values that meet the criteria.
    The function replaces 'is_offensive' values from 1 to 0 where the 'target' column is 'OTHER' and the 'is_offensive' column is 1.
    """
    idx = df.loc[((df["target"] == "OTHER") & (df["is_offensive"] == 1))].index
    df.loc[idx, "is_offensive"] = 0
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/teknofest_data.csv")
    df = mintlemon_data_preprocessing(df, "text")
    df = normalize_numeric_text_in_dataframe_column(df, "text")
    df = remove_short_text(df, 5)
    df = replace_is_offensive(df)

    df.to_csv("result.csv", index=False)
