import re
import pandas as pd
from typing import List
from mintlemon import Normalizer

# def contains_number(text: str) -> bool:
#     """
#     Check whether the input text contains any numerical digits.

#     Parameters
#     ----------
#     text : str
#         The input text to check.

#     Returns
#     -------
#     bool
#         True if the input text contains numerical digits, False otherwise.
#     """
#     if not isinstance(text, str):
#         return False

#     numerical_pattern = re.compile(r"\d+")
#     numerical_matches = numerical_pattern.findall(text)

#     if not numerical_matches:
#         return False
#     else:
#         return True


# def revise_numbers_in_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
#     """
#     Replace numerical digits in the specified column of the input pandas DataFrame.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The input DataFrame.
#     column_name : str
#         The name of the column to revise.

#     Returns
#     -------
#     pandas.DataFrame
#         The revised DataFrame.
#     """
#     for index, row in df.iterrows():
#         text = row[column]
#         if contains_number(text):
#             revised_text = Normalizer.convert_text_numbers(text)
#             df.at[index, column] = revised_text
#     return df

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
    #df[column] = df[column].apply(Normalizer.normalize_turkish_chars)
    #df[column] = df[column].apply(Normalizer.deasciify)
    df[column] =  df[column].apply(Normalizer.remove_accent_marks)
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
    # CSV dosyasını oku ve verileri DataFrame'e yükle
    df = pd.read_csv("data/teknofest_data.csv")
    df = mintlemon_data_preprocessing(df, "text")
    #df = revise_numbers_in_column(df, "text")
    df = remove_short_text(df, 5)
    df = replace_is_offensive(df)

    df.to_csv("result.csv", index=False)
