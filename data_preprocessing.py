import re
import pandas as pd
from typing import List
from mintlemon import Normalizer


class DataPreprocessor:
    """
    A class to preprocess text data in a pandas DataFrame.

    This class provides methods to perform various preprocessing steps on a given pandas DataFrame column
    containing text data. The preprocessing steps include normalizing numeric text, removing punctuations,
    normalizing Turkish characters, converting characters to lowercase, removing short text, and replacing
    specific values in the 'is_offensive' column.

    Attributes
    ----------
    df : pd.DataFrame
        The input pandas DataFrame to preprocess.
    text_column : str
        The name of the column containing the text data to preprocess.

    Methods
    -------
    preprocess() -> pd.DataFrame:
        Apply all preprocessing steps to the input pandas DataFrame and return the preprocessed DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from data_preprocessing import DataPreprocessor
    >>> df = pd.DataFrame({'text': ['2 milyon suriyeli yerine bu 10 milyon siyahi insanların gelmesine razıyım',
    ...                             'Hiç sayı yok bu metinde'],
    ...                             'target': ['OTHER', 'OTHER'],
    ...                             'is_offensive': [0, 1]})
    
    >>> preprocessor = DataPreprocessor(df, "text")
    >>> preprocessed_df = preprocessor.preprocess()
    >>> print(preprocessed_df)
    
       text                                                                             target          is_offensive
    0  iki milyon suriyeli yerine bu on milyon siyahi insanların gelmesine razıyım      OTHER            1
    1                                            Hiç sayı yok bu metinde                OTHER            0
    """
    def __init__(self, df: pd.DataFrame, column: str):
        self.df = df
        self.text_column = column

    def normalize_numeric_text_in_dataframe_column(self) -> pd.DataFrame:
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
        for index, row in self.df.iterrows():
            cell_text = row[self.text_column]
            if any(char.isdigit() for char in cell_text):
                words = cell_text.split()
                revised_text = " ".join([Normalizer.convert_text_numbers(word)if word.isdigit() else word for word in words])
                self.df.at[index, self.text_column] = revised_text

    def mintlemon_data_preprocessing(self) -> None:
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
        self.df[self.text_column] = self.df[self.text_column].apply(Normalizer.remove_punctuations)
        self.df[self.text_column] = self.df[self.text_column].apply(Normalizer.remove_accent_marks)
        #self.df[self.text_column] = self.df[self.text_column].apply(Normalizer.normalize_turkish_chars)
        #self.df[self.text_column] = self.df[self.text_column].apply(Normalizer.deasciify)
        self.df[self.text_column] = self.df[self.text_column].apply(Normalizer.lower_case)

    def remove_short_text(self, min_len: int = 5) -> None:
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
        result = [index for index, i in enumerate(self.df[self.text_column]) if len(str(i)) < min_len]
        self.df.drop(self.df.index[result], inplace=True)
        
    def replace_is_offensive(self) -> None:
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
        idx = self.df.loc[((df["target"] == "OTHER") & (self.df["is_offensive"] == 1))].index
        self.df.loc[idx, "is_offensive"] = 0
    
    def preprocess(self) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the input pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame.
        """
        self.df = self.mintlemon_data_preprocessing()
        self.df = self.normalize_numeric_text_in_dataframe_column()
        self.df = self.remove_short_text()
        self.df = self.replace_is_offensive()
        
        return self.df


if __name__ == "__main__":
    """
    Read a CSV file containing text data, preprocess it using the DataPreprocessor class,
    remove duplicate rows, and save the preprocessed data to a new CSV file.
    
    This script reads a CSV file with text data, creates a DataPreprocessor object with the input
    DataFrame and the name of the text column, preprocesses the text data, and removes duplicate
    rows based on the text column. Finally, the preprocessed and deduplicated DataFrame is saved
    to a new CSV file.
    """
    df = pd.read_csv("data/teknofest_data.csv", sep="|")
    
    preprocessor = DataPreprocessor(df, "text")
    df = preprocessor.preprocess()

    print(df[df.duplicated(subset='text')].count())
    df.drop_duplicates(subset='text', inplace=True)
    print(df[df.duplicated(subset='text')].count())

    df.to_csv("data/result.csv", index=False)


# def convert_offensive_contractions(text):
#     words = text.lower().split()
#     new_text = [dict_ex[word] if word in contraction_conversion_dict else word for word in words]
#     return " ".join(new_text)

# #input_text_1 = "doğduğun günün a.w"
# #input_text_2 = "doğduğun günün a.w"
# #input_text_3 = "doğduğun günün amk"
# #output_text_1 = convert_offensive_contractions(input_text_1)
# #print(output_text_1)

# def convert_offensive_contractions(df, column="text") -> pd.DataFrame:
#     """
#     Replace offensive contractions in the specified DataFrame column.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The DataFrame containing the column to process.
#     column : str
#         The name of the column in the DataFrame to apply the contractions conversion.

#     Returns
#     -------
#     pandas.DataFrame
#         The DataFrame with offensive contractions replaced in the specified column.

#     Examples
#     --------
#     >>> data = {'text': ["doğduğun günün aq", "doğduğun günün a.w"]}
#     >>> df = pd.DataFrame(data)
#     >>> df = convert_offensive_contractions(df, 'text')
    
#        text
#     0  doğduğun günün amına koyayım
#     1  doğduğun günün amına koyayım
#     """
#     df[column] = df[column].apply(
#         lambda text: " ".join(
#             [dict_ex[word] if word in dict_ex else word for word in text.lower().split()]
#         )
#     )
#     return df

data = {'text': ["doğduğun günün aq", "doğduğun günün a.w"]}
df = pd.DataFrame(data)
df = convert_offensive_contractions(df, 'text')
print(df.head())