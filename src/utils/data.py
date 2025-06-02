# Module with functionalities used for data loading.
#
import re
import pandas as pd
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint


def loadRecentFile(in_path: Path, regex: str) -> pd.DataFrame:
    """ Subroutine used to read the most recent parquet file within the specified 
    input directory and applying regular expressions to select only files of interest. """
    try:
        # load the most recent file from mounts
        matched_files = [e.name for e in in_path.iterdir() if re.match(regex, e.name)]
        selected_file = in_path / max(matched_files)
        df = pd.read_parquet(selected_file)
        df = df.fillna(np.nan)

        return df
    
    except ValueError as ex:
        pprint('Exception "{}" possibly due to the fact that no match of {} has been found in {}. Files in directory {}.'.format(
            ex, 
            regex,
            str(in_path),
            matched_files
        ))
        raise ex    
    except Exception as ex:
        pprint('Unhandled exception "{}".'.format(ex))
        raise ex


def selectColumns(df: pd.DataFrame, regex: str) -> pd.DataFrame:
    """ Select dataframe columns based on the input regular expression. """
    return df[[c for c in df.columns if re.match(regex, c)]].copy()


def encodeCategorical(df: pd.DataFrame, regex: str) -> Tuple[pd.DataFrame, dict]:
    """ Encode categorical variables. """
    df = df.copy()

    label_encoders = {}
    for col in df.columns:
        if re.match(regex, col):
            # adjust the label encoder
            label_encoder = LabelEncoderWithMissing().fit(df[col].values)
            label_encoders[col] = label_encoder
            df[col] = label_encoder.transform(df[col].values)

    return df, label_encoders


@dataclass
class Loader:
    path_to_data: str
    mounts_directory: str
    data_version: str
    file_regex: str = None
    splits_directory: str = None

    def loadSplitDF(self) -> pd.DataFrame:
        """ Load the split data. """
        if self.splits_directory is None:
            pprint('No split directory specified. Parameter "splits_directory" is None.')
            return None
        
        # get the path to the splits
        path_to_splits = Path(self.path_to_data) / self.splits_directory / self.data_version

        if not path_to_splits.exists():
            raise FileNotFoundError('Path to splits %s not found' % path_to_splits)
        
        return loadRecentFile(path_to_splits, '.*' if self.file_regex is None else self.file_regex)
        

    def loadDataDF(self) -> pd.DataFrame:
        """ Load the neuroimaging and clinical data information. """
        # get the path to the data
        path_to_mounts = Path(self.path_to_data) / self.mounts_directory / self.data_version

        if not path_to_mounts.exists():
            raise FileNotFoundError('Path to mounts %s not found' % path_to_mounts)
        
        return loadRecentFile(path_to_mounts, '.*' if self.file_regex is None else self.file_regex)


class LabelEncoderWithMissing:
    """
    This encoder behaves similarly to sklearn's LabelEncoder, but it allows NaN values
    to be treated as a separate class and encoded as `np.nan`.

    Methods:
    - fit(y): Learns the unique classes from the data, excluding missing values.
    - transform(y): Transforms the input data into encoded labels.
    - fit_transform(y): Combines fitting and transforming in a single step.
    - inverse_transform(y): Reverts the encoded labels back to their original class labels.

    Attributes:
    - classes_: List of unique class labels.
    - mapping_: Dictionary mapping unique classes to their corresponding encoded values.
    """
    def __init__(self):
        self.classes_ = None
        self.mapping_ = None

    def fit(self, y):
        # extract unique classes excluding missing values
        na_mask = pd.isna(y)
        unique_classes = np.unique(y[~na_mask])

        # create the mapping dictionary
        self.mapping_ = {class_: idx for idx, class_ in enumerate(unique_classes)}

        # save the unique classes for the inverse transform
        self.classes_ = unique_classes

        return self

    def transform(self, y):
        # transform the values to the coded class selecting `np.nan` for missing values
        encoded = np.array([
            self.mapping_.get(val, np.nan) for val in y
        ])

        return encoded

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        # inverse coding using the -1 mark as None (for type cohercion)
        decoded = np.array([
            self.classes_[idx] if not pd.isna(idx) else None for idx in y
        ])
        return decoded
