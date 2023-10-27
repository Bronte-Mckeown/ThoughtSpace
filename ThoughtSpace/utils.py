import operator
import os
from datetime import datetime
from difflib import SequenceMatcher
import pandas as pd
from typing import List

def setupanalysis(
    path: str = None, pathprefix: str = "analysis", includetime: bool = True
) -> str:
    """
    Setup a folder for analysis.

    Args:
        path (str): Path to the folder where the analysis folder should be created.
        pathprefix (str): Prefix for the analysis folder.
        includetime (bool): If True, the analysis folder will be named with a timestamp.

    Returns:
        str: Path to the analysis folder.
    """
    if path is None:
        path = os.getcwd()
    if includetime:
        dt_string = datetime.now().strftime("%d%m%Y_%H-%M-%S")
        path_sub = f"{pathprefix}_{dt_string}"
    else:
        path_sub = f"{pathprefix}"
    if os.path.exists(os.path.join(path, path_sub)):
        print(
            "analysis folder already exists, results in this folder may be overwritten"
        )
    os.makedirs(os.path.join(path, path_sub), exist_ok=True)
    return os.path.join(path, path_sub)


def returnhighest(df: pd.DataFrame, n: int) -> str:
    """
    Returns the top n positive and negative values in a dataframe as a string.

    Args:
        df: A pandas dataframe.
        n: The number of top values to return.

    Returns:
        A string containing the top n positive and negative values.
    """
    posdf = abs(df)
    posdf = posdf.sort_values(ascending=False)
    highvals = posdf[0:n].index.tolist()
    realvals = df.loc[highvals]
    posvals = realvals[realvals > 0]
    highposvals = posvals.sort_values(ascending=False).index.tolist()
    negvals = realvals[realvals < 0]
    highnegvals = negvals.sort_values(ascending=False).index.tolist()
    posoutstring = "_".join(highposvals)
    negoutstring = "_".join(highnegvals)
    if negoutstring == "":
        return f"Positive_{posoutstring}"
    if posoutstring == "":
        return f"Positive_{negoutstring}"
    pmean = posvals.mean()
    nmean = negvals.mean()
    return (
        f"Positive_{posoutstring}_Negative_{negoutstring}"
        if pmean > nmean
        else f"Negative_{negoutstring}_Positive_{posoutstring}"
    )


def clean_substrings(names: List[str]) -> str:
    """
    This function takes a list of strings and returns the longest common substring
    that occurs in all of the strings.
    :param names: List of strings
    :return: Longest common substring
    """
    substring_counts = {}
    for i in range(0, len(names)):
        for j in range(i + 1, len(names)):
            string1 = names[i]
            string2 = names[j]
            match = SequenceMatcher(None, string1, string2).find_longest_match(
                0, len(string1), 0, len(string2)
            )
            matching_substring = string1[match.a : match.a + match.size]
            if matching_substring not in substring_counts:
                substring_counts[matching_substring] = 1
            else:
                substring_counts[matching_substring] += 1
    max_occurring_substring = max(substring_counts.items(), key=operator.itemgetter(1))
    if max_occurring_substring[1] < len(names):
        return None
    return max_occurring_substring[0]
