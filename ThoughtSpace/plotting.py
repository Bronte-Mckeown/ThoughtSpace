import os
import matplotlib.cm as cm
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
from ThoughtSpace.utils import clean_substrings, returnhighest
from wordcloud import WordCloud
import pandas as pd

def plot_stats(df, path: str):
    means = df.mean()
    stds = df.std()
    question_names = means.index.tolist()
    max_subst = clean_substrings(question_names)
    if max_subst is not None:
        question_names = [x.replace(max_subst, "") for x in question_names]
        
    plt.bar(question_names,means,yerr=stds)
    plt.title("Average responses")
    plt.xlabel("Item")
    plt.ylabel("Average response")
    plt.xticks(rotation = 45, ha='right')
    plt.tight_layout()
    plt.savefig(path + "_responses.png")
    plt.close()
    print('e')
    pass
    

def plot_scree(pca, path: str):
    """
    Plot the scree plot of the PCA.

    :param pca: The PCA object.
    :param path: The path to save the plot.
    """
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(
        PC_values, pca.explained_variance_ratio_ * 100, "o-", linewidth=2, color="blue"
    )
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.savefig(path + "_varexp.png")
    plt.close()
    plt.plot(PC_values, pca.explained_variance_, "o-", linewidth=2, color="blue")
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalues")
    plt.savefig(path + "_eigenvalues.png")
    plt.close()


def save_wordclouds(df: pd.DataFrame, path: str, font: str = "helvetica", n_items_filename: int = 3) -> None:
    """
    This function saves wordclouds to a given path.

    Args:
        df (pd.DataFrame): DataFrame containing the wordclouds.
        path (str): Path to save the wordclouds.
        font (str): Font name for wordclouds.
        n_items_filename (int): Number of items to be included in the filename.

    Returns:
        None
    """
    question_names = df.index.tolist()
    question_names[0] = question_names[0].split("_")[0]
    max_subst = clean_substrings(question_names)
    if max_subst is not None:
        question_names = [x.replace(max_subst, "") for x in question_names]
        df.index = question_names
    for col in df.columns:
        subdf = abs(df[col])
        def _color(x, *args, **kwargs):
            value = df[col][x]
            if value >= 0:
                # Assign red color for positive values
                return "#BB0000"  # Hex code for red
            else:
                # Assign blue color for negative values
                return "#00156A"  # Hex code for blue

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FONT_DIR = os.path.join(BASE_DIR, f'fonts\\{font}.ttf')

        wc = WordCloud(
            font_path= FONT_DIR,
            background_color="white",
            color_func=_color,
            width=400,
            height=400,
            prefer_horizontal=1,
            min_font_size=8,
            max_font_size=200,
        )
        df_dict = subdf.to_dict()
        wc = wc.generate_from_frequencies(frequencies=df_dict)
        highest = returnhighest(df[col], n_items_filename)
        wc.to_file(os.path.join(path, f"{col}_{highest}.png"))

