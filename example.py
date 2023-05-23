import pandas as pd
from ThoughtSpace._base import basePCA
from sklearn.model_selection import train_test_split
data = pd.read_csv("scratch//data//output.csv")
model = basePCA(n_components=4,rotation=False)

projected_results = model.fit_project(data)
model.save(path="results",pathprefix="PCA_results")


