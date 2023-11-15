import pandas as pd
from ThoughtSpace.pca import basePCA

data = pd.read_csv("scratch//data//example_data.csv")

model = basePCA(n_components=4,rotation="promax")

projected_results = model.fit_project(data)
model.save(path="results",pathprefix="PCA_results")


