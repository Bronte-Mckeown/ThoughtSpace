import pandas as pd
from ThoughtSpace.pca import basePCA

data = pd.read_csv("examples/output.csv")

model = basePCA(n_components=4,rotation="varimax")

projected_results = model.cv(data)
model.save(path="results",pathprefix="PCA_results")


