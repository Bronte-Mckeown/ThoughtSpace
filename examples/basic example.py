import pandas as pd
from ThoughtSpace.pca import basePCA

data = pd.read_csv("examples/output.csv")

model = basePCA(n_components=4,rotation="promax")

projected_results = model.fit_transform(data)
model.save(path="results",pathprefix="PCA_results")


