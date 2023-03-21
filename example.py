import pandas as pd
from ThoughtSpace._base import basePCA

data = pd.read_csv("output.csv")
model = basePCA()
output = model.fit_project(data)
loadings = model.loadings
print(f"Loadings: {loadings}, PCA scores: {output}")
