import pandas as pd
from ThoughtSpace.pca import groupedPCA

data = pd.read_csv("scratch//data//example_data.csv")

model = groupedPCA("sample")
output = model.fit_project(data)
loadings = model.loadings
model.save(path="results")

model = groupedPCA("age_group")
output = model.fit_project(data)
loadings = model.loadings
model.save(path="results")
