import pandas as pd
from ThoughtSpace._base import groupedPCA

data = pd.read_csv("scratch//data//output.csv")
model = groupedPCA("Id_number")
output = model.fit_project(data)
loadings = model.loadings
print(f"Loadings: {loadings.to_markdown()}, PCA scores: {output.to_markdown()}")

model = groupedPCA("Task_name")
output = model.fit_project(data)
loadings = model.loadings
print(f"Loadings: {loadings.to_markdown()}, PCA scores: {output.to_markdown()}")
