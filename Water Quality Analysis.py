import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from seaborn import countplot
import plotly.express as px


data =pd.read_csv('water_potability.csv') # reading csv file
#always check the null
data=data.dropna()
data.isnull().sum()

print(data.isnull().sum())

# plt.figure(figsize=(15, 10))
# sns.countplot(data.Potability)
# plt.title("Distribution of Unsafe and Safe Water")
# plt.show()


# Running Sucessfully 
# data = data
# figure = px.histogram(data, x = "ph", 
#                       color = "Potability", 
#                       title= "Factors Affecting Water Quality: PH")
# figure.show()

# data = data
# figure = px.histogram(data, x = "Hardness", 
#                       color = "Potability", 
#                       title= "Factors Affecting Water Quality: PH")
# figure.show()
data = data
# figure = px.histogram(data, x = "Solids", 
#                       color = "Potability", 
#                       title= "Factors Affecting Water Quality: PH")
# figure.show()

# checking the correlation 
correlation = data.corr()
correlation["ph"].sort_values(ascending=False)
print(correlation["ph"].sort_values(ascending=False))

from pycaret.classification import *
clf = setup(data, target = "Potability", session_id = 786)
compare_models()


model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()

# plot_model(model, plot='confusion_matrix')
plot_model(model, plot='feature')
# plot_model(model, plot='learning')

