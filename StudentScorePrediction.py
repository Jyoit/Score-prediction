# %%
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# %%
url="http://bit.ly/w-data"
df= pd.read_csv(url)
df.head()

# %%
sns.lmplot(x="Hours",y="Scores", data=df)


# %%
x=df[["Hours"]]
y=df["Scores"]
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)

# %%
model= linear_model.LinearRegression()
model.fit(x_train,y_train)

# %%
x_predicted_score=model.predict(x_train)
import matplotlib.pyplot as plt
plt.scatter(y_train,x_predicted_score)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual Price VS Predicted Price")
plt.show()

# %%

y_predicted_score=model.predict(x_test)
dataf=pd.DataFrame({"Actual Score":y_test,"Predicted Score":y_predicted_score})
dataf

# %%
#predict score if student will study for 9.25 hr/day
hours=[[9.25]]
predict_score=model.predict(hours)
data=pd.DataFrame({"Hours":hours,"Predicted Score":predict_score})
data

# %%
from sklearn import metrics
mae= metrics.mean_absolute_error(y_test,y_predicted_score)
print("Mean Absolute Error : ", mae)


