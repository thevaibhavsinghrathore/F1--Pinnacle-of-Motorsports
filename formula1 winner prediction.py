
# # **Pinnacle of Motorsports: F1**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import streamlit as st

# %% [markdown]
# ### **Loading Data**

# %%
races = pd.read_csv("./data/races.csv",na_values='\\N')
drivers = pd.read_csv("./data/drivers.csv",na_values='\\N')
constructors = pd.read_csv("./data/constructors.csv",na_values='\\N')
results = pd.read_csv("./data/results.csv",na_values='\\N')
circuits = pd.read_csv("./data/circuits.csv",na_values='\\N')
qualifying = pd.read_csv("./data/qualifying.csv",na_values='\\N')
status = pd.read_csv("./data/status.csv",na_values='\\N')
drivers_standing = pd.read_csv("./data/driver_standings.csv",na_values='\\N')
constructors_standings = pd.read_csv("./data/constructor_standings.csv",na_values='\\N')
pit_stops = pd.read_csv("./data/pit_stops.csv",na_values='\\N')
lap_times = pd.read_csv("./data/lap_times.csv",na_values='\\N')
sprint_results = pd.read_csv("./data/sprint_results.csv",na_values='\\N')




# %% [markdown]
# ### **Merging Data**

# %%
drivers = drivers.drop(columns=['url'])
constructors = constructors.drop(columns=['url'])
circuits = circuits.drop(columns=['url'])
f1 = results.merge(races, on='raceId') \
.merge(drivers, on='driverId') \
.merge(constructors, on='constructorId') \
.merge(circuits, on='circuitId') \
.merge(status, on='statusId', how= 'left')


# %% [markdown]
# ### **Adding Qualifying Data**

# %%
qualifying = qualifying[['raceId', 'driverId', 'position']].rename(columns={'position':'qualifying_position'})
f1 = f1.merge(qualifying, on=['raceId', 'driverId'], how='left') 


# %% [markdown]
# ### **Adding Driver Standings**

# %%
drivers_standing = drivers_standing[['raceId', 'driverId', 'points']].rename(columns={'points': 'driver_points'})
f1 = f1.merge(drivers_standing, on= ['raceId', 'driverId'], how='left')

# %% [markdown]
# ### **Adding Constructor Standings**

# %%
constructors_standings = constructors_standings[['raceId', 'constructorId', 'points']].rename(columns={'points': 'constructor_points'})
f1 = f1.merge(constructors_standings, on=['raceId', 'constructorId'], how='left')

# %% [markdown]
# ### **Adding Average Laptime**

# %%
lap_times['lap_time'] = lap_times['milliseconds']
avg_lap_times = lap_times.groupby(['raceId', 'driverId'])['lap_time'].mean().reset_index()
avg_lap_times.rename(columns={'lap_time': 'avg_lap_time'}, inplace= True)
f1 = f1.merge(avg_lap_times, on=['raceId', 'driverId'], how='left')

# %% [markdown]
# ### **Adding PitStops**

# %%
pit_stop_counts = pit_stops.groupby(['raceId', 'driverId']).size().reset_index(name='num_of_pitstops')
f1 = f1.merge(pit_stop_counts, on=['raceId','driverId'], how='left')

# %% [markdown]
# ### **Adding Sprint Results**

# %%
sprint_results = sprint_results[['raceId', 'driverId','position']].rename(columns={'position': 'sprint_position'})
f1 = f1.merge(sprint_results, on=['raceId', 'driverId'], how='left')

# %% [markdown]
# ### **FINAL PROCESSING**

# %%
f1['race_year'] = pd.to_datetime(f1['date']).dt.year
f1 = f1.rename(columns={
  'round' : 'round',
  'name_x' : 'race_name',
  'name_y' : 'driver_name',
  'name' : 'constructor_name',
  'positionOrder': 'positionOrder'
})

# %%
columns_to_use = [
    'race_year', 'round', 'circuitRef', 'driverRef', 'constructorRef',
    'grid', 'positionOrder','status','qualifying_position',
    'driver_points', 'constructor_points', 'avg_lap_time', 'num_of_pitstops', 'sprint_position'
]

df = f1[columns_to_use].dropna()

# %% [markdown]
# ### **Creating a Target Variable**

# %%
df['win'] = (df['positionOrder']==1).astype(int)

# %% [markdown]
# ### **Encoding Categorical Features**

# %%
from sklearn.calibration import LabelEncoder


status_encoder = LabelEncoder()
df['status'] = status_encoder.fit_transform(df['status'].astype(str))
df_encoded = pd.get_dummies(df, columns=['circuitRef', 'driverRef', 'constructorRef'])

# %% [markdown]
# ### **Defining Features and Targets**

# %%
X = df_encoded.drop(['positionOrder', 'win'], axis =1)
Y = df_encoded['win']

# %% [markdown]
# ### **Train/Test Split**

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)

# %% [markdown]
# ### **Train Model**

# %%
model = RandomForestClassifier(n_estimators=100, random_state=21)
model.fit(X_train, Y_train)

# %% [markdown]
# ### **Predict and Evaluate**

# %%
Y_pred = model.predict(X_test)
st.title("F1 Race Winner Prediction")
st.subheader("Model Performance")
st.write(f"Accuracy:{accuracy_score(Y_test,Y_pred):.2f}")
st.text("Classification Report")
st.text(classification_report(Y_test, Y_pred))

# %% [markdown]
# ### **Feature Importance**

# %%
importances = model.feature_importances_
indicies = np.argsort(importances)[-10:]
fig, ax = plt.subplots()
ax.barh(range(len(indicies)), importances[indicies], align = "center")
ax.set_yticks(range(len(indicies)))
ax.set_yticklabels([X.columns[i] for i in indicies])
ax.set_title("Top 10 Feature Importances")
st.pyplot(fig)

# %% [markdown]
# ### **Additional Visualizer**

# %%
st.subheader("Feature Analysis")

# %% [markdown]
# ### **Correlation Heatmap**

# %%
st.write("### Correlation Heatmap")
corr = df[['driver_points', 'constructor_points', 'qualifying_position', 'avg_lap_time', 'num_of_pitstops', 'sprint_position', 'win']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# %% [markdown]
# ### **Distribution plots for key features**

# %%
for feature in ['driver_points', 'constructor_points', 'qualifying_position', 'avg_lap_time', 'num_of_pitstops']:
  st.write(f"Distribution of {feature} by Win Status")
  fig, ax = plt.subplots()
  sns.histplot(data=df, x=feature, hue='win', multiple='stack', kde=True, ax=ax)
  st.pyplot(fig)

# %% [markdown]
# ## **Save Model**

# %%
joblib.dump(model, 'f1_winner_model.pkl')

# %% [markdown]
# ### **Streamlit Input**

# %%
st.header("Predict Future Race Winner")
circuits = df['circuitRef'].unique()
drivers = df['driverRef'].unique()
constructors = df['constructorRef'].unique()

race_year = st.number_input("Race Year", value=2025)
round_num = st.number_input("Round", value=1)
grid = st.number_input("Grid Position", value=1)
qual_pos = st.number_input("Qualifying Position", value=1)
driver_pts = st.number_input("Driver Points", value=0.0)
constructor_pts = st.number_input("Constructor Points", value=0.0)
avg_lap_time = st.number_input("Avg Lap Time (ms)", value=90000.0)
num_pits = st.number_input("Number of Pit Stops", value=2)
sprint_pos = st.number_input("Sprint Position", value=1)
driver = st.selectbox("Driver", sorted(drivers))
constructor = st.selectbox("Constructor", sorted(constructors))
circuit = st.selectbox("Circuit", sorted(circuits))

# %% [markdown]
# ### **Prepare Input Features**

# %%
input_data = {col:0 for col in X.columns}
input_data['race_year'] = race_year
input_data['round'] = round_num
input_data['grid'] = grid
input_data['qualifying_position'] = qual_pos
input_data['driver_points'] = driver_pts
input_data['constructor_points'] = constructor_pts
input_data['avg_lap_time'] = avg_lap_time
input_data['num_of_pitstops'] = num_pits
input_data['sprint_position'] = sprint_pos

# %% [markdown]
# ### **Set one-hot encoded values**

# %%
if f'driverRef_{driver}' in input_data:
    input_data[f'driverRef_{driver}'] = 1
if f'constructorRef_{constructor}' in input_data:
    input_data[f'constructorRef_{constructor}'] = 1
if f'circuitRef_{circuit}' in input_data:
    input_data[f'circuitRef_{circuit}'] = 1

input_df = pd.DataFrame([input_data])

# %% [markdown]
# ### **Predict and Show Results**

# %%
if st.button("Predict Winner"):
  prediction = model.predict(input_df)
  result = "Predicted to Win the Race!" if prediction[0] ==1 else "Not Predicted to Win!"
  st.subheader(result)


