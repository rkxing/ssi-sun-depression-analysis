import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# depression bar
depression_data = pd.read_csv("depression-rates-by-state-2023.csv")
depression_data = depression_data[["state", "DepressionRatesByStateAge-AdjustedDepressionRate"]]
depression_data.columns = ['State', 'Age-Adjusted Depression Rate']
depression_data_sorted = depression_data.sort_values('Age-Adjusted Depression Rate')
depression_data_sorted.plot.barh(x='State', y='Age-Adjusted Depression Rate',
                          title='Figure A2: Age-Adjusted Depression Rate by State', figsize=(10,9))

# annual sun bar
annual_sun = pd.read_csv("Average Annual Sunshine by State.csv")
annual_sun_sorted = annual_sun.sort_values('% Sun')
annual_sun_sorted.plot.barh(x='State', y='% Sun', title='Figure A1: Average Annual % Sun by State',
                     figsize=(10,9), color='orange')

# winter sun bar
winter_sun = pd.read_csv("Average Winter Sunshine by State.csv")
winter_sun_sorted = winter_sun.sort_values('% Sun Winter')
winter_sun_sorted.plot.barh(x='State', y='% Sun Winter', title='Average % Sun by State in Winter',
                     figsize=(10,9), color='green')

# annual clear days bar
annual_sun_sorted = annual_sun.sort_values('Clear Days')
annual_sun_sorted.plot.barh(x='State', y='Clear Days', title='Annual Clear Days by State',
                     figsize=(10,9), color='purple')

# winter clear days bar
winter_sun_sorted = winter_sun.sort_values('Clear Days')
winter_sun_sorted.plot.barh(x='State', y='Clear Days', title='Clear Days by State in Winter',
                     figsize=(10,9), color='red')

# mental health expenditures per capita bar
SMHA = pd.read_csv("State Mental Health Agency Expenditures per Capita in Every State.csv")
SMHA_sorted = SMHA.sort_values('Mental Health Expenditures per Capita USD')
SMHA_sorted.plot.barh(x='State', y='Mental Health Expenditures per Capita USD',
                      title='Figure B1: Mental Health Expenditures per Capita by State', figsize=(10,9), color='green')


### depression vs annual sun scatter w/ LoBF ###

# drop DE
corrected_depression_data = depression_data.drop([7])
# drop FL
corrected_annual_sun = annual_sun.drop([7])

x, y = corrected_annual_sun['% Sun'], corrected_depression_data['Age-Adjusted Depression Rate']
a, b = np.polyfit(x, y, 1)

dep_and_annual_sun = corrected_depression_data.merge(corrected_annual_sun[['State', '% Sun']])

corr_dep_and_annual_sun = dep_and_annual_sun.corr(numeric_only=True)
r = corr_dep_and_annual_sun.at['Age-Adjusted Depression Rate', '% Sun']
r = '{:0.4f}'.format(r)

ax = dep_and_annual_sun.plot.scatter(x='% Sun', y='Age-Adjusted Depression Rate')
for _, row in dep_and_annual_sun.iterrows():
    ax.annotate(row['State'], (row['% Sun'], row['Age-Adjusted Depression Rate']), xytext=(2,2),
                textcoords='offset points', family='sans-serif', fontsize=4)
plt.title(f'Figure 1: % Sun Annually by State vs. Age-Adjusted Depression Rate. r = {r}', fontsize=10)
plt.plot(x, a * x + b, color='orange')


### depression vs winter sun scatter w/ LoBF ###

# drop DE and WV
corrected_depression_data = depression_data.drop([7, 46])
# drop FL
corrected_winter_sun = winter_sun.drop([7])

x1, y1 = corrected_winter_sun['% Sun Winter'], corrected_depression_data['Age-Adjusted Depression Rate']
a1, b1 = np.polyfit(x1, y1, 1)

dep_and_winter_sun = corrected_depression_data.merge(corrected_winter_sun[['State', '% Sun Winter']])
dep_and_winter_sun.plot.scatter(x='% Sun Winter', y='Age-Adjusted Depression Rate',
                                title='% Sun in Winter by State vs. Age-Adjusted Depression Rate')
plt.plot(x1, a1 * x1 + b1, color='orange')


### depression vs annual sun scatter w/ LoBF minus outliers WV, HI ###

# drop DE, WV, HI
corrected_depression_data = depression_data.drop([7, 9, 46])
# drop FL, WV, HI
corrected_annual_sun = annual_sun.drop([7, 9, 46])

x1, y1 = corrected_annual_sun['% Sun'], corrected_depression_data['Age-Adjusted Depression Rate']
a1, b1 = np.polyfit(x1, y1, 1)

dep_and_annual_sun = corrected_depression_data.merge(corrected_annual_sun[['State', '% Sun']])

corr_dep_and_annual_sun = dep_and_annual_sun.corr(numeric_only=True)
r1 = corr_dep_and_annual_sun.at['Age-Adjusted Depression Rate', '% Sun']
r1 = '{:0.4f}'.format(r1)

ax = dep_and_annual_sun.plot.scatter(x='% Sun', y='Age-Adjusted Depression Rate')
for _, row in dep_and_annual_sun.iterrows():
    ax.annotate(row['State'], (row['% Sun'], row['Age-Adjusted Depression Rate']), xytext=(2,2),
                textcoords='offset points', family='sans-serif', fontsize=4)
plt.title(f'Figure 2: % Sun Annually by State vs. Age-Adjusted Depression Rate. r = {r1}', fontsize=10)
plt.plot(x1, a1 * x1 + b1, color='orange')


### depression vs mental health expenditures per capita scatter w/ LoBF ###

# drop FL
corrected_SMHA = SMHA.drop([8])

x2, y2 = corrected_SMHA['Mental Health Expenditures per Capita USD'], depression_data['Age-Adjusted Depression Rate']
a2, b2 = np.polyfit(x2, y2, 1)

dep_and_SMHA = depression_data.merge(corrected_SMHA[['State', 'Mental Health Expenditures per Capita USD']])

corr_dep_and_SMHA = dep_and_SMHA.corr(numeric_only=True)
r2 = corr_dep_and_SMHA.at['Age-Adjusted Depression Rate', 'Mental Health Expenditures per Capita USD']
r2 = '{:0.4f}'.format(r2)

ax = dep_and_SMHA.plot.scatter(x='Mental Health Expenditures per Capita USD', y='Age-Adjusted Depression Rate')
for _, row in dep_and_SMHA.iterrows():
    ax.annotate(row['State'], (row['Mental Health Expenditures per Capita USD'], row['Age-Adjusted Depression Rate']), xytext=(2,2),
                textcoords='offset points', family='sans-serif', fontsize=4)
plt.title(f'Figure B2: Mental Health Expenditures per Capita by State vs. Age-Adjusted Depression Rate. r = {r2}',
          fontsize=8.5)
plt.plot(x2, a2 * x2 + b2, color='orange')


plt.show()