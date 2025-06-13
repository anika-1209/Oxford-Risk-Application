import pandas as pd

#getting the personality dataset
personality_df = pd.read_csv("personality_data.csv")

#viewing dataframe to get familiarized with features
personality_df.head()

#retrieving assets dataset after using API key
assets_df = pd.read_csv("assets_data.csv")

#viewing dataframe to get familiarized with features
assets_df.head(10)

#combining dataframes based on common factor "_id"
combined_df = pd.merge(personality_df, assets_df, on="_id")  
#viewing new dataframe
combined_df.head()

#filtering assets based on GBP and creating a new dataframe for ease of analysis
GBP_df = combined_df[combined_df['asset_currency'] == 'GBP']
GBP_df.head()
#Grouping data by id and summing up the total asset values
total_assets = GBP_df.groupby('_id')['asset_value'].sum().reset_index()
total_assets.head(10)

#merging 'total_assets' with risk tolerance score from personality_df after matching values on '_id'
risk_tolerance_df = pd.merge(total_assets, personality_df[['_id', 'risk_tolerance']], on='_id')
risk_tolerance_df.head(10)

#identifying individual with highest total asset value in GBP
top_individual = risk_tolerance_df.loc[risk_tolerance_df['asset_value'].idxmax()]
print(top_individual)

###Exploratory Data Analysis###

#For ease and accuracy of analysis, let's start by converting all asset values to a single currency (GBP)
#Defining conversion rates to GBP
conversion_rates = {
    'GBP': 1.0,
    'USD': 0.75,
    'EUR': 0.84,
    'AUD': 0.48,
    'JPY': 0.0051
}

# Creating a new column with values converted to GBP
combined_df['asset_value_gbp'] = combined_df.apply(
    lambda row: row['asset_value'] * conversion_rates.get(row['asset_currency'], 0),
    axis=1
)
#descriptive statistics of updated combined_df
combined_df.describe()

##Finding Correlation between personality traits and risk tolerance score##
import seaborn as sns
import matplotlib.pyplot as plt

traits = ['confidence', 'risk_tolerance', 'composure', 'impulsivity', 'impact_desire']
correlation_matrix = combined_df[traits].corr()
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Personality Traits and Risk Tolerance')
plt.tight_layout()
plt.show()

##Comparing risk tolerance with asset types##
#Grouping by asset type and calculate mean, median, and count of risk tolerance
asset_risk_stats = combined_df.groupby('asset_allocation')['risk_tolerance'].agg(['mean', 'median', 'count']).reset_index()

#Sorting to inspect high vs. low risk asset types
asset_risk_stats = asset_risk_stats.sort_values(by='mean', ascending=False)

print(asset_risk_stats)

#Visualizing dataframe for intuitive understanding
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=combined_df,
    x='risk_tolerance',
    y='asset_allocation'
)
plt.title('Distribution of Risk Tolerance by Asset Type')
plt.xlabel('Risk Tolerance')
plt.ylabel('Asset Type')
plt.tight_layout()
plt.show()

##Finding behavioural outliers in asset value vs risk tolerance##
#Summing up total assets in GBP per person
total_assets_gbp = combined_df.groupby('_id')['asset_value_gbp'].sum().reset_index(name='total_asset_gbp')
risk_tolerance = combined_df[['_id', 'risk_tolerance']].drop_duplicates()
asset_risk_df = pd.merge(total_assets_gbp, risk_tolerance, on='_id')

asset_risk_df.head(10)

#Creating a scatter plot of risk tolerance vs total asset value in GBP
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=asset_risk_df,
    x="risk_tolerance",
    y="total_asset_gbp",
)

plt.title("Risk Tolerance vs Total Asset Value (GBP)", fontsize=14)
plt.xlabel("Risk Tolerance")
plt.ylabel("Total Asset Value (GBP)")
plt.tight_layout()
plt.show()

##Personality Trait Clustering using KMeans##
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

personality_traits = ['confidence', 'risk_tolerance', 'composure', 'impulsivity', 'impact_desire']
traits_df = combined_df[['_id'] + personality_traits].drop_duplicates()

#Standardizing traits
scaler = StandardScaler()
traits_scaled = scaler.fit_transform(traits_df[personality_traits])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
traits_df['cluster'] = kmeans.fit_predict(traits_scaled)

# Mean personality trait values by cluster
cluster_summary = traits_df.groupby('cluster')[personality_traits].mean()
print(cluster_summary)


#Additional insights: finding total asset value and asset preference per cluster##
#Total asset value:

#Merging total assets with personality + cluster assignments
clustered_df = pd.merge(traits_df, total_assets_gbp, on='_id')

# Calculating average total GBP asset value per cluster
cluster_wealth = clustered_df.groupby('cluster')['total_asset_gbp'].mean().sort_values(ascending=False)
print(cluster_wealth)

sns.barplot(x=cluster_wealth.index, y=cluster_wealth.values)
plt.xlabel('Cluster')
plt.ylabel('Average Total GBP Asset Value')
plt.title('Average GBP Wealth per Personality Cluster')
plt.show()

#Asset preference:
#Adding cluster labels to combined_df
combined_df_clustered = pd.merge(combined_df, traits_df[['_id', 'cluster']], on='_id')

# Counting asset types per cluster
asset_preferences = (
    combined_df_clustered.groupby(['cluster', 'asset_allocation'])
    .size()
    .unstack(fill_value=0)
)
print(asset_preferences)

# Converting to row-wise percentages
asset_preferences_pct = asset_preferences.div(asset_preferences.sum(axis=1), axis=0)
asset_preferences_pct.head()


asset_preferences_pct.plot(
    kind='bar', stacked=True, figsize=(10, 6), colormap='tab20c'
)
plt.ylabel('Proportion of Asset Types')
plt.xlabel('Cluster')
plt.title('Asset Type Preference per Cluster (Normalized)')
plt.legend(title='Asset Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()