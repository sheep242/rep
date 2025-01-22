# Import necessary libraries
import pandas as pd
import scipy.stats as stats


#Load the CSV files into DataFrames
leads_df = pd.read_csv('leads.csv')
calls_df = pd.read_csv('calls.csv')
signups_df = pd.read_csv('signups.csv')

def clean_column_names(df):
    """
    Converts column names of a DataFrame to lowercase and replaces spaces with underscores.
    
    Parameters:
    df (pd.DataFrame): The DataFrame whose column names need to be cleaned.
    
    Returns:
    pd.DataFrame: The DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

#Convert column names into standardised forms
leads_df = clean_column_names(leads_df)
calls_df = clean_column_names(calls_df)
signups_df = clean_column_names(signups_df)


# 1. Which agent made the most calls?


# Analyze which agent made the most calls
agent_calls_count = calls_df['agent'].value_counts()
most_calls_agent = agent_calls_count.idxmax()
most_calls_count = agent_calls_count.max()

print("1. agent who made most calls:")
print(f"agent {most_calls_agent.upper()} made {most_calls_count} calls")



# 2. Average calls for signed-up leads
# Find the signed-up leads' phone_numbers
full_calls_phone_numbers = calls_df.merge(leads_df, left_on='phone_number', right_on='phone_number', how='outer').reset_index(drop=True)

final_df = full_calls_phone_numbers.merge(signups_df, left_on='name', right_on='lead', how='left')

approved_calls = final_df[final_df['approval_decision']=='APPROVED']


# Group by 'lead' to count the number of calls per lead
calls_per_signed_up_lead = approved_calls.groupby('lead').size()

# Convert the result to a DataFrame for clarity
average_calls_signed_up = calls_per_signed_up_lead.mean()

# Display the resulting dataset with call count per lead

print(f"2. Average Calls per Signed-Up lead: {average_calls_signed_up:.3f}")





# 3.

agent_signup_counts = approved_calls.groupby('agent')['phone_number'].nunique()

# Find the agent with most signups
top_signup_agent = agent_signup_counts.idxmax()
top_signup_count = agent_signup_counts.max()

print(f"3a. agent with most signups: {top_signup_agent.upper()}")
print(f"    Number of signups: {top_signup_count}")
print("\n3b. Assumptions:")
print("- A signup is attributed to the agent who made a call to that lead")
print("- Only unique leads count towards signup count")


# 4.
agent_signups_count = approved_calls.groupby('agent').size()
agent_calls_count = final_df.groupby('agent').size()

# Calculate the number of signups per call for each agent
signups_per_call = agent_signups_count / agent_calls_count

# Identify the agent with the most signups per call
most_signups_per_call_agent = signups_per_call.idxmax()
most_signups_per_call_value = signups_per_call.max()

print(f"4a. agent with most signups per call: {most_signups_per_call_agent}")
print(f"    signups per call: {most_signups_per_call_value:.3f}")


# 4b(i): Analytical approach - Compare lead characteristics

# Compare lead characteristics across agents, this is an example for one region
lead_comparison = final_df.groupby('agent').agg({
    'age': 'mean',
    'region': lambda x: x.mode().iloc[0],
    'sector': lambda x: x.mode().iloc[0]
})

print("\n4b(i) Analytical lead Comparison:")
print(lead_comparison)

'''
Analytical Insights:

Identifies top agent by signups per call
Compares lead characteristics, such as in different regions/ages, etc .across agents

Operational Suggestions:

Implement randomized lead assignment
Conduct A/B testing of lead distribution
Create blind review process for lead potential
Rotate leads across agents systematically

'''

#5.  


# Define interested outcomes
interested_outcomes = ['INTERESTED', 'CALL BACK LATER']
interested_calls = calls_df[calls_df['call_outcome'].isin(interested_outcomes)]

# Merge interested calls with leads to get region
interested_regions = interested_calls.merge(leads_df, on='phone_number', how='left')

# Calculate percentage of interested calls by region
region_interest_rates = interested_regions.groupby('region').agg({
    'phone_number': ['count', lambda x: len(x) / len(calls_df) * 100]
})
region_interest_rates.columns = ['total_interested_calls', 'interest_percentage']
region_interest_rates = region_interest_rates.sort_values('interest_percentage', ascending=False)

print("region Interest Analysis:")
print(region_interest_rates)

# Find the most interested region
most_interested_region = region_interest_rates.index[0]
most_interested_percentage = region_interest_rates.loc[most_interested_region, 'interest_percentage']

print(f"\nMost Interested region: {most_interested_region}")
print(f"Interest Percentage: {most_interested_percentage:.3f}%")


# 6
signup_data = signups_df.merge(leads_df, left_on='lead', right_on='name', how='left')

# (a) Calculate approval probability by region
region_approval = signup_data.groupby('region').agg(
    total_signups=('approval_decision', 'count'),
    approved_signups=('approval_decision', lambda x: (x == 'APPROVED').sum())
)
region_approval['approval_probability'] = region_approval['approved_signups'] / region_approval['total_signups']
region_approval = region_approval.sort_values('approval_probability', ascending=False)

# Most likely region to be approved
top_region = region_approval.index[0]
top_approval_prob = region_approval.loc[top_region, 'approval_probability']

print(f"\n(a) Most Likely region: {top_region}")
print(f"    Approval Probability: {top_approval_prob:.3f} or {top_approval_prob*100:.1f}%")

# (b) Statistical Significance - Chi-Square Test
# Contingency table of approvals vs. non-approvals by region
contingency_table = pd.crosstab(signup_data['region'], signup_data['approval_decision'])

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\n(b) Chi-Square Test:")
print(f"    p-value: {p_value:.3f}")
print(f"    Statistically Significant: {p_value < 0.05}")



# 7. 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score




# Prepare training data
calls_with_leads = calls_df.merge(leads_df, on='phone_number', how='left')
calls_with_signups = calls_with_leads.merge(
    signups_df[['lead']], 
    left_on='name', 
    right_on='lead', 
    how='left'
)
calls_with_signups['signup'] = calls_with_signups['lead'].notna().astype(int)

# Select features
features = ['region', 'sector', 'age']
X = calls_with_signups[features]
y = calls_with_signups['signup']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['region', 'sector'])
    ])

# ML Pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
print("Model Performance:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("AUC ROC Score:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# Predict on uncalled leads
called_leads = set(calls_df['phone_number'])
uncalled_leads = leads_df[~leads_df['phone_number'].isin(called_leads)]

# Predict signup probabilities for uncalled leads
uncalled_leads_features = uncalled_leads[features]
signup_probs = clf.predict_proba(uncalled_leads_features)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score


# AUC score for a binary classifier

import numpy as np
# Select top 1000 leads
top_leads_indices = np.argsort(signup_probs)[-1000:]
top_leads = uncalled_leads.iloc[top_leads_indices]

print("\nlead Prioritization:")
print(f"Total Top leads: {len(top_leads)}")
print(f"Average signup Probability: {signup_probs[top_leads_indices].mean():.2%}")
# (c) Assumptions
print("\nKey Methodology Assumptions:")
print("1. Future signup behavior mirrors historical patterns")
print("2. region and sector are primary signup predictors")
print("3. Uncalled leads have similar characteristics to called leads")
print("4. Random agent allocation for calls")


# Feature selection by imporance
import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = clf.named_steps['classifier'].feature_importances_

num_cols = preprocessor.transformers_[0][2]  
cat_cols = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
all_columns = num_cols + list(cat_cols)

# Create a DataFrame to visualize the importance
feature_importance_df = pd.DataFrame({
    'Feature': all_columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importances from Random Forest")
plt.show()

