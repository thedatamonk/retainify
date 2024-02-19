import pandas as pd

# Load the uploaded CSV file
file_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# Remove the 'Churn' column
df_modified = df.drop('Churn', axis=1)

# Select a subset of the rows, for example, the first 100 rows
df_subset = df_modified.sample(n=10)

# Save the resultant DataFrame to a new CSV file
output_file_path = 'data/test_data.csv'
df_subset.to_csv(output_file_path, index=False)

output_file_path
