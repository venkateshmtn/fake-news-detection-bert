import pandas as pd

# Load data
df_fake = pd.read_csv('data/Fake.csv')
df_real = pd.read_csv('data/True.csv')

# Add labels
df_fake['label'] = 0
df_real['label'] = 1

# Combine
df = pd.concat([df_fake, df_real], ignore_index=True)

# Shuffle (VERY IMPORTANT)
df = df.sample(frac=1).reset_index(drop=True)

# Combine title + text
df['content'] = df['title'] + " " + df['text']

# Keep only needed columns
df = df[['content', 'label']]

# Check
print(df.head())
print("Shape:", df.shape)