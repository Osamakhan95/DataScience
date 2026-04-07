# =============================================================================
# Data Science Assignment 2 — Titanic Dataset Analysis
# =============================================================================

# ── 1. Imports ────────────────────────────────────────────────────────────────
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import warnings
warnings.filterwarnings('ignore')

# ── 2. Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

# ── 3. Load Dataset ───────────────────────────────────────────────────────────
df = sns.load_dataset('titanic')

# =============================================================================
# PART 1.1: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# 4. FIRST 8 ROWS & LAST 5 ROWS
print("=" * 70)
print("FIRST 8 ROWS")
print("=" * 70)
print(df.head(8).to_string())

print("\n" + "=" * 70)
print("LAST 5 ROWS")
print("=" * 70)
print(df.tail(5).to_string())

# 5. SHAPE, DATA TYPES, AND STATISTICAL SUMMARIES
print("\n" + "=" * 70)
print("SHAPE OF DATASET")
print("=" * 70)
print(f"Rows: {df.shape[0]}   |   Columns: {df.shape[1]}")

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes.to_string())

numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY — NUMERIC COLUMNS")
print("=" * 70)
print(df[numeric_cols].describe().round(3).to_string())

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY — CATEGORICAL / BOOLEAN COLUMNS")
print("=" * 70)
print(df[categorical_cols].describe().to_string())

# 6. MISSING VALUES — COUNT & PERCENTAGE
print("\n" + "=" * 70)
print("MISSING VALUES PER COLUMN (INITIAL)")
print("=" * 70)

missing_count  = df.isnull().sum()
missing_pct    = (df.isnull().sum() / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Missing Count'     : missing_count,
    'Missing Percentage': missing_pct.astype(str) + ' %'
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df.to_string() if not missing_df.empty else "No missing values found.")

# 7. SURVIVAL RATES
print("\n" + "=" * 70)
print("SURVIVAL RATES")
print("=" * 70)

overall_rate = df['survived'].mean() * 100
overall_df = pd.DataFrame({
    'Total Passengers': [len(df)],
    'Survivors': [df['survived'].sum()],
    'Survival Rate': [f"{overall_rate:.2f} %"]
}, index=['Overall'])
print("\nOverall Survival Rate:")
print(overall_df.to_string())

pclass_df = (
    df.groupby('pclass')['survived']
    .agg(Total_Passengers='count', Survivors='sum')
    .assign(Survival_Rate=lambda x: (x['Survivors'] / x['Total_Passengers'] * 100).round(2).astype(str) + ' %')
)
pclass_df.index.name = 'Passenger Class'
print("\nSurvival Rate by Passenger Class:")
print(pclass_df.to_string())


# =============================================================================
# PART 1.2: DATA CLEANING & FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("DATA CLEANING & FEATURE ENGINEERING")
print("=" * 70)

# 1. Impute missing 'age' values
# Strategy: Group-based median by 'sex' and 'pclass'.
# Reason: Age distributions heavily depend on passenger class (wealthier passengers likely older) 
# and sex. Using the median is robust to outliers, making it a reliable imputation method per subgroup.
print("\n--- 1. Imputing 'age' ---")
df['age'] = df.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
print("Missing 'age' values filled using grouped median by 'sex' and 'pclass'.")

# 2. Drop the 'deck' column
# Reason: The 'deck' column has an extremely high percentage of missing values (~77%). 
# Imputing such a large proportion of data would introduce significant bias and artificial noise.
print("\n--- 2. Dropping 'deck' ---")
if 'deck' in df.columns:
    df.drop('deck', axis=1, inplace=True)
print("'deck' column dropped because >77% of values were missing.")

# 3. Fill missing 'embarked' values
print("\n--- 3. Filling 'embarked' ---")
mode_embarked = df['embarked'].mode()[0]
df['embarked'].fillna(mode_embarked, inplace=True)
# Also apply to 'embark_town' which is redundant but has the same 2 missing values
if 'embark_town' in df.columns:
    mode_embark_town = df['embark_town'].mode()[0]
    df['embark_town'].fillna(mode_embark_town, inplace=True)

print(f"Missing 'embarked' filled with mode: '{mode_embarked}'.")
print(f"Total remaining nulls in 'embarked': {df['embarked'].isnull().sum()}")

# 4. Create new column 'family_size'
print("\n--- 4. Creating 'family_size' ---")
df['family_size'] = df['sibsp'] + df['parch'] + 1
print("Created 'family_size' column (sibsp + parch + 1).")

# 5. Create 'travel_group' column
print("\n--- 5. Creating 'travel_group' ---")
def get_travel_group(size):
    if size == 1:
        return 'Solo'
    elif 2 <= size <= 4:
        return 'Small'
    else:
        return 'Large'

df['travel_group'] = df['family_size'].apply(get_travel_group)
print("\nValue Counts for 'travel_group':")
print(df['travel_group'].value_counts().to_string())

# 6. Create 'age_group' column
print("\n--- 6. Creating 'age_group' ---")
def get_age_group(age):
    if age <= 12:
        return 'Child'
    elif 13 <= age <= 17:
        return 'Teen'
    elif 18 <= age <= 59:
        return 'Adult'
    else:
        return 'Senior'

df['age_group'] = df['age'].apply(get_age_group)
print("\nValue Counts for 'age_group':")
print(df['age_group'].value_counts().to_string())

# 7. Final null-check
print("\n--- 7. Final Null-Check ---")
final_nulls = df.isnull().sum()
if final_nulls.sum() == 0:
    print("SUCCESS: The dataframe is fully clean. No missing values remain.")
else:
    print("WARNING: Some columns still have missing values:")
    print(final_nulls[final_nulls > 0].to_string())

print("\n" + "=" * 70)
print("Script execution completed successfully.")
print("=" * 70)
