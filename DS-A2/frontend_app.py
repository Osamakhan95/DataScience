import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Titanic Dataset Analysis", layout="wide")
st.title("🚢 Titanic Dataset Analysis Dashboard")
st.markdown("This frontend dashboard implements all parts of the Data Science Assignment 2.")

# Sidebar Navigation
st.sidebar.title("Navigation")
sections = [
    "Part 1.1: Initial Inspection",
    "Part 1.2: Data Cleaning",
    "Part 2: Univariate Analysis",
    "Part 3: Bivariate & Multivariate",
    "Parts 4 & 5: Advanced & Critical"
]
choice = st.sidebar.radio("Go to part:", sections)

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    np.random.seed(42)
    return sns.load_dataset('titanic')

@st.cache_data
def clean_data(df_raw):
    df = df_raw.copy()
    
    # 1. Impute missing 'age' values using group-based median by sex and pclass
    df['age'] = df.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
    
    # 2. Drop the 'deck' column
    if 'deck' in df.columns:
        df.drop('deck', axis=1, inplace=True)
        
    # 3. Fill missing 'embarked' values using mode
    mode_embarked = df['embarked'].mode()[0]
    df['embarked'].fillna(mode_embarked, inplace=True)
    if 'embark_town' in df.columns:
        df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
        
    # 4. Create new column family_size (sibsp + parch + 1)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    
    # 5. Create travel_group column
    def get_travel_group(size):
        if size == 1: return 'Solo'
        elif 2 <= size <= 4: return 'Small'
        else: return 'Large'
    df['travel_group'] = df['family_size'].apply(get_travel_group)
    
    # 6. Create age_group column
    def get_age_group(age):
        if age <= 12: return 'Child'
        elif 13 <= age <= 17: return 'Teen'
        elif 18 <= age <= 59: return 'Adult'
        else: return 'Senior'
    df['age_group'] = df['age'].apply(get_age_group)
    
    # Fix pyarrow serialization issues by converting category types to broad objects or strings
    for col in df.select_dtypes(['category']).columns:
        df[col] = df[col].astype(str)
        
    return df

df_raw = load_raw_data()
df_clean = clean_data(df_raw)

# -----------------------------------------------------------------------------
# Part 1.1: Setup & Initial Inspection
# -----------------------------------------------------------------------------
if choice == "Part 1.1: Initial Inspection":
    st.header("Part 1.1: Setup & Initial Inspection")
    
    st.subheader("1. Setup & Load Dataset")
    st.markdown("Loaded dataset using `sns.load_dataset('titanic')` and set random seed to 42.")
    
    st.subheader("2. First 8 and Last 5 Rows")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 8 Rows**")
        st.dataframe(df_raw.head(8).astype(str))
    with col2:
        st.write("**Last 5 Rows**")
        st.dataframe(df_raw.tail(5).astype(str))
        
    st.subheader("3. Shape & Data Types")
    st.write(f"**Shape**: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    dtypes_df = pd.DataFrame(df_raw.dtypes, columns=['Data Type']).astype(str)
    st.dataframe(dtypes_df.T)
    
    st.subheader("4. Statistical Summaries")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    cat_cols = df_raw.select_dtypes(include=['object', 'category', 'bool']).columns
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Numeric Columns**")
        st.dataframe(df_raw[numeric_cols].describe().round(3).astype(str))
    with col4:
        st.write("**Categorical Columns**")
        st.dataframe(df_raw[cat_cols].describe().astype(str))
        
    st.subheader("5. Missing Values")
    missing_c = df_raw.isnull().sum()
    missing_p = (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
    missing_df = pd.DataFrame({'Count': missing_c, 'Percentage (%)': missing_p})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
    st.dataframe(missing_df.astype(str))
    
    st.subheader("6. Survival Rates")
    overall_rate = df_raw['survived'].mean() * 100
    st.write(f"**Overall Survival Rate**: {overall_rate:.2f}%")
    pclass_rate = df_raw.groupby('pclass')['survived'].agg(
        Total='count', Survivors='sum'
    ).assign(Rate_pct=lambda x: (x['Survivors']/x['Total']*100).round(2))
    st.dataframe(pclass_rate.astype(str))

# -----------------------------------------------------------------------------
# Part 1.2: Data Cleaning & Feature Engineering
# -----------------------------------------------------------------------------
elif choice == "Part 1.2: Data Cleaning":
    st.header("Part 1.2: Data Cleaning & Feature Engineering")
    
    st.markdown("""
    **1. Impute missing 'age' values:** 
    Group-based median by `sex` and `pclass` was used. *Justification*: Age strongly correlates with class and gender. A wealthy older male parameter provides a much more accurate impute than the global mean.
    
    **2. Drop the 'deck' column:** 
    *Justification*: Over 77% of the data in the `deck` column was missing. Imputing this would introduce fabricated data and statistical bias.
    
    **3. Fill missing 'embarked' values:** 
    Filled with the mode. Verified no nulls remain.
    """)
    
    st.markdown("**4 & 5. Created `family_size` and `travel_group`**")
    st.write("Value Counts for Travel Group:")
    travel_counts = pd.DataFrame(df_clean['travel_group'].value_counts()).reset_index()
    travel_counts.columns = ['Travel Group', 'Count']
    st.dataframe(travel_counts.astype(str))
    
    st.markdown("**6. Created `age_group`**")
    st.write("Value Counts for Age Group:")
    age_counts = pd.DataFrame(df_clean['age_group'].value_counts()).reset_index()
    age_counts.columns = ['Age Group', 'Count']
    st.dataframe(age_counts.astype(str))
    
    st.subheader("7. Final Null Check")
    final_nulls = df_clean.isnull().sum().sum()
    if final_nulls == 0:
        st.success("SUCCESS! The dataframe is clean. 0 missing values remain.")
    else:
        st.error(f"Missing values found: {final_nulls}")
        
    st.write("**Cleaned Data Preview**")
    st.dataframe(df_clean.head().astype(str))

# -----------------------------------------------------------------------------
# Part 2: Univariate Analysis
# -----------------------------------------------------------------------------
elif choice == "Part 2: Univariate Analysis":
    st.header("Part 2: Univariate Analysis & Distributions")
    
    st.subheader("1 & 2. Age Histograms & KDE")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df_clean['age'], bins=int(df_clean['age'].max()/5), ax=axes1[0], color='skyblue')
    axes1[0].set_title("Bin Size: 5")
    sns.histplot(df_clean['age'], bins=int(df_clean['age'].max()/15), kde=True, ax=axes1[1], color='salmon')
    axes1[1].set_title("Bin Size: 15 (Best with KDE)")
    sns.histplot(df_clean['age'], bins=int(df_clean['age'].max()/30), ax=axes1[2], color='lightgreen')
    axes1[2].set_title("Bin Size: 30")
    st.pyplot(fig1)
    st.info("**Interpretation**: A bin size of 15 effectively smooths out small fluctuations while capturing the main distribution peaks, notably the large chunk of young adults in their 20s-30s. Adding the KDE curve highlights exactly where the density centers.")

    st.subheader("3. Age KDE (Survivors vs Non-Survivors)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=df_clean, x='age', hue='survived', common_norm=False, fill=True, ax=ax2)
    st.pyplot(fig2)
    st.info("**Interpretation**: Children (age < 10) had higher density among survivors, confirming a priority in rescue. Meanwhile, young adults aged 20-30 show a sharp spike in non-survivors, reflecting the large loss of able-bodied lower-class individuals.")

    st.subheader("4. Fare Distribution (Log Transformation)")
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_clean['fare'], bins=30, ax=axes3[0])
    axes3[0].set_title("Original Fare")
    sns.histplot(np.log1p(df_clean['fare']), bins=30, ax=axes3[1], color='purple')
    axes3[1].set_title("Log(1+Fare)")
    st.pyplot(fig3)
    st.info("**Interpretation**: The original fare is enormously right-skewed; most people paid cheap fares but a few paid >500. The logarithmic transformation un-skews the data and creates a much more symmetric, readable normal distribution ideal for models.")

    st.subheader("5. Fare Box Plot by Pclass")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_clean, x='pclass', y='fare', ax=ax4, palette='Set2')
    st.pyplot(fig4)
    st.info("**Interpretation**: First-class fares range significantly with extreme outliers upwards of 500. Third-class fares are tightly clustered near zero. This highlights massive wealth inequality aboard.")

    st.subheader("6. Count Plots")
    fig5, axes5 = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes5.flatten()
    sns.countplot(data=df_clean, x='pclass', ax=axes_flat[0], palette='Blues')
    sns.countplot(data=df_clean, x='sex', ax=axes_flat[1], palette='Pastel1')
    sns.countplot(data=df_clean, x='embarked', ax=axes_flat[2], palette='Pastel2')
    sns.countplot(data=df_clean, x='travel_group', ax=axes_flat[3], order=['Solo', 'Small', 'Large'])
    sns.countplot(data=df_clean, x='age_group', ax=axes_flat[4], order=['Child', 'Teen', 'Adult', 'Senior'])
    fig5.delaxes(axes_flat[5])
    plt.tight_layout()
    st.pyplot(fig5)
    st.info("**Interpretation**: Third class vastly outnumbers others. Males double the amount of females. Most passengers embarked closely at Southampton ('S'), and overwhelmingly traveled solo and as adults.")

# -----------------------------------------------------------------------------
# Part 3: Bivariate & Multivariate Analysis
# -----------------------------------------------------------------------------
elif choice == "Part 3: Bivariate & Multivariate":
    st.header("Part 3: Bivariate & Multivariate Analysis")
    
    st.subheader("1. Survival Rates (Bar Charts)")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=df_clean, x='sex', y='survived', ax=axes1[0], ci=None)
    sns.barplot(data=df_clean, x='pclass', y='survived', ax=axes1[1], ci=None)
    sns.barplot(data=df_clean, x='age_group', y='survived', ax=axes1[2], ci=None, order=['Child','Teen','Adult','Senior'])
    axes1[0].set_ylabel("Survival Rate")
    for ax in axes1[1:]: ax.set_ylabel("")
    st.pyplot(fig1)
    st.info("**Interpretation**: Women had a massively higher survival rate than men (~74% vs ~18%). Survival strongly inversely correlates with passenger class. Children survived most frequently among age cohorts.")
    
    st.subheader("2. Grouped Survival Rate (Women and Children First)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_clean, x='pclass', y='survived', hue='sex', ci=None, ax=ax2)
    ax2.set_ylabel("Survival Rate")
    st.pyplot(fig2)
    st.info("**Interpretation**: Across all classes, females out-survived males significantly, validating 'Women and children first'. Interestingly, a third-class female had roughly the same survival chance as a first-class male.")
    
    st.subheader("3. Pearson Correlation Heatmap")
    num_df = df_clean.select_dtypes(include=[np.number])
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax3, vmin=-1, vmax=1)
    st.pyplot(fig3)
    st.info("**Interpretation**: Pclass and Fare are strongly negatively correlated (-0.55). Survival positively correlates with fare (+0.26) and negatively with pclass (-0.34), showing money bought lifeboats.")
    
    st.subheader("4. Age vs Fare Scatter colored by Survival")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_clean, x='age', y='fare', hue='survived', alpha=0.6, ax=ax4)
    st.pyplot(fig4)
    st.info("**Interpretation**: Surviving dots (1) cluster along the top (high fares), while perished dots (0) form a dense wall near zero fare for young adults. Fare overwhelmingly dictated outcome more than age alone.")
    
    st.subheader("5. Pairplot")
    st.write("Generating pairplot... (this may take a few seconds)")
    fig_pair = sns.pairplot(df_clean[['age', 'fare', 'pclass', 'family_size', 'survived']], hue='survived', corner=True)
    st.pyplot(fig_pair)
    st.info("**Interpretation**: The pairplot concisely summarizes multivariate distributions. Clear separation exists in fare and pclass between survivors and victims, but age separation is mostly prominent only for children.")

# -----------------------------------------------------------------------------
# Parts 4 & 5: Advanced Visualization & Critical Analysis
# -----------------------------------------------------------------------------
elif choice == "Parts 4 & 5: Advanced & Critical":
    st.header("Parts 4 & 5: Advanced Visualization & Critical Analysis")
    
    st.subheader("1. Violin Plot: Age by Pclass, Split by Sex")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df_clean, x='pclass', y='age', hue='sex', split=True, inner='quart', ax=ax1)
    st.pyplot(fig1)
    st.info("**Interpretation**: First class contains noticeably older passengers symmetrically distributed across sexes. Third class skews heavily toward younger demographics with tightly clustered medians around age 25.")
    
    st.subheader("2. FacetGrid KDE: Age faceted by Sex & Pclass")
    g = sns.FacetGrid(df_clean, row='sex', col='pclass', hue='survived', margin_titles=True, aspect=1.2)
    g.map(sns.kdeplot, 'age', fill=True)
    g.add_legend()
    st.pyplot(g)
    st.info("**Interpretation**: Breaking this down reveals catastrophic survival for third-class adult men (sharp non-survivor spike at age 25). Alternatively, almost all first/second-class females survived regardless of age.")
    
    st.subheader("3. Narrative Chart")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df_clean, x='pclass', y='survived', hue='sex', markers=["o", "x"], linestyles=["-", "--"], ax=ax3)
    ax3.set_title("Survival Disparity: The Impact of Class and Gender", fontsize=16)
    ax3.set_ylabel("Survival Probability")
    ax3.annotate('Near 100% survival for top class females', xy=(0, 0.96), xytext=(0.5, 0.8),
                 arrowprops={"facecolor": "black", "shrink": 0.05})
    ax3.annotate('Dismal survival for 3rd class men', xy=(2, 0.13), xytext=(1.5, 0.3),
                 arrowprops={"facecolor": "red", "shrink": 0.05})
    st.pyplot(fig3)
    st.info("**Interpretation**: The gap between female and male survival remains consistent across classes, but the baseline chance drops identically per class. Societal norms and financial privilege drove survival simultaneously.")
    
    st.subheader("4. Misleading vs Correct Visualizations")
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    # Misleading (Y-axis truncated)
    sns.barplot(data=df_clean, x='pclass', y='survived', ci=None, ax=axes4[0])
    axes4[0].set_ylim(0.2, 0.7) 
    axes4[0].set_title("Misleading (Truncated Y-axis)")
    # Corrected
    sns.barplot(data=df_clean, x='pclass', y='survived', ci=None, ax=axes4[1])
    axes4[1].set_ylim(0, 1)
    axes4[1].set_title("Corrected (Proper Zero Baseline)")
    st.pyplot(fig4)
    st.info("**Interpretation**: The misleading chart truncates the Y-axis from 0.2 to 0.7, drastically exaggerating the visual height disparity between classes. The corrected version starting at 0 accurately depicts the true ratio.")
    
    st.subheader("5. Reflection")
    st.markdown("""
    **Insightful Plot**: The **FacetGrid KDE chart** offered the most profound insight. By separating distributions into a 2x3 grid across class and gender, we can precisely pinpoint the tragedy (3rd class men) and the salvation (1st class women) natively alongside age distributions.
    
    **3 Hypotheses for Further Investigation:**
    1. *Family Sacrifice:* Did large family groups (`Large` travel_group) have lower survival rates because they refused to abandon family members to board lifeboats alone?
    2. *Cabin Location Matrix:* Did passengers with filled `embark_town` or cabin data survive more often simply because their cabins were closer to the deck?
    3. *Crew vs Passenger:* If we map external data of crew members, did able-bodied crew members perish at higher rates than 3rd class adult males due to their ship duties?
    """)
