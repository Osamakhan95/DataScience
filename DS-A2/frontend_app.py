# type: ignore
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Titanic Dataset EDA & Analysis", layout="wide")
st.title("🚢 Titanic Dataset Exploratory Data Analysis")
st.markdown("This dashboard fullfills all the requirements for **Data Science Assignment 2**.")

# Sidebar Navigation
st.sidebar.title("Navigation")
sections = [
    "Part 1: Setup & Data Cleaning",
    "Part 2: Univariate Analysis",
    "Part 3: Bivariate & Multivariate",
    "Part 4 & 5: Storytelling & Reflection"
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
def process_data(df_raw):
    df = df_raw.copy()
    
    # Q2 (a): Impute missing age 
    df['age'] = df.groupby(['sex', 'pclass'])['age'].transform(lambda x: x.fillna(x.median()))
    
    # Q2 (b): Drop deck
    if 'deck' in df.columns:
        df.drop('deck', axis=1, inplace=True)
        
    # Q2 (c): Fill missing embarked
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    if 'embark_town' in df.columns:
        df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
        
    # Q2 (d): family_size and travel_group
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    def get_travel_group(size):
        if size == 1: return 'Solo'
        elif 2 <= size <= 4: return 'Small'
        else: return 'Large'
    df['travel_group'] = df['family_size'].apply(get_travel_group)
    
    # Q2 (e): age_group
    def get_age_group(age):
        if age <= 12: return 'Child'
        elif 13 <= age <= 17: return 'Teen'
        elif 18 <= age <= 59: return 'Adult'
        else: return 'Senior'
    df['age_group'] = df['age'].apply(get_age_group)
    
    # Streamlit Serialization Patch
    for col in df.select_dtypes(['category']).columns:
        df[col] = df[col].astype(str)
        
    return df

df_raw = load_raw_data()
df_clean = process_data(df_raw)

# -----------------------------------------------------------------------------
# Part 1: Dataset Overview & Cleaning
# -----------------------------------------------------------------------------
if choice == "Part 1: Setup & Data Cleaning":
    st.header("Part 1: Exploratory Data Analysis & Cleaning")
    
    st.subheader("Q1: Initial Inspection")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 8 Rows**")
        st.dataframe(df_raw.head(8).astype(str))
    with col2:
        st.write("**Last 5 Rows**")
        st.dataframe(df_raw.tail(5).astype(str))
        
    st.write(f"**Shape**: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    
    st.write("**Data Types & Summaries**")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns
    cat_cols = df_raw.select_dtypes(include=['object', 'category', 'bool']).columns
    col3, col4 = st.columns(2)
    with col3:
        st.write("*Numeric Summary*")
        st.dataframe(df_raw[num_cols].describe().round(3).astype(str))
    with col4:
        st.write("*Categorical Summary*")
        st.dataframe(df_raw[cat_cols].describe().astype(str))
        
    missing_df = pd.DataFrame({
        'Count': df_raw.isnull().sum(),
        'Percentage (%)': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Count'] > 0]
    st.write("**Missing Values**")
    st.dataframe(missing_df.astype(str))
    st.markdown("""
    * **Age** (~20% missing): Likely omitted by passengers or lost in the confusion of the disaster recording.
    * **Deck** (~77% missing): Cabin locations were mostly unknown for 2nd and 3rd class passengers.
    * **Embarked / Embark_town** (0.22% missing): Very isolated recording error for two passengers.
    """)
    
    st.write("**Survival Rates**")
    overall_sr = (df_raw['survived'].mean() * 100)
    pclass_sr = df_raw.groupby('pclass')['survived'].agg(['count', 'sum']).rename(columns={'count': 'Total_Passengers', 'sum': 'Survivors'})
    pclass_sr['Survival_Rate (%)'] = (pclass_sr['Survivors'] / pclass_sr['Total_Passengers'] * 100).round(2)
    
    st.write(f"*Overall Survival*: {overall_sr:.2f}%")
    st.dataframe(pclass_sr.astype(str))
    
    st.subheader("Q2: Data Cleaning & Feature Engineering")
    st.markdown("""
    **(a) Imputing Age:** Used a group-based median grouping by `sex` and `pclass`. This assumes people of the same class and gender have similar age demographics natively, which is robust to outliers, although it could artificially shrink variance.
    
    **(b) Dropping Deck:** Dropped `deck` because ~77% is missing. Imputing this many rows would introduce massive bias and artificial trends.
    
    **(c) Missing Embarked:** Filled the 2 missing rows with the dataset mode (`S` - Southampton).
    """)
    
    st.write("**Q2 (d) & (e): Engineered Value Counts**")
    st.dataframe(df_clean['travel_group'].value_counts().reset_index().rename(columns={'index':'group', 'count':'Count'}).astype(str))
    st.dataframe(df_clean['age_group'].value_counts().reset_index().rename(columns={'index':'group', 'count':'Count'}).astype(str))
    
    st.success(f"Final Null Check: {df_clean.isnull().sum().sum()} missing values remaining! Clean.")


# -----------------------------------------------------------------------------
# Part 2: Univariate Analysis
# -----------------------------------------------------------------------------
elif choice == "Part 2: Univariate Analysis":
    st.header("Part 2: Univariate Analysis & Distributions")
    
    st.subheader("Q3 (a & b): Age Distribution Deep-Dive")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df_clean['age'], bins=int(80/5), ax=axes1[0])
    axes1[0].set_title("Bin Size: 5 (Under-smooths)")
    sns.histplot(df_clean['age'], bins=int(80/15), kde=True, ax=axes1[1])
    axes1[1].set_title("Bin Size: 15 (Best with KDE)")
    sns.histplot(df_clean['age'], bins=int(80/30), ax=axes1[2])
    axes1[2].set_title("Bin Size: 30 (Over-smooths)")
    st.pyplot(fig1)
    st.markdown("""
    **Interpretation**: The bin size of 15 is the "best" size—it balances capturing the main density around young adults without being overly noisy like size 5 or overly generalized like size 30. The KDE curve shows the distribution is somewhat normal but explicitly right-skewed, featuring a large peak between 20-30 years old. This reflects a real-world demographic where a massive chunk of passengers were young emigrating adults searching for new life in America.
    """)
    
    st.subheader("Q3 (c): Age KDE for Survivors vs Non-Survivors")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=df_clean[df_clean['survived']==1], x='age', color='g', label='Survived (1)', fill=True, ax=ax2)
    sns.kdeplot(data=df_clean[df_clean['survived']==0], x='age', color='r', label='Did Not Survive (0)', fill=True, ax=ax2)
    ax2.legend()
    st.pyplot(fig2)
    st.markdown("""
    **Interpretation**: Between age range 0-10, the green survival density spikes above red, signaling prioritized rescuing of children. Conversely, between 20-30 years, there's a huge spike in red (non-survivors), indicating the large loss of young adult lives, many of whom were 3rd class men restricted from lifeboats.
    """)
    
    st.subheader("Q4 (a): Fare Analysis")
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_clean['fare'], bins=30, ax=axes3[0])
    axes3[0].set_title("Original Fare")
    sns.histplot(np.log1p(df_clean['fare']), bins=30, ax=axes3[1], color='orange')
    axes3[1].set_title("Log(1+Fare)")
    st.pyplot(fig3)
    st.markdown("""
    **Interpretation**: Originally, the fare distribution is heavily right-skewed with most values clustered near zero. Applying the `np.log1p` transformation normalizes the distribution by dampening massive extremes, making patterns and variations among lower fares much more statistically informative.
    """)
    
    st.subheader("Q4 (b & c): Fare Box Plot & Outliers")
    outliers = len(df_clean[df_clean['fare'] > 300])
    st.write(f"There are {outliers} extreme fare outliers above 300.")
    st.markdown("These are legitimate historical values representing massive first-class suites boarded by elite business figures (e.g. Cardeza family paying £512).")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_clean, x='pclass', y='fare', ax=ax4, palette='Set2')
    st.pyplot(fig4)
    st.markdown("**Interpretation**: First class contains the greatest internal spread and extreme outliers (>$500), showing highly tiered premium pricing. Third class is clustered almost completely tightly around negligible fares.")
    
    st.subheader("Q5 (a): Count Plots (Pclass, Sex, Embarked)")
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=df_clean, x='pclass', ax=axes5[0])
    sns.countplot(data=df_clean, x='sex', ax=axes5[1])
    sns.countplot(data=df_clean, x='embarked', ax=axes5[2])
    st.pyplot(fig5)
    st.markdown("""
    * **Pclass**: Overwhelmingly, third-class passengers dominated the ship.
    * **Sex**: The vessel skewed heavily male.
    * **Embarked**: Southampton was the overwhelmingly dominant port of departure.
    """)
    
    st.subheader("Q5 (b): Travel Group Distribution")
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_clean, x='travel_group', order=['Solo', 'Small', 'Large'], ax=ax6)
    st.pyplot(fig6)
    st.markdown("**Interpretation**: Traveling solo was the absolute norm on the Titanic. This implies many passengers were predominantly individuals migrating alone for work rather than vacationing large families.")
    
    st.subheader("Q5 (c): Age Group Bar Chart")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_clean, x='age_group', order=['Child', 'Teen', 'Adult', 'Senior'], ax=ax7)
    st.pyplot(fig7)
    st.markdown("**Interpretation**: 'Adult' is by far the most represented; 'Senior' is the least. Given the 1912 voyage aimed at bringing fresh workforce immigrants to America, this massive peak of prime-aged adults is highly expected.")
    
    st.subheader("Q5 (d): Sex and Survived Count Plots")
    fig8, axes8 = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df_clean, x='sex', ax=axes8[0], palette='Blues')
    sns.countplot(data=df_clean, x='survived', ax=axes8[1], palette='Reds')
    st.pyplot(fig8)
    st.markdown("**Interpretation**: Count plots alone are highly insufficient; they show many men boarded, and many died, but they do NOT show the *intersection* rate of dying by gender. We need bivariate rates or grouped bar charts to truly see survival correlation.")

# -----------------------------------------------------------------------------
# Part 3: Bivariate & Multivariate Analysis
# -----------------------------------------------------------------------------
elif choice == "Part 3: Bivariate & Multivariate":
    st.header("Part 3: Bivariate & Multivariate Analysis")
    
    st.subheader("Q6 (a & b): Survival Rates by Group")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=df_clean, x='sex', y='survived', ax=axes1[0], errorbar=None)
    sns.barplot(data=df_clean, x='pclass', y='survived', ax=axes1[1], errorbar=None)
    sns.barplot(data=df_clean, x='age_group', y='survived', ax=axes1[2], errorbar=None, order=['Child','Teen','Adult','Senior'])
    axes1[0].set_ylabel("Survival Rate")
    for ax in axes1[1:]: ax.set_ylabel("")
    st.pyplot(fig1)
    st.markdown("""
    **Interpretation**: 
    1. **Sex**: Women vastly outsurvived men (>70% vs <20%). Yes, highly expected given ship protocols.
    2. **Pclass**: First class retained a significantly higher rate than third class, showing financial privilege dictated access to safety.
    3. **Age Group**: Children were the most protected group, validating that young age prioritized loading.
    """)
    
    st.subheader("Q6 (c): Grouped Survival Rate (Sex & Pclass)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_clean, x='pclass', y='survived', hue='sex', errorbar=None, ax=ax2)
    ax2.set_ylabel("Survival Rate")
    st.pyplot(fig2)
    st.markdown("**Interpretation**: The 'women and children first' narrative holds up *within* each class individually, as females vastly survive more than men in the same class. However, systemic class divides persist: a 3rd class woman had roughy the survival chance of a 1st class man.")
    
    st.subheader("Q6 (d): Travel Group Hypothesis")
    st.markdown("""
    **Hypothesis (Pre-plot)**: I hypothesize that 'Small' groups have higher survival rates than 'Solo' because families help each other, but 'Large' groups had worst rates as keeping many members together hindered fast escapes.
    """)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_clean, x='travel_group', y='survived', order=['Solo', 'Small', 'Large'], errorbar=None, ax=ax3)
    st.pyplot(fig3)
    st.markdown("**Post-Plot Observation**: The data perfectly **supports** the hypothesis! Small groups achieved the highest survival rates (~50%+), while both Large and Solo groups perished much more frequently.")
    
    st.subheader("Q7: Pearson Correlation Heatmap")
    num_df = df_clean.select_dtypes(include=[np.number])
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax4)
    ax4.set_title("Numeric Pearson Correlation Matrix")
    st.pyplot(fig4)
    st.markdown("""
    **Top 3 Strongest Correlations (Excluding Diagonal)**:
    1. **Pclass & Fare (-0.55)**: Strong negative. First class (1) correlates naturally to paying the highest premium fares.
    2. **Pclass & Age (-0.41)**: Moderate negative. Older established individuals had money to afford first/second-class cabins.
    3. **Sibsp & Family_Size (+0.89)**: Positive correlation. Native mathematical definition as one calculates the other.
    
    **Weakly Correlated Surprise**:
    `Fare` and `Family_Size` (+0.22) are surprisingly weak. Intuitively, large families must buy huge multi-person tickets. However, families often travelled in ultra-cheap 3rd class, nullifing the fare growth from sheer numbers.
    **Limitation of Pearson**: It strictly measures *linear* relationships. It completely fails to detect complex non-linear nuances.
    """)
    
    st.subheader("Q8 (a & b): Scatter & Pairplots")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_clean, x='age', y='fare', hue='survived', palette={0:'red', 1:'green'}, alpha=0.5, ax=ax5)
    st.pyplot(fig5)
    st.markdown("**Scatter Interpretation**: Green survivor dots explicitly cluster across the top half (high fare first class). The red perished dots massively cluster at the bottom left-center, mapping the death zone for poor adult males.")
    
    st.write("Generating Pairplot... (Please wait...)")
    fig_pair = sns.pairplot(df_clean[['age', 'fare', 'pclass', 'survived', 'family_size']], hue='survived', palette={0:'red', 1:'green'}, plot_kws={'alpha':0.5})
    st.pyplot(fig_pair)
    st.markdown("**Pairplot Insight**: The most informative panel is the **Pclass vs Fare scatter/KDE intersections**. It reveals simultaneously what a single chart couldn't: the exact distribution boundaries where class separation mathematically restricted lifeboat access.")


# -----------------------------------------------------------------------------
# Part 4 & 5: Storytelling & Reflection
# -----------------------------------------------------------------------------
elif choice == "Part 4 & 5: Storytelling & Reflection":
    st.header("Part 4 & 5: Storytelling, Misleading Visuals & Reflection")

    st.subheader("Q9 (a): Violin Plot - Age by Pclass, Split by Sex")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df_clean, x='pclass', y='age', hue='sex', split=True, ax=ax1)
    st.pyplot(fig1)
    st.markdown("**Interpretation**: Age distributions widen significantly towards the upper classes. Third class age is intensely concentrated around a tight young demographic for both sexes, whereas First Class (Class 1) possesses the **widest age spread**, accommodating highly varied mature demographics.")

    st.subheader("Q9 (b): Boxplot with Overlay Strip Plot (Fare by Embarked)")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_clean, x='embarked', y='fare', ax=ax2, color="lightgray", showfliers=False)
    sns.stripplot(data=df_clean, x='embarked', y='fare', ax=ax2, size=3, alpha=0.5, jitter=True)
    st.pyplot(fig2)
    st.markdown("**Interpretation**: Combining boxplots (which show summary statistics/medians) and stripplots (which show raw density point-by-point) allows viewers to see exactly *where and how many* raw individuals represent the outliers that drive the boxplot metrics up, providing immense granularity.")

    st.subheader("Q10 (a): FacetGrid KDE Plots")
    g = sns.FacetGrid(df_clean, row='sex', col='pclass', height=3, aspect=1.2)
    g.map(sns.kdeplot, 'age', fill=True, color='purple')
    st.pyplot(g)
    st.markdown("**Interpretation**: Third class curves (last column) are extremely tall and narrow (showing concentrated youths). First class curves (first column) are flattened, proving diverse aged wealthy individuals.")

    st.subheader("Q10 (b): Catplot - Survival by Age, Pclass, and Sex")
    catfig = sns.catplot(data=df_clean, x='age_group', y='survived', col='pclass', hue='sex', kind='bar', errorbar=None, order=['Child','Teen','Adult','Senior'])
    st.pyplot(catfig)
    st.markdown("**Insight**: A surprisingly high survival rate exists for Class 3 children historically expected to perish due to locked gate class barriers. A surprisingly *dismal* rate exists for Class 2 Adult Males, who chivalrously let wives aboard while remaining obedient to officers restricting deck space.")

    st.subheader("Q11: Annotated Narrative Chart")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df_clean, x='pclass', y='survived', hue='sex', markers=["o", "s"], linestyles=["-", "--"], ax=ax3)
    ax3.set_title("Deep Divide: The Cross-Impact of Class and Gender on Survival", fontsize=16, fontweight='bold')
    ax3.set_ylabel("Survival Probability")
    ax3.set_xlabel("Passenger Ticket Class")
    ax3.annotate('Near guaranteed survival\nfor high-society women', xy=(0, 0.96), xytext=(0.5, 0.8),
                 arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=11)
    ax3.annotate('Catastrophic death toll\nfor lower-class men', xy=(2, 0.13), xytext=(1.2, 0.3),
                 arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=11)
    st.pyplot(fig3)
    st.markdown("""
    **Narrative Caption**: This chart highlights the severe societal barriers present aboard the Titanic. By plotting passenger class against survival probability while bifurcating by gender, we immediately observe that while "women and children first" was heavily enforced universally, the baseline chance of accessing a lifeboat uniformly dissolved along rigid class lines. The top echelon of women experienced nearly perfect salvation, while the lowest class of men faced almost certain doom. One question a reader might still have is whether physical cabin locations or discriminatory access protocols caused these sheer class cliffs.
    """)

    st.subheader("Q12: Misleading Visualizations")
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Misleading
    agg_mislead = df_clean.groupby('sex')['survived'].mean().reset_index()
    sns.barplot(data=agg_mislead, x='sex', y='survived', ax=axes4[0])
    axes4[0].set_ylim(0.18, 0.75)  # Truncate to make them look comparable heights
    axes4[0].set_title("Misleading Bar Chart (Truncated Y-Axis)")
    
    # Corrected
    sns.barplot(data=df_clean, x='sex', y='survived', errorbar=None, ax=axes4[1])
    axes4[1].set_ylim(0, 1)        # Proper baseline
    axes4[1].set_title("Corrected Bar Chart (Zero Baseline)")
    st.pyplot(fig4)
    st.markdown("""
    *(a)* **Why it's misleading**: The first chart artificially trims the Y-axis (from 0.18 to 0.75) to make the bar for males look falsely disproportionately tall compared to females, implying survival was somewhat neck-and-neck natively. The corrected chart starts accurately at 0 to explicitly show the gaping cavern.
    
    *(b)* Another previously used chart, the **overall un-logged Fare Histogram**, could easily mislead a non-expert. Viewers might think the dataset contains exclusively zero values given the massive vertical spike, not realizing the X-axis is stretched invisibly sideways extending up to £512. Trimming the x-axis limits or highlighting the tail mathematically prevents misreading!
    """)

    st.subheader("Q13: EDA Reflection")
    st.markdown("""
    **(a) 3 Predictive Hypotheses for Further Modeling:**
    1. *Hypothesis 1*: Did passengers sharing identical surnames and identical tickets (families) sink or swim *together* as cohesive clusters? *(Need ticket string text and surname extraction data).*
    2. *Hypothesis 2*: Was survival intrinsically tied to proximity to the upper deck? *(Need extensive data mapping cabin numbers logically to ship schematics).*
    3. *Hypothesis 3*: Did embarked locations determine survival due to socio-economic disparities native to the boarding towns? *(Need historical income averages corresponding to Southampton vs Cherbourg).*

    **(b) Skiena's Reflection on Insight vs Pictures:**
    Reflecting on my work naturally proves Skiena true. The most profoundly insightful chart I developed was the **FacetGrid Point Plot (Narrative Chart)** overlaying Sex, Pclass, and Survival. Instead of producing generic, standard histograms individually that look pretty but mean nothing collectively, this chart integrated three massive multi-dimensional tensions into a single easily digestible visual. It transformed raw points inherently exposing the historical realities of 1912 societal caste discrimination and priority algorithms. That is pure insight.
    """)
