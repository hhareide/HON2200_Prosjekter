import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def preliminary_gilens_analysis(file_path):
    """
    Performs a first-stage exploratory analysis on Martin Gilens' 
    Economic Inequality and Political Representation dataset.
    """
    print("Loading STATA dataset...")
    try:
        # pd.read_stata reads Stata files directly into a pandas DataFrame
        df = pd.read_stata(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    print("\n=== 1. STRUCTURAL & QUALITY OVERVIEW ===")
    print(f"Dataset Shape: {df.shape[0]} rows (proposals), {df.shape[1]} columns")
    
    # Harmonize column names. Gilens' data often uses 'p10'/'p50'/'p90' 
    # or 'pred4_10'/'pred4_50'/'pred4_90' along with 'outcome' or 'change'.
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['outcome', 'change', 'adopt', 'policy_outcome']:
            col_mapping[col] = 'outcome'
        elif '10' in col_lower: col_mapping[col] = 'p10'
        elif '50' in col_lower or 'median' in col_lower: col_mapping[col] = 'p50'
        elif '90' in col_lower or 'affluent' in col_lower: col_mapping[col] = 'p90'
            
    if col_mapping:
        df = df.rename(columns=col_mapping)
        print("Mapped columns for analytical consistency:", col_mapping)
        
    required_cols = ['p10', 'p50', 'p90', 'outcome']
    missing_reqs = [c for c in required_cols if c not in df.columns]
    if missing_reqs:
        print(f"CRITICAL WARNING: Missing baseline columns: {missing_reqs}")
        print("Available columns in your file are:", list(df.columns))
        return df

    # Drop missing values in analytical columns for a clean baseline check
    df_clean = df.dropna(subset=required_cols).copy()
    print(f"Rows available after removing missing values: {len(df_clean)}")

    print("\n=== 2. BASELINE STATUS QUO BIAS ===")
    enactment_rate = df_clean['outcome'].mean()
    print(f"Overall policy enactment rate: {enactment_rate:.2%}")
    print("Insight: American public policy possesses a heavy 'status quo bias'.")
    print("The vast majority of proposed policy changes fail to become law.")

    print("\n=== 3. CONVERGENCE & MULTICOLLINEARITY CHECK ===")
    corr_matrix = df_clean[['p10', 'p50', 'p90']].corr()
    print("Correlation Matrix of Policy Preferences Across Income Groups:")
    print(corr_matrix.round(3))
    print("\nInsight: Note the extremely high correlation (typically r > 0.85).")
    print("Rich, middle-class, and poor citizens agree on most things. This creates")
    print("severe multicollinearity, which will mask true control in standard regressions.")

    print("\n=== 4. ISOLATING PREFERENCE GAPS (THE CONFLICT ZONES) ===")
    df_clean['gap_90_50'] = df_clean['p90'] - df_clean['p50']
    df_clean['abs_gap_90_50'] = df_clean['gap_90_50'].abs()
    
    print(df_clean['abs_gap_90_50'].describe())
    
    # Subset to look only at cases where the affluent and the middle class genuinely disagree
    threshold = 10  # 10 percentage point divergence
    conflict_df = df_clean[df_clean['abs_gap_90_50'] >= threshold]
    print(f"\nNumber of policy proposals where p90 and p50 diverge by >= {threshold}%: {len(conflict_df)}")
    
    if len(conflict_df) > 0:
        # Calculate how often policy aligns with each group when they disagree
        rich_win = ((conflict_df['p90'] >= 50) == (conflict_df['outcome'] == 1)).mean()
        middle_win = ((conflict_df['p50'] >= 50) == (conflict_df['outcome'] == 1)).mean()
        print(f"When preferences diverge by >= {threshold} points:")
        print(f"  - Policy matches the Affluent (p90) preference: {rich_win:.2%}")
        print(f"  - Policy matches the Middle Class (p50) preference: {middle_win:.2%}")

    print("\n=== 5. PRELIMINARY PREDICTIVE MODELING ===")
    # Model A: Bivariate Logit for Middle Class
    print("\n--- Model A: Bivariate Logit (Middle Class Support Only) ---")
    model_a = smf.logit("outcome ~ p50", data=df_clean).fit(disp=0)
    print(model_a.summary().tables[1])
    
    # Model B: Bivariate Logit for Affluent Class
    print("\n--- Model B: Bivariate Logit (Affluent Support Only) ---")
    model_b = smf.logit("outcome ~ p90", data=df_clean).fit(disp=0)
    print(model_b.summary().tables[1])
    
    # Model C: Multivariate Logit (The Gilens-Page Race for Influence)
    print("\n--- Model C: Multivariate Logit (Simultaneous Income Group Influence) ---")
    try:
        model_c = smf.logit("outcome ~ p10 + p50 + p90", data=df_clean).fit(disp=0)
        print(model_c.summary().tables[1])
    except Exception as e:
        print(f"Multivariate model failed: {e}. Check for perfect separation or missing data.")

    # Generate and save diagnostic visualizations
    _generate_eda_plots(df_clean)
    
    return df_clean

def _generate_eda_plots(df):
    plt.figure(figsize=(16, 5))
    
    # Plot 1: Preference Distributions
    plt.subplot(1, 3, 1)
    sns.kdeplot(df['p10'], fill=True, label='Poor (p10)', color='blue', alpha=0.2)
    sns.kdeplot(df['p50'], fill=True, label='Middle Class (p50)', color='green', alpha=0.2)
    sns.kdeplot(df['p90'], fill=True, label='Affluent (p90)', color='red', alpha=0.2)
    plt.title('Distribution of Policy Support Levels')
    plt.xlabel('% Favoring Policy Change')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 2: Correlation Heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(df[['p10', 'p50', 'p90']].corr(), annot=True, cmap='coolwarm', vmin=0.6, vmax=1)
    plt.title('Preference Correlation Heatmap')
    
    # Plot 3: Non-parametric Responsiveness Curves
    plt.subplot(1, 3, 3)
    sns.regplot(x='p50', y='outcome', data=df, scatter=False, label='Middle Class (p50)', color='green', lowess=True)
    sns.regplot(x='p90', y='outcome', data=df, scatter=False, label='Affluent (p90)', color='red', lowess=True)
    plt.title('Policy Responsiveness Curves (Lowess)')
    plt.xlabel('% Support for Policy Change')
    plt.ylabel('Observed Enactment Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gilens_preliminary_eda.png', dpi=300)
    print("\n[Visualizations saved locally as 'gilens_preliminary_eda.png']")

cleaned_df = preliminary_gilens_analysis("DS1_v2.dta")