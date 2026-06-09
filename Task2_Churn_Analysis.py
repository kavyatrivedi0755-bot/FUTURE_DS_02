# ============================================================
# FUTURE INTERNS — Task 2: Customer Retention & Churn Analysis
# Author : Kavya Trivedi | B.Tech DS & AI | SRMU Lucknow
# Tools  : Python + Power BI
# Repo   : FUTURE_DS_02
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

# ============================================================
# CELL 1 — LOAD DATA
# ============================================================
df = pd.read_csv('telco_churn.csv')
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape          : {df.shape}")
print(f"Columns        : {df.columns.tolist()}")
print(f"\nChurn Distribution:\n{df['Churn'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ============================================================
# CELL 2 — DATA CLEANING
# ============================================================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df['Churn_Flag'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Tenure groups for cohort analysis
df['Tenure_Group'] = pd.cut(df['tenure'],
    bins=[0,12,24,36,48,60,72],
    labels=['0-12 mo','13-24 mo','25-36 mo','37-48 mo','49-60 mo','61-72 mo'])

# Monthly charge segments
df['Charge_Segment'] = pd.cut(df['MonthlyCharges'],
    bins=[0,35,65,90,200],
    labels=['Low (<$35)','Mid ($35-65)','High ($65-90)','Premium (>$90)'])

print("\n✅ Cleaning complete")
print(f"Overall Churn Rate : {df['Churn_Flag'].mean()*100:.2f}%")
print(f"Total Customers    : {len(df):,}")
print(f"Churned Customers  : {df['Churn_Flag'].sum():,}")

# ============================================================
# CELL 3 — CORE CHURN ANALYSIS
# ============================================================
# By Contract
contract_churn = (df.groupby('Contract')['Churn_Flag']
                    .agg(['mean','sum','count'])
                    .reset_index())
contract_churn.columns = ['Contract','Churn_Rate','Churned','Total']
contract_churn['Churn_Rate%'] = (contract_churn['Churn_Rate']*100).round(2)
print("\n📋 Churn by Contract Type:")
print(contract_churn[['Contract','Total','Churned','Churn_Rate%']].to_string(index=False))

# By Payment Method
payment_churn = (df.groupby('PaymentMethod')['Churn_Flag']
                   .agg(['mean','sum','count'])
                   .reset_index())
payment_churn.columns = ['PaymentMethod','Churn_Rate','Churned','Total']
payment_churn['Churn_Rate%'] = (payment_churn['Churn_Rate']*100).round(2)
payment_churn = payment_churn.sort_values('Churn_Rate%', ascending=False)
print("\n💳 Churn by Payment Method:")
print(payment_churn[['PaymentMethod','Total','Churned','Churn_Rate%']].to_string(index=False))

# By Tenure Group
tenure_churn = (df.groupby('Tenure_Group', observed=True)['Churn_Flag']
                  .agg(['mean','sum','count'])
                  .reset_index())
tenure_churn.columns = ['Tenure_Group','Churn_Rate','Churned','Total']
tenure_churn['Churn_Rate%']    = (tenure_churn['Churn_Rate']*100).round(2)
tenure_churn['Retention_Rate%']= (100 - tenure_churn['Churn_Rate%']).round(2)
print("\n⏱ Churn by Tenure Group:")
print(tenure_churn[['Tenure_Group','Total','Churned','Churn_Rate%','Retention_Rate%']].to_string(index=False))

# Monthly Charges — Churned vs Retained
charges = df.groupby('Churn')['MonthlyCharges'].mean().reset_index()
charges.columns = ['Churn','Avg_MonthlyCharge']
charges['Avg_MonthlyCharge'] = charges['Avg_MonthlyCharge'].round(2)
print("\n💰 Avg Monthly Charges — Churned vs Retained:")
print(charges.to_string(index=False))

# ============================================================
# CELL 4 — COHORT RETENTION TABLE
# ============================================================
cohort = (df.groupby('Tenure_Group', observed=True)['Churn_Flag']
            .agg(Total='count', Churned='sum')
            .reset_index())
cohort['Retained']       = cohort['Total'] - cohort['Churned']
cohort['Retention_%']    = (cohort['Retained']/cohort['Total']*100).round(2)
cohort['Churn_%']        = (cohort['Churned']/cohort['Total']*100).round(2)

print("\n📊 COHORT RETENTION TABLE:")
print(cohort.to_string(index=False))
cohort.to_csv('retention_cohort.csv', index=False)
print("\n✅ Saved: retention_cohort.csv")

# ============================================================
# CELL 5 — VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold', y=1.01)

# Chart 1: Churn by Contract
colors1 = ['#EF4444','#F59E0B','#10B981']
bars = axes[0,0].bar(contract_churn['Contract'], contract_churn['Churn_Rate%'], color=colors1)
axes[0,0].set_title('Churn Rate by Contract Type', fontweight='bold')
axes[0,0].set_ylabel('Churn Rate (%)')
for bar,v in zip(bars, contract_churn['Churn_Rate%']):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'{v}%', ha='center', fontweight='bold', fontsize=10)

# Chart 2: Retention Curve by Tenure
axes[0,1].plot(cohort['Tenure_Group'].astype(str), cohort['Retention_%'],
               marker='o', linewidth=2.5, color='#3B82F6', markersize=8)
axes[0,1].fill_between(range(len(cohort)), cohort['Retention_%'],
                        alpha=0.12, color='#3B82F6')
axes[0,1].set_title('Customer Retention Curve by Tenure', fontweight='bold')
axes[0,1].set_ylabel('Retention Rate (%)')
axes[0,1].set_ylim(50, 105)
axes[0,1].set_xticklabels(cohort['Tenure_Group'].astype(str), rotation=30, ha='right')
for i,(x,y) in enumerate(zip(range(len(cohort)), cohort['Retention_%'])):
    axes[0,1].annotate(f'{y}%', (x,y), textcoords='offset points',
                       xytext=(0,8), ha='center', fontsize=9)

# Chart 3: Payment Method Churn
axes[1,0].barh(payment_churn['PaymentMethod'], payment_churn['Churn_Rate%'],
               color='#8B5CF6')
axes[1,0].set_title('Churn Rate by Payment Method', fontweight='bold')
axes[1,0].set_xlabel('Churn Rate (%)')
for i,(v) in enumerate(payment_churn['Churn_Rate%']):
    axes[1,0].text(v+0.3, i, f'{v}%', va='center', fontsize=9)

# Chart 4: Monthly Charges Churned vs Retained
color_churn = ['#10B981','#EF4444']
bars4 = axes[1,1].bar(charges['Churn'], charges['Avg_MonthlyCharge'], color=color_churn, width=0.5)
axes[1,1].set_title('Avg Monthly Charges: Retained vs Churned', fontweight='bold')
axes[1,1].set_ylabel('Avg Monthly Charge ($)')
for bar,v in zip(bars4, charges['Avg_MonthlyCharge']):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'${v}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('chart_churn_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: chart_churn_analysis.png")

# ============================================================
# CELL 6 — ADDITIONAL: CHURN BY INTERNET SERVICE
# ============================================================
internet_churn = (df.groupby('InternetService')['Churn_Flag']
                    .mean()
                    .reset_index())
internet_churn.columns = ['InternetService','Churn_Rate%']
internet_churn['Churn_Rate%'] = (internet_churn['Churn_Rate%']*100).round(2)
print("\n🌐 Churn by Internet Service:")
print(internet_churn.to_string(index=False))

# ============================================================
# CELL 7 — EXPORT FOR POWER BI
# ============================================================
df.to_csv('telco_churn_cleaned.csv', index=False)
contract_churn.to_csv('contract_churn.csv', index=False)
payment_churn.to_csv('payment_churn.csv', index=False)
tenure_churn.to_csv('tenure_churn.csv', index=False)
internet_churn.to_csv('internet_churn.csv', index=False)

print("\n" + "="*50)
print("✅ ALL FILES EXPORTED — Ready for Power BI!")
print("="*50)
print("  telco_churn_cleaned.csv  ← Main dashboard source")
print("  retention_cohort.csv     ← Cohort table")
print("  contract_churn.csv       ← Contract breakdown")
print("  payment_churn.csv        ← Payment method breakdown")
print("  tenure_churn.csv         ← Tenure breakdown")
print("  internet_churn.csv       ← Internet service breakdown")

# ============================================================
# CELL 8 — ACTIONABLE RECOMMENDATIONS
# ============================================================
top_churn_contract = contract_churn.loc[contract_churn['Churn_Rate%'].idxmax(), 'Contract']
top_churn_payment  = payment_churn.iloc[0]['PaymentMethod']
early_churn        = tenure_churn[tenure_churn['Tenure_Group']=='0-12 mo']['Churn_Rate%'].values[0]
late_churn         = tenure_churn[tenure_churn['Tenure_Group']=='61-72 mo']['Churn_Rate%'].values[0]
churned_charge     = charges[charges['Churn']=='Yes']['Avg_MonthlyCharge'].values[0]
retained_charge    = charges[charges['Churn']=='No']['Avg_MonthlyCharge'].values[0]
charge_diff        = churned_charge - retained_charge

print("\n" + "="*50)
print("ACTIONABLE RECOMMENDATIONS")
print("="*50)
print(f"""
1. CONTRACT MIGRATION INCENTIVE
   '{top_churn_contract}' customers have the highest churn.
   → Offer 15% discount to switch to annual plan at the 90-day mark.

2. EARLY TENURE ONBOARDING (0-12 months churn = {early_churn}%)
   → Implement check-ins at Day 7, Day 30, Day 90 to improve retention.

3. PAYMENT METHOD NUDGE
   '{top_churn_payment}' users churn the most.
   → Offer $5/month credit to switch to automatic payment methods.

4. PRICING REVIEW (Churned customers pay ${charge_diff:.2f}/mo more)
   → Introduce a mid-tier plan to retain price-sensitive customers.

5. LOYALTY REWARD AT 3-YEAR MARK (churn drops to {late_churn}% after 61+ months)
   → Create a loyalty reward at 36 months to reinforce long-term commitment.
""")
