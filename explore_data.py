import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for prettier plots
sns.set_style("whitegrid")

print("Loading dataset...")
df = pd.read_csv('data/apartments.csv')

print("\n" + "="*60)
print("BASIC INFORMATION")
print("="*60)
print(f"Number of apartments: {len(df)}")
print(f"Number of features: {len(df.columns) - 1}")  # Minus rent (target)
print(f"\nColumns: {', '.join(df.columns)}")

print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ No missing values!")
else:
    print("⚠️ Warning: Missing values detected!")

print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("RENT DISTRIBUTION BY BHK")
print("="*60)
print(df.groupby('bhk')['rent'].agg(['mean', 'min', 'max', 'count']))

print("\n" + "="*60)
print("RENT DISTRIBUTION BY LOCALITY (Top 5)")
print("="*60)
top_localities = df.groupby('locality')['rent'].mean().sort_values(ascending=False).head(5)
print(top_localities)

print("\n" + "="*60)
print("CORRELATION WITH RENT")
print("="*60)
# Calculate correlation for numeric columns only
numeric_cols = ['sq_ft', 'bhk', 'floor', 'furnished', 'parking', 'age_years', 'rent']
correlation = df[numeric_cols].corr()['rent'].sort_values(ascending=False)
print(correlation)

print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

# Create figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Apartment Rent Analysis', fontsize=16, fontweight='bold')

# Plot 1: Rent distribution
axes[0, 0].hist(df['rent'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Rent (₹)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Rent Distribution')
axes[0, 0].axvline(df['rent'].mean(), color='red', linestyle='--', label=f'Mean: ₹{df["rent"].mean():,.0f}')
axes[0, 0].legend()

# Plot 2: Rent vs Square Feet
axes[0, 1].scatter(df['sq_ft'], df['rent'], alpha=0.5, color='green')
axes[0, 1].set_xlabel('Square Feet')
axes[0, 1].set_ylabel('Rent (₹)')
axes[0, 1].set_title('Rent vs Size')

# Plot 3: Rent by BHK
df.boxplot(column='rent', by='bhk', ax=axes[1, 0])
axes[1, 0].set_xlabel('BHK')
axes[1, 0].set_ylabel('Rent (₹)')
axes[1, 0].set_title('Rent Distribution by BHK')
plt.sca(axes[1, 0])
plt.xticks([1, 2, 3], ['1 BHK', '2 BHK', '3 BHK'])

# Plot 4: Top 5 localities by average rent
top_5 = df.groupby('locality')['rent'].mean().sort_values(ascending=False).head(5)
axes[1, 1].barh(range(len(top_5)), top_5.values, color='coral')
axes[1, 1].set_yticks(range(len(top_5)))
axes[1, 1].set_yticklabels(top_5.index)
axes[1, 1].set_xlabel('Average Rent (₹)')
axes[1, 1].set_title('Top 5 Localities by Rent')

plt.tight_layout()
plt.savefig('data/data_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations saved to: data/data_analysis.png")

plt.show()

print("\n" + "="*60)
print("EXPLORATION COMPLETE!")
print("="*60)
print("Key Insights:")
print(f"1. Most common: {df['bhk'].mode()[0]} BHK apartments")
print(f"2. Most expensive locality: {df.groupby('locality')['rent'].mean().idxmax()}")
print(f"3. Biggest rent factor: sq_ft (correlation: {correlation['sq_ft']:.3f})")
print(f"4. Oldest building: {df['age_years'].max()} years")
print(f"5. Newest building: {df['age_years'].min()} years")