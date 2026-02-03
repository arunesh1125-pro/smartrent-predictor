# Import necessary libraries
import pandas as pd
import numpy as np
import os
os.makedirs("data", exist_ok=True)

# Set random seed for reproducibility
# (same seed = same random numbers every time you run)
np.random.seed(42)

print("Starting apartment data generation...")
print("-" * 50)

# Define Chennai localities with base rent
localities = {
    'Anna Nagar': 18000,      # Premium area
    'T Nagar': 16000,          # Commercial hub
    'Velachery': 14000,        # IT corridor
    'Adyar': 22000,            # Premium coastal
    'Mylapore': 17000,         # Traditional area
    'Porur': 13000,            # Developing area
    'Tambaram': 10000,         # Budget friendly
    'Guindy': 15000,           # Near airport
    'Kodambakkam': 14000,      # Central location
    'Nungambakkam': 20000      # Premium central
}

print(f"Generating data for {len(localities)} localities:")
for loc, base in localities.items():
    print(f"  - {loc}: Base rent ₹{base:,}")
print("-" * 50)

# Number of apartments to generate
n_samples = 200
print(f"\nGenerating {n_samples} apartment listings...")

# Create empty list to store apartment data
data = []

# Generate each apartment
for i in range(n_samples):
    # Randomly select locality
    locality = np.random.choice(list(localities.keys()))
    base_rent = localities[locality]
    
    # Generate random features
    sq_ft = np.random.randint(400, 1801)  # 400 to 1800 sq ft
    bhk = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])  # More 2BHK common
    floor = np.random.randint(0, 16)  # Ground to 15th floor
    furnished = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])  # 0=Unfurn, 1=Semi, 2=Fully
    parking = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])  # 0-2 parking spots
    age_years = np.random.randint(0, 31)  # 0 to 30 years old
    
    # Calculate rent using realistic formula
    rent = base_rent                    # Start with locality base
    rent += sq_ft * 15                  # Add ₹15 per sq ft
    rent += bhk * 3000                  # Add ₹3000 per bedroom
    rent += floor * 200                 # Add ₹200 per floor
    rent += furnished * 2500            # Semi=₹2500, Fully=₹5000 extra
    rent += parking * 1500              # Add ₹1500 per parking
    rent -= age_years * 150             # Subtract ₹150 per year of age
    
    # Add realistic random variation (±₹2000)
    rent += np.random.randint(-2000, 2001)
    
    # Ensure minimum rent of ₹5000
    rent = max(5000, rent)
    
    # Store apartment as dictionary
    apartment = {
        'sq_ft': sq_ft,
        'bhk': bhk,
        'floor': floor,
        'locality': locality,
        'furnished': furnished,
        'parking': parking,
        'age_years': age_years,
        'rent': rent
    }
    
    data.append(apartment)
    
    # Progress indicator every 50 apartments
    if (i + 1) % 50 == 0:
        print(f"  Generated {i + 1}/{n_samples} apartments...")

print(f"✅ All {n_samples} apartments generated!")
print("-" * 50)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(data)

# Save to CSV file
output_path = 'data/apartments.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n✅ Dataset saved to: {output_path}")

# Display statistics
print("\n" + "=" * 50)
print("DATASET STATISTICS")
print("=" * 50)
print(f"Total apartments: {len(df)}")
print(f"\nRent range: ₹{df['rent'].min():,} - ₹{df['rent'].max():,}")
print(f"Average rent: ₹{df['rent'].mean():,.0f}")
print(f"Median rent: ₹{df['rent'].median():,.0f}")

print(f"\nSize range: {df['sq_ft'].min()} - {df['sq_ft'].max()} sq ft")
print(f"Average size: {df['sq_ft'].mean():.0f} sq ft")

print(f"\nBHK distribution:")
print(df['bhk'].value_counts().sort_index())

print(f"\nLocality distribution:")
print(df['locality'].value_counts())

print("\n" + "=" * 50)
print("SAMPLE DATA (First 10 apartments)")
print("=" * 50)
print(df.head(10).to_string(index=False))

print("\n✅ Data generation complete!")
print(f"You can find the data at: {output_path}")