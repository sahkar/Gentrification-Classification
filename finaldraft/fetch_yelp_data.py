#!/usr/bin/env python3
"""
Fetch Yelp business data for ALL zip codes in Bay Area metros
Expands the original 11 zip codes to all 182 zip codes from ZORI data
"""

import requests
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

# Yelp API setup
headers = {
    "Authorization": f"Bearer {os.getenv('YELP_API_KEY')}"
}

def yelp_search(location, categories=None, zip_code=None, limit=50, offset=0):
    """Search Yelp API for businesses"""
    url = "https://api.yelp.com/v3/businesses/search"
    
    params = {
        "location": location,
        "limit": limit,
        "offset": offset
    }
    
    if categories:
        params["categories"] = categories
    
    if zip_code:
        params["location"] = zip_code
    
    response = requests.get(url, headers=headers, params=params)
    
    # Rate limiting - be nice to the API
    time.sleep(0.1)
    
    return response.json()


def collect_zip_businesses(zip_code, categories_list, max_results=200):
    """
    Pull multiple categories for a single ZIP code.
    categories_list: ["cafes", "bubbletea", "matcha", ...]
    max_results: total limit per category
    """
    all_rows = []

    for cat in categories_list:
        for offset in range(0, max_results, 50):  # Yelp limit = 50 per request
            try:
                data = yelp_search(location=zip_code, categories=cat, offset=offset)
                businesses = data.get("businesses", [])
                
                if not businesses:
                    break
                
                # Check for API errors
                if "error" in data:
                    print(f"  ⚠ API error for zip {zip_code}, category {cat}: {data['error'].get('description', 'Unknown error')}")
                    break

                for b in businesses:
                    all_rows.append({
                        "zip": zip_code,
                        "name": b.get("name"),
                        "rating": b.get("rating"),
                        "review_count": b.get("review_count"),
                        "price": b.get("price", None),
                        "lat": b["coordinates"]["latitude"] if "coordinates" in b and "latitude" in b["coordinates"] else None,
                        "lon": b["coordinates"]["longitude"] if "coordinates" in b and "longitude" in b["coordinates"] else None,
                        "categories": ", ".join([c["title"] for c in b.get("categories", [])]),
                        "main_category": cat
                    })
            except Exception as e:
                print(f"  ⚠ Error fetching zip {zip_code}, category {cat}, offset {offset}: {e}")
                break

    return pd.DataFrame(all_rows)


def get_all_zori_zip_codes(data_dir="raw_data"):
    """Extract all zip codes from ZORI data for Bay Area metros"""
    print("="*70)
    print("EXTRACTING ZIP CODES FROM ZORI DATA")
    print("="*70)
    
    data_dir = Path(data_dir)
    zori_path = data_dir / "Zip_zori_uc_sfrcondomfr_sm_month.csv"
    
    if not zori_path.exists():
        raise FileNotFoundError(f"ZORI data not found: {zori_path}")
    
    zori_df = pd.read_csv(zori_path)
    
    # Extract zip codes from RegionName
    zori_df['zip'] = zori_df['RegionName'].astype(str).str.zfill(5)
    
    # Filter to Bay Area metros
    bay_area_metros = [
        "San Francisco-Oakland-Berkeley, CA",
        "San Jose-Sunnyvale-Santa Clara, CA"
    ]
    
    zori_df = zori_df[zori_df['Metro'].isin(bay_area_metros)]
    
    unique_zips = sorted(zori_df['zip'].unique().tolist())
    
    print(f"\n✓ Found {len(unique_zips)} unique zip codes in Bay Area metros")
    print(f"  Metros: {bay_area_metros}")
    print(f"\nFirst 20 zip codes: {unique_zips[:20]}")
    print(f"Last 20 zip codes: {unique_zips[-20:]}")
    
    return unique_zips


def main():
    print("="*70)
    print("YELP DATA EXPANSION - ALL BAY AREA ZIP CODES")
    print("="*70)
    
    # Get all zip codes from ZORI data
    all_zips = get_all_zori_zip_codes(data_dir="raw_data")
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING ZIP CODES: {len(all_zips)} total zip codes")
    print(f"{'='*70}")
    
    # Trendy categories (same as EDA notebook)
    trendy_categories = [
        "matcha", "bubbletea", "cafes", "coffee", 
        "newamerican", "vegan", "poke", 
        "cocktailbars", "desserts", "pilates"
    ]
    
    output_path = Path("raw_data") / "bay_area_yelp_businesses_all_zips.csv"
    
    # Check if we should skip existing file
    if output_path.exists():
        print(f"\n⚠ CSV file '{output_path}' already exists.")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("  Skipping data collection.")
            return
    
    # Check for API key
    if not os.getenv('YELP_API_KEY'):
        raise ValueError("YELP_API_KEY not found in environment. Set it in .env file or export it.")
    
    print(f"\n{'='*70}")
    print("PULLING YELP BUSINESS DATA")
    print(f"{'='*70}")
    print(f"  Zip codes to process: {len(all_zips)}")
    print(f"  Categories per zip: {len(trendy_categories)}")
    print(f"  Estimated API calls: {len(all_zips) * len(trendy_categories) * 4} (max)")
    print(f"  Output file: {output_path}")
    print(f"\n  This may take a while...")
    
    all_dfs = []
    successful_zips = []
    failed_zips = []
    
    for zip_code in tqdm(all_zips, desc="Processing zip codes"):
        try:
            df_zip = collect_zip_businesses(zip_code, trendy_categories)
            
            if len(df_zip) > 0:
                all_dfs.append(df_zip)
                successful_zips.append(zip_code)
            else:
                failed_zips.append(zip_code)
                print(f"  ⚠ No businesses found for zip {zip_code}")
        except Exception as e:
            failed_zips.append(zip_code)
            print(f"  ⚠ Error processing zip {zip_code}: {e}")
    
    if not all_dfs:
        print("\n❌ No data collected!")
        return
    
    # Combine all dataframes
    yelp_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (same business might appear in multiple categories)
    initial_count = len(yelp_df)
    yelp_df = yelp_df.drop_duplicates(subset=['zip', 'name', 'lat', 'lon'], keep='first')
    final_count = len(yelp_df)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yelp_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Total businesses collected: {initial_count:,}")
    print(f"✓ After deduplication: {final_count:,}")
    print(f"✓ Successful zip codes: {len(successful_zips)}/{len(all_zips)}")
    print(f"✓ Failed zip codes: {len(failed_zips)}")
    
    if failed_zips:
        print(f"\n  Failed zip codes: {failed_zips[:10]}{'...' if len(failed_zips) > 10 else ''}")
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"\n  Unique zip codes in data: {yelp_df['zip'].nunique()}")
    print(f"  Zip codes with data: {sorted(yelp_df['zip'].unique())}")
    
    # Show distribution
    print(f"\n{'='*70}")
    print("ZIP CODE DISTRIBUTION")
    print(f"{'='*70}")
    zip_counts = yelp_df.groupby('zip').size().sort_values(ascending=False)
    print(f"\nTop 10 zip codes by business count:")
    for zip_code, count in zip_counts.head(10).items():
        print(f"  {zip_code}: {count:,} businesses")
    
    print(f"\nBottom 10 zip codes by business count:")
    for zip_code, count in zip_counts.tail(10).items():
        print(f"  {zip_code}: {count:,} businesses")


if __name__ == "__main__":
    main()

