#!/usr/bin/env python3
"""
Fetch Yelp Business Data
Fetches Yelp business data for all zip codes in Bay Area metros.
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
    params = {"location": location, "limit": limit, "offset": offset}
    
    if categories:
        params["categories"] = categories
    if zip_code:
        params["location"] = zip_code
    
    response = requests.get(url, headers=headers, params=params)
    time.sleep(0.1)
    return response.json()


def collect_zip_businesses(zip_code, categories_list, max_results=200):
    """Collect businesses for a single ZIP code across multiple categories"""
    all_rows = []

    for cat in categories_list:
        for offset in range(0, max_results, 50):
            try:
                data = yelp_search(location=zip_code, categories=cat, offset=offset)
                businesses = data.get("businesses", [])
                
                if not businesses:
                    break
                
                if "error" in data:
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
                break

    return pd.DataFrame(all_rows)


def get_all_zori_zip_codes(data_dir="raw_data"):
    """Extract all zip codes from ZORI data for Bay Area metros"""
    print("Extracting zip codes from ZORI data...")
    
    data_dir = Path(data_dir)
    zori_path = data_dir / "Zip_zori_uc_sfrcondomfr_sm_month.csv"
    
    if not zori_path.exists():
        raise FileNotFoundError(f"ZORI data not found: {zori_path}")
    
    zori_df = pd.read_csv(zori_path)
    zori_df['zip'] = zori_df['RegionName'].astype(str).str.zfill(5)
    
    bay_area_metros = [
        "San Francisco-Oakland-Berkeley, CA",
        "San Jose-Sunnyvale-Santa Clara, CA"
    ]
    
    zori_df = zori_df[zori_df['Metro'].isin(bay_area_metros)]
    unique_zips = sorted(zori_df['zip'].unique().tolist())
    
    print(f"✓ Found {len(unique_zips)} zip codes in Bay Area metros")
    
    return unique_zips


def main():
    print("="*70)
    print("YELP DATA COLLECTION")
    print("="*70)
    
    all_zips = get_all_zori_zip_codes(data_dir="raw_data")
    
    trendy_categories = [
        "matcha", "bubbletea", "cafes", "coffee", 
        "newamerican", "vegan", "poke", 
        "cocktailbars", "desserts", "pilates"
    ]
    
    output_path = Path("raw_data") / "bay_area_yelp_businesses_all_zips.csv"
    partial_data_path = output_path.parent / f"{output_path.stem}_partial.csv"
    
    all_dfs = []
    processed_zips = set()

    if partial_data_path.exists():
        try:
            existing_df = pd.read_csv(partial_data_path)
            processed_zips = set(existing_df['zip'].unique())
            all_dfs.append(existing_df)
            print(f"  Resuming: Found {len(processed_zips)} already processed zip codes")
        except Exception as e:
            print(f"  Error loading partial data: {e}. Starting fresh.")
            processed_zips = set()
            all_dfs = []

    if output_path.exists():
        response = input(f"\nFile '{output_path}' exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping data collection.")
            return
    
    if not os.getenv('YELP_API_KEY'):
        raise ValueError("YELP_API_KEY not found in environment. Set it in .env file.")
    
    zips_to_process = [z for z in all_zips if z not in processed_zips]
    print(f"\nProcessing {len(zips_to_process)} zip codes...")
    
    successful_zips = list(processed_zips)
    failed_zips = []
    
    for zip_code in tqdm(zips_to_process, desc="Processing zip codes"):
        try:
            df_zip = collect_zip_businesses(zip_code, trendy_categories)
            
            if len(df_zip) > 0:
                all_dfs.append(df_zip)
                successful_zips.append(zip_code)
            else:
                failed_zips.append(zip_code)
        except Exception as e:
            failed_zips.append(zip_code)
        
        if len(successful_zips) % 10 == 0 and len(successful_zips) > len(processed_zips):
            temp_df = pd.concat(all_dfs, ignore_index=True)
            temp_df.to_csv(partial_data_path, index=False)
    
    if not all_dfs:
        print("\nNo data collected!")
        return
    
    yelp_df = pd.concat(all_dfs, ignore_index=True)
    initial_count = len(yelp_df)
    yelp_df = yelp_df.drop_duplicates(subset=['zip', 'name', 'lat', 'lon'], keep='first')
    final_count = len(yelp_df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yelp_df.to_csv(output_path, index=False)
    
    if partial_data_path.exists():
        partial_data_path.unlink()
    
    print(f"\n✓ Data collection complete")
    print(f"  Total businesses: {initial_count:,}")
    print(f"  After deduplication: {final_count:,}")
    print(f"  Successful zip codes: {len(successful_zips)}/{len(all_zips)}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()

