"""
Convert ml-ddd_sensitivity_table MovieLens IDs to IMDb IDs
and filter for movies in Reddit dataset
"""
import os
import pandas as pd
import pickle

# File paths
links_csv = "data/movielens/links.csv"
sensitivity_csv = "data/movielens/ml-ddd_sensitivity_table.csv"
reddit_pkl = "data/reddit/entity2id_with_titles.pkl"
output_csv = "data/movielens/ml-ddd_sensitivity_with_imdb.csv"


def main():
    print("="*80)
    print("CONVERT SENSITIVITY TABLE TO IMDB IDS")
    print("="*80)

    # Load links.csv (MovieLens ID -> IMDb ID mapping)
    print("\n1. Loading links.csv...")
    links = pd.read_csv(links_csv)
    print(f"   Loaded {len(links)} MovieLens ID to IMDb ID mappings")

    # Add 'tt' prefix to imdbId to match Reddit format
    links['imdb_id'] = 'tt' + links['imdbId'].astype(str).str.zfill(7)
    print(f"   Sample mapping: movieId={links.iloc[0]['movieId']} -> imdb_id={links.iloc[0]['imdb_id']}")

    # Load ml-ddd_sensitivity_table.csv
    print("\n2. Loading ml-ddd_sensitivity_table.csv...")
    sensitivity = pd.read_csv(sensitivity_csv)
    print(f"   Loaded {len(sensitivity)} movies with sensitivity data")
    print(f"   Columns: {len(sensitivity.columns)} warning labels")

    # Merge with links to add IMDb ID
    print("\n3. Merging with links to add IMDb IDs...")
    sensitivity_with_imdb = sensitivity.merge(
        links[['movieId', 'imdb_id']],
        left_on='work_id',
        right_on='movieId',
        how='left'
    )
    print(f"   After merge: {len(sensitivity_with_imdb)} rows")

    # Check how many have IMDb IDs
    has_imdb = sensitivity_with_imdb['imdb_id'].notna().sum()
    print(f"   Movies with IMDb IDs: {has_imdb}/{len(sensitivity_with_imdb)} ({has_imdb/len(sensitivity_with_imdb)*100:.1f}%)")

    # Load Reddit dataset IMDb IDs
    print("\n4. Loading Reddit dataset...")
    with open(reddit_pkl, 'rb') as f:
        reddit_data = pickle.load(f)
    reddit_imdb_ids = set(reddit_data['imdb_id'].values)
    print(f"   Reddit dataset has {len(reddit_imdb_ids)} unique movies")

    # Filter for movies in Reddit dataset
    print("\n5. Filtering for movies in Reddit dataset...")
    sensitivity_filtered = sensitivity_with_imdb[
        sensitivity_with_imdb['imdb_id'].isin(reddit_imdb_ids)
    ].copy()
    print(f"   Filtered: {len(sensitivity_filtered)} movies (from {len(sensitivity_with_imdb)})")
    print(f"   Coverage: {len(sensitivity_filtered)/len(reddit_imdb_ids)*100:.1f}% of Reddit movies have sensitivity data")

    # Reorder columns: imdb_id, work_id (movieId), then warning columns
    cols = ['imdb_id', 'work_id'] + [col for col in sensitivity_filtered.columns
                                      if col not in ['imdb_id', 'work_id', 'movieId']]
    sensitivity_filtered = sensitivity_filtered[cols]

    # Save to CSV
    print("\n6. Saving results...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sensitivity_filtered.to_csv(output_csv, index=False)
    print(f"   Saved to: {output_csv}")
    print(f"   Shape: {sensitivity_filtered.shape}")
    print(f"   Columns: imdb_id, work_id, + {len(sensitivity_filtered.columns)-2} warning labels")

    # Print sample
    print("\n7. Sample data:")
    print(sensitivity_filtered.head(3).to_string())

    # Statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total sensitivity table entries:      {len(sensitivity):>6}")
    print(f"Entries with IMDb IDs:                {has_imdb:>6} ({has_imdb/len(sensitivity)*100:>5.1f}%)")
    print(f"Entries in Reddit dataset:            {len(sensitivity_filtered):>6} ({len(sensitivity_filtered)/len(sensitivity)*100:>5.1f}%)")
    print(f"Reddit movies with sensitivity data:  {len(sensitivity_filtered):>6} ({len(sensitivity_filtered)/len(reddit_imdb_ids)*100:>5.1f}%)")

    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print(f"\nOutput file: {output_csv}")


if __name__ == '__main__':
    main()
