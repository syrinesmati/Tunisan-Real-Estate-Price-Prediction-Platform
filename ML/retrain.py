"""
Model Retraining Script
Checks for new data and retrains models if needed
"""
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from train import RealEstatePipeline


def check_for_new_data(last_train_date: datetime) -> bool:
    """Check if new scraped data is available"""
    scraped_data_path = Path("../back/data/scraped_properties.csv")
    
    if not scraped_data_path.exists():
        return False
    
    # Check file modification time
    file_mtime = datetime.fromtimestamp(scraped_data_path.stat().st_mtime)
    
    return file_mtime > last_train_date


def merge_datasets():
    """Merge Kaggle dataset with scraped data"""
    kaggle_path = Path("data/raw/tunisia_real_estate.csv")
    scraped_path = Path("../back/data/scraped_properties.csv")
    
    dfs = []
    
    if kaggle_path.exists():
        df_kaggle = pd.read_csv(kaggle_path)
        df_kaggle['source'] = 'kaggle'
        dfs.append(df_kaggle)
    
    if scraped_path.exists():
        df_scraped = pd.read_csv(scraped_path)
        df_scraped['source'] = 'scraped'
        dfs.append(df_scraped)
    
    if dfs:
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        df_combined = df_combined.drop_duplicates(
            subset=['governorate', 'city', 'area', 'price'],
            keep='last'
        )
        
        # Save combined dataset
        output_path = Path("data/processed/combined_data.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(output_path, index=False)
        
        print(f"✅ Combined dataset saved: {len(df_combined)} rows")
        return output_path
    
    return None


def retrain_models():
    """Retrain models with new data"""
    print("\n" + "="*60)
    print("🔄 RETRAINING MODELS WITH NEW DATA")
    print("="*60)
    
    # Merge datasets
    combined_data_path = merge_datasets()
    
    if combined_data_path is None:
        print("❌ No data available for retraining")
        return
    
    # Run training pipeline
    pipeline = RealEstatePipeline()
    
    # Train for both transaction types
    pipeline.run_experiment(str(combined_data_path), transaction_type='sale')
    pipeline.run_experiment(str(combined_data_path), transaction_type='rent')
    
    # Update last training timestamp
    with open("data/last_training.txt", "w") as f:
        f.write(datetime.now().isoformat())
    
    print("\n✅ Retraining completed!")


def main():
    """Main retraining logic"""
    last_train_file = Path("data/last_training.txt")
    
    if last_train_file.exists():
        with open(last_train_file, "r") as f:
            last_train_date = datetime.fromisoformat(f.read().strip())
        print(f"Last training: {last_train_date}")
        
        if check_for_new_data(last_train_date):
            print("📊 New data detected!")
            retrain_models()
        else:
            print("ℹ️  No new data. Skipping retraining.")
    else:
        print("🆕 First time training")
        retrain_models()


if __name__ == "__main__":
    main()
