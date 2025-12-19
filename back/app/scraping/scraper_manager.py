"""
Web Scraper Manager
Coordinates scraping from multiple Tunisian real estate websites
"""
import asyncio
import pandas as pd
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from app.scraping.tayara_scraper import TayaraScraper
from app.scraping.mubawab_scraper import MubawabScraper
from app.models.schemas import TransactionType


class ScraperManager:
    """Manages multiple scrapers and data collection"""
    
    def __init__(self):
        self.scrapers = {
            'tayara': TayaraScraper(),
            'mubawab': MubawabScraper(),
        }
        self.status = {
            'is_scraping': False,
            'last_scrape': None,
            'total_scraped': 0
        }
        
        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    async def scrape_all(
        self,
        governorates: Optional[List[str]] = None,
        transaction_type: Optional[TransactionType] = None,
        max_pages: int = 10
    ):
        """Scrape all configured websites"""
        self.status['is_scraping'] = True
        all_properties = []
        
        try:
            for scraper_name, scraper in self.scrapers.items():
                print(f"🕷️  Scraping {scraper_name}...")
                
                properties = await scraper.scrape(
                    governorates=governorates,
                    transaction_type=transaction_type,
                    max_pages=max_pages
                )
                
                all_properties.extend(properties)
                print(f"✅ {scraper_name}: {len(properties)} properties")
            
            # Save to CSV
            if all_properties:
                df = pd.DataFrame([p.dict() for p in all_properties])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.data_dir / f"scraped_properties_{timestamp}.csv"
                df.to_csv(filepath, index=False)
                
                # Also save as latest
                latest_path = self.data_dir / "scraped_properties.csv"
                df.to_csv(latest_path, index=False)
                
                print(f"💾 Saved {len(all_properties)} properties to {filepath}")
            
            self.status['last_scrape'] = datetime.now().isoformat()
            self.status['total_scraped'] = len(all_properties)
            
        except Exception as e:
            print(f"❌ Scraping error: {e}")
            raise
        finally:
            self.status['is_scraping'] = False
    
    def get_status(self) -> dict:
        """Get current scraping status"""
        return self.status
    
    def get_data_stats(self) -> dict:
        """Get statistics about scraped data"""
        try:
            latest_path = self.data_dir / "scraped_properties.csv"
            if latest_path.exists():
                df = pd.read_csv(latest_path)
                return {
                    "total_properties": len(df),
                    "by_transaction_type": df.groupby('transaction_type').size().to_dict(),
                    "by_governorate": df.groupby('governorate').size().to_dict(),
                    "last_updated": self.status['last_scrape']
                }
        except Exception as e:
            print(f"Error getting data stats: {e}")
        
        return {"total_properties": 0}
