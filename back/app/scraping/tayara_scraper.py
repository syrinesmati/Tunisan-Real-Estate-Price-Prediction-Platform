"""
Tayara.tn Scraper
Popular Tunisian classifieds website
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from app.models.schemas import Property, TransactionType
from app.scraping.base_scraper import BaseScraper


class TayaraScraper(BaseScraper):
    """Scraper for Tayara.tn"""
    
    def __init__(self):
        super().__init__("Tayara.tn")
        self.base_url = "https://www.tayara.tn"
        
    async def scrape(
        self,
        governorates: Optional[List[str]] = None,
        transaction_type: Optional[TransactionType] = None,
        max_pages: int = 10
    ) -> List[Property]:
        """Scrape Tayara.tn listings"""
        properties = []
        
        # Build search URL
        search_url = f"{self.base_url}/immobilier"
        
        # TODO: Add filters for governorates and transaction_type
        # This is a template - actual implementation depends on website structure
        
        try:
            for page in range(1, max_pages + 1):
                print(f"  Scraping {self.name} page {page}...")
                
                response = requests.get(
                    f"{search_url}?page={page}",
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find property listings
                # NOTE: Selectors need to be updated based on actual website structure
                listings = soup.find_all('div', class_='listing-item')
                
                if not listings:
                    break
                
                for listing in listings:
                    property_obj = self.parse_property(listing)
                    if property_obj:
                        properties.append(property_obj)
                
                self._sleep()
                
        except Exception as e:
            print(f"Error scraping {self.name}: {e}")
        
        return properties
    
    def parse_property(self, element) -> Optional[Property]:
        """Parse property from HTML element"""
        try:
            # NOTE: This is a template - selectors need to match actual website
            title = element.find('h3', class_='title').text.strip()
            price_text = element.find('span', class_='price').text.strip()
            
            # Extract price (remove "DT" or "TND")
            price = float(price_text.replace('DT', '').replace('TND', '').replace(',', '').strip())
            
            location = element.find('span', class_='location').text.strip()
            url = self.base_url + element.find('a')['href']
            
            # Parse location (typically "City, Governorate")
            parts = location.split(',')
            city = parts[0].strip() if parts else "Unknown"
            governorate = parts[1].strip() if len(parts) > 1 else "Unknown"
            
            return Property(
                title=title,
                price=price,
                governorate=governorate,
                city=city,
                property_type="apartment",  # Default, parse from title if possible
                transaction_type=TransactionType.SALE,  # Parse from listing
                url=url
            )
            
        except Exception as e:
            print(f"Error parsing property: {e}")
            return None
