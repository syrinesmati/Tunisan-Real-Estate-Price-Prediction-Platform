"""
Mubawab.tn Scraper
Real estate platform in Tunisia
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from app.models.schemas import Property, TransactionType
from app.scraping.base_scraper import BaseScraper


class MubawabScraper(BaseScraper):
    """Scraper for Mubawab.tn"""
    
    def __init__(self):
        super().__init__("Mubawab.tn")
        self.base_url = "https://www.mubawab.tn"
        
    async def scrape(
        self,
        governorates: Optional[List[str]] = None,
        transaction_type: Optional[TransactionType] = None,
        max_pages: int = 10
    ) -> List[Property]:
        """Scrape Mubawab.tn listings"""
        properties = []
        
        # Build search URL based on transaction type
        if transaction_type == TransactionType.RENT:
            search_url = f"{self.base_url}/fr/louer"
        else:
            search_url = f"{self.base_url}/fr/acheter"
        
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
                listings = soup.find_all('li', class_='listingBox')
                
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
            title = element.find('h2', class_='title').text.strip()
            price_text = element.find('span', class_='price').text.strip()
            
            # Extract price
            price = float(price_text.replace('DT', '').replace('TND', '').replace(',', '').strip())
            
            location = element.find('div', class_='location').text.strip()
            url = element.find('a')['href']
            if not url.startswith('http'):
                url = self.base_url + url
            
            # Parse location
            parts = location.split(',')
            city = parts[0].strip() if parts else "Unknown"
            governorate = parts[1].strip() if len(parts) > 1 else "Unknown"
            
            # Try to extract area
            area_elem = element.find('span', text=lambda t: 'm²' in t if t else False)
            area = None
            if area_elem:
                area_text = area_elem.text.strip()
                area = float(area_text.replace('m²', '').strip())
            
            return Property(
                title=title,
                price=price,
                governorate=governorate,
                city=city,
                property_type="apartment",
                transaction_type=TransactionType.SALE,
                area=area,
                url=url
            )
            
        except Exception as e:
            print(f"Error parsing property: {e}")
            return None
