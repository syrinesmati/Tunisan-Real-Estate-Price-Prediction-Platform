"""
Base Scraper Class
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import asyncio
import time
from app.models.schemas import Property, TransactionType
from app.core.config import settings


class BaseScraper(ABC):
    """Base class for all scrapers"""
    
    def __init__(self, name: str):
        self.name = name
        self.delay = settings.SCRAPING_DELAY
        
    @abstractmethod
    async def scrape(
        self,
        governorates: Optional[List[str]] = None,
        transaction_type: Optional[TransactionType] = None,
        max_pages: int = 10
    ) -> List[Property]:
        """Scrape properties from the website"""
        pass
    
    def _sleep(self):
        """Sleep between requests to be respectful"""
        time.sleep(self.delay)
    
    @abstractmethod
    def parse_property(self, element) -> Optional[Property]:
        """Parse a single property from HTML element"""
        pass
