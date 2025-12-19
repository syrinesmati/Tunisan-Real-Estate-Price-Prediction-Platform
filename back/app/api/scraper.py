"""
Web Scraping API Endpoints
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import ScrapeRequest, ScrapeResponse
from app.scraping.scraper_manager import ScraperManager
from datetime import datetime

router = APIRouter()
scraper_manager = ScraperManager()


@router.post("/scrape", response_model=ScrapeResponse)
async def trigger_scraping(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger web scraping of Tunisian real estate websites
    
    Runs in background to avoid blocking
    """
    try:
        # Add scraping task to background
        background_tasks.add_task(
            scraper_manager.scrape_all,
            governorates=request.governorates,
            transaction_type=request.transaction_type,
            max_pages=request.max_pages
        )
        
        return ScrapeResponse(
            status="scraping_started",
            properties_scraped=0,  # Will be updated after completion
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scraping error: {str(e)}"
        )


@router.get("/status")
async def get_scraping_status():
    """Get status of scraping operations"""
    return scraper_manager.get_status()


@router.get("/data-stats")
async def get_scraped_data_stats():
    """Get statistics about scraped data"""
    try:
        stats = scraper_manager.get_data_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
