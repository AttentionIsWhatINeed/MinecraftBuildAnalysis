import sys
import os
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scraper.scraper import Scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Initialize scraper
    scraper = Scraper()
    
    # Run scraping
    results = scraper.scrape()
    
    print("DONE.")

if __name__ == '__main__':
    main()
