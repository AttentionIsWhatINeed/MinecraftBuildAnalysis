import json
import logging
import re
import requests
import time
import traceback
from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Scraper:
    """
    The Scraper for GrabCraft Minecraft builds.
    """
    
    def __init__(
        self,
        base_url: str = "https://www.grabcraft.com",
        output_dir: str = "data/raw",
        images_dir: str = "images",
        metadata_dir: str = "metadata",
        metadata_file: str = "builds_metadata.json",
        checkpoint_file: str = "checkpoint.json",
        
        delay: float = 1.0,
        timeout: int = 10,
        resume: bool = True
    ):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.timeout = timeout
        self.resume = resume
        
        # Create output directories
        self.images_dir = self.output_dir / images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.output_dir / metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.metadata_dir / metadata_file
        
        # HTTP headers for polite scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Metadata storage
        self.metadata = []
        
        # Checkpoint/recovery system
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.checkpoint_data = self._load_checkpoint()
        
        logger.info(f"Scraper initialized. Output directory: {self.output_dir}")
        if self.checkpoint_data and self.resume:
            logger.info(f"Checkpoint found. Resume from category {self.checkpoint_data.get('current_category_idx', 0)}")
    
    def _load_checkpoint(self) -> Dict:
        """Load Checkpoint."""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            traceback.print_exc()
        return {}
    
    def _save_checkpoint(self, category_idx: int, build_idx: int, total_categories: int):
        """Save current scraping progress."""
        try:
            checkpoint = {
                'current_category_idx': category_idx,
                'current_build_idx': build_idx,
                'total_categories': total_categories,
                'timestamp': datetime.now().isoformat(),
                'total_builds_processed': len(self.metadata)
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Checkpoint saved: cat={category_idx}, build={build_idx}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            traceback.print_exc()
    
    def _clear_checkpoint(self):
        """To clear current checkpoint."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("Checkpoint cleared successfully.")
        except Exception as e:
            logger.warning(f"Error clearing checkpoint: {e}")
            traceback.print_exc()
            
    def fetch_page(self, url: str) -> str:
        """
        Fetch a webpage with polite delay.
            
        Returns:
            str: HTML content of the page
        """
        try:
            time.sleep(self.delay)  # Polite delay
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching website {url}: {e}")
            traceback.print_exc()
            return None
        
    def _load_metadata(self) -> List[Dict]:
        """Load previously saved metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            traceback.print_exc()
        return []
    
    def _save_metadata(self):
        """Save all collected metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved ({len(self.metadata)} builds)")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            traceback.print_exc()
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about scraped data.
        
        Returns:
            Dict: Various statistics
        """
        if not self.metadata:
            return {'total_builds': 0, 'total_images': 0, 'tags': {}}
        
        all_tags = {}
        total_images = 0
        
        for build in self.metadata:
            total_images += build.get('images_count', 0)
            for tag in build.get('tags', []):
                all_tags[tag] = all_tags.get(tag, 0) + 1
        
        return {
            'total_builds': len(self.metadata),
            'total_images': total_images,
            'average_images_per_build': round(total_images / len(self.metadata), 2) if self.metadata else 0,
            'unique_tags': len(all_tags),
            'tags': all_tags,
            'images_dir': str(self.images_dir),
            'metadata_file': str(self.metadata_file)
        }

    """
    ============================================================================
    Main scraping methods
    ============================================================================
    """
    def get_category_urls(self, max_categories) -> List[str]:
        """
        Get list of build category URLs from GrabCraft.
            
        Returns:
            List[str]: List of category URLs
        """
        category_urls = []
        
        try:
            home_html = self.fetch_page(self.base_url)
            if not home_html:
                return category_urls
            
            soup = BeautifulSoup(home_html, 'html.parser')
                
            category_links = []
            for li in soup.find_all("li", class_=re.compile(r"^cats-")):
                a_tag = li.find("a", href=True)
                if a_tag:
                    category_links.append(a_tag)
                    
            if not category_links:
                raise Exception("No category links found on the homepage. Check if the site structure has changed.")
            
            for link in category_links[:max_categories]:
                href = link.get('href', '')
                full_url = urljoin(self.base_url, href)
                if full_url not in category_urls:
                    category_urls.append(full_url)
                    logger.info(f"Found category: {full_url}")
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            traceback.print_exc()
        
        return category_urls
    
    def extract_builds_from_category(self, category_url: str) -> List[Dict]:
        """
        Extract all builds from a category page.
            
        Returns:
            List[Dict]: List of build information dictionaries
        """
        builds = []
        
        try:
            html = self.fetch_page(category_url)
            if not html:
                return builds
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find build items
            build_items = soup.find_all('div', class_=lambda x: x and 'product-box' in str(x).lower())
            
            if not build_items:
                logger.warning(f"No build items found in {category_url}")
            
            for index, item in enumerate(build_items):
                try:
                    build_info = self._extract_build_info(index, item, category_url)
                    if build_info:
                        # Extract tags from the build's details page
                        if build_info.get('build_url'):
                            try:
                                detail = self._extract_images_and_tags(build_info['build_url'])
                                
                                if not detail.get('image_urls') or not detail.get('tags'):
                                    if not detail.get('image_urls'):
                                        logger.warning(f"Build discard: No images found for build: {build_info.get('build_url')}")
                                    if not detail.get('tags'):
                                        logger.warning(f"Build discard: No tags found for build: {build_info.get('build_url')}")
                                    continue
                                        
                                build_info['tags'] = detail.get('tags')
                                build_info['image_urls'] = detail.get('image_urls')
                            except Exception as e:
                                logger.warning(f"Failed to extract images and tags for build {build_info.get('build_url')}: {e}")
                                continue
                        
                        builds.append(build_info)
                        logger.info(f"""
                                    ------------------------------------------------------------------------------------------
                                    Found build: {build_info.get('title', 'Unknown')}
                                    URL: {build_info.get('build_url')}
                                    Tags: {build_info.get('tags', [])}
                                    Images: {len(build_info.get('image_urls', []))}
                                    ------------------------------------------------------------------------------------------
                                    """)
                except Exception as e:
                    logger.error(f"Error extracting build info for build {index} in category {category_url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error extracting builds from category {category_url}: {e}")
        
        return builds
    
    def _extract_build_info(self, index: int, item, category_url: str) -> Dict:
        """
        Extract build information from a single item element.
            
        Returns:
            Dict: Build information from the category page
        """
        try:
            # Extract title from h3.name or image alt text
            title = None
            name_elem = item.find('h3', class_='name')
            if name_elem:
                a_elem = name_elem.find('a')
                if a_elem:
                    title = a_elem.get_text(strip=True)
                    build_url = urljoin(self.base_url, a_elem.get('href', ''))
            
            if not title:
                title = 'Unknown Build'
                logger.warning(f"No build link found for build {index} in category {category_url}")
            if not build_url:
                logger.warning(f"Build URL not found for build {index} in category {category_url}")
                return None
            
            # Tags will be extracted from the details page, so initialize as empty
            tags = []
            
            return {
                'title': title,
                'build_url': build_url,
                'tags': tags,
                'category_url': category_url,
                'scraped_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error extracting build info: {e}")
            return None
    
    def _extract_images_and_tags(self, build_url: str) -> Dict:
        """
        Extract tags and images from the build's details page.
            
        Returns:
        """
        image_urls = []
        tags = []
        try:
            html = self.fetch_page(build_url)
            if not html:
                return {}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            image_div = soup.find('div', id='main_pics')
            if image_div:
                images = image_div.find_all('img')
                image_urls = [img.get('src', '') for img in images]
            else:
                logger.warning(f"No image div with id 'main_pics' found in build details page: {build_url}")
            
            tag_td = soup.find('td', class_='value tags')
            if tag_td:
                tags_text = tag_td.get_text(strip=True)
                tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
                logger.info(f"Extracted {len(tags)} tags: {tags}")
            else:
                logger.warning(f"No tags td with class 'value tags' found in build details page: {build_url}")
            
        except Exception as e:
            logger.error(f"Error extracting tags: {e}")
            traceback.print_exc()
        
        return {
            'image_urls': image_urls,
            'tags': tags
        }
    
    def download_all_images_for_build(self, build_info: Dict, build_idx: int) -> Dict:
        """
        Download all images for a single build.
            
        Returns:
            Dict: Updated build_info with local image paths
        """
        build_info['local_image_paths'] = []
        build_info['images_count'] = 0
        
        image_urls = build_info.get('image_urls', [])
        
        if not build_info.get('build_url'):
            logger.warning(f"No image URL in Dict for {build_info.get('title', 'Unknown')}")
            return build_info
        
        try:
            # Create build directory
            build_id = f"build_{build_idx:05d}_{build_info.get('title', 'unknown')[:30]}"
            build_id = "".join(c for c in build_id if c.isalnum() or c in ('_', '-'))[:50]
            build_dir = self.images_dir / build_id
            build_dir.mkdir(parents=True, exist_ok=True)
            
            # Download each image
            for img_idx, img_url in enumerate(image_urls):
                try:
                    # Generate filename
                    filename = f"image_{img_idx:03d}.jpg"
                    save_path = build_dir / filename
                    
                    # Skip if already downloaded
                    if save_path.exists():
                        logger.debug(f"Image already exists: {filename}")
                        build_info['local_image_paths'].append(str(save_path))
                        continue
                    
                    # Download image
                    if self.download_image(img_url, save_path):
                        build_info['local_image_paths'].append(str(save_path))
                    
                except Exception as e:
                    logger.error(f"Error downloading image {img_idx}[{img_url}] for build {build_idx}: {e}")
                    traceback.print_exc()
                    continue
            
            build_info['images_count'] = len(build_info['local_image_paths'])
            build_info['build_directory'] = str(build_dir)
            
            if build_info['images_count'] > 0:
                logger.info(f"Successfully downloaded {build_info['images_count']} images for {build_info.get('title', 'Unknown')}")
            else:
                logger.warning(f"No images successfully downloaded for {build_info.get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error downloading images for build {build_info.get('title', 'Unknown')}: {e}")
            traceback.print_exc()
        
        return build_info
    
    def download_image(self, image_url: str, save_path: Path) -> bool:
        """
        Download an image from URL and save locally with validation.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            time.sleep(self.delay)
            response = requests.get(image_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Open image to validate
            img = Image.open(BytesIO(response.content))
            
            # Ensure RGB mode for JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save image
            img.save(save_path, 'JPEG', quality=85)
            logger.info(f"Downloaded image: {save_path.name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            traceback.print_exc()
            return False
    
    def scrape(self, max_categories: int = 100, max_builds_per_category: int = 100, resume: bool = None) -> Dict:
        """
        Main scraping method.
            
        Returns:
            Dict: Summary of scraping results
        """
        if resume is None:
            resume = self.resume
        
        logger.info("Starting scraping process...")
        
        results = {
            'total_builds': 0,
            'successful_builds': 0,
            'failed_builds': 0,
            'total_images_downloaded': 0,
            'average_images_per_build': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Load previous metadata if resuming
        if resume and self.checkpoint_data:
            try:
                existing_metadata = self._load_metadata()
                self.metadata = existing_metadata
                results['total_builds'] = len(self.metadata)
                logger.info(f"Resumed from checkpoint: {len(self.metadata)} builds already scraped")
            except Exception as e:
                logger.warning(f"Could not load previous metadata: {e}")
                
        # Determine starting point if resuming
        start_category_idx = 0
        if resume and self.checkpoint_data:
            start_category_idx = self.checkpoint_data.get('current_category_idx', 0)
            logger.info(f"Resuming from category index {start_category_idx}")
        
        ## Main Scraping Process
        # Get categories
        category_urls = self.get_category_urls(max_categories)
        logger.info(f"Found {len(category_urls)} categories")
        
        if not category_urls:
            logger.error("No categories found on the website. Check if the site structure has changed.")
            return results
        
        # Process each category
        for cat_idx in range(start_category_idx, len(category_urls)):
            category_url = category_urls[cat_idx]
            logger.info(f"Processing category {cat_idx + 1}/{len(category_urls)}: {category_url}")
            
            try:
                builds = self.extract_builds_from_category(category_url)
                builds = builds[:max_builds_per_category]
                
                # Process each build
                for build_idx, build in enumerate(builds):
                    try:
                        global_build_idx = results['total_builds']
                        results['total_builds'] += 1
                        
                        logger.info(f"Processing build {build_idx + 1}/{len(builds)}: {build.get('title', 'Unknown')}")
                        
                        # Download all images for this build
                        build = self.download_all_images_for_build(build, global_build_idx)
                        
                        # Update results
                        if build.get('images_count', 0) > 0:
                            results['successful_builds'] += 1
                            results['total_images_downloaded'] += build['images_count']
                        else:
                            results['failed_builds'] += 1
                        
                        # Store metadata
                        self.metadata.append(build)
                        
                        # Save metadata after each build (incremental save)
                        self._save_metadata()
                        
                        # Save checkpoint after each build
                        self._save_checkpoint(cat_idx, build_idx, len(category_urls))
                        
                    except Exception as e:
                        logger.error(f"Error processing build [{build_idx}] {build.get('title', 'Unknown')}: {e}")
                        results['failed_builds'] += 1
                        self._save_checkpoint(cat_idx, build_idx, len(category_urls))
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing category [{cat_idx}] {category_urls[cat_idx]}: {e}")
                self._save_checkpoint(cat_idx, 0, len(category_urls))
                continue
        
        # Calculate average
        if results['successful_builds'] > 0:
            results['average_images_per_build'] = round(
                results['total_images_downloaded'] / results['successful_builds'], 2
            )
        
        # Clear checkpoint on successful completion
        self._clear_checkpoint()
        
        logger.info(f"Scraping completed. Results: {results}")
        return results