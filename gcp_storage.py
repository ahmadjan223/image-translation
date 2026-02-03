import logging
import io
from typing import Optional
from google.cloud import storage
from google.api_core import exceptions
from PIL import Image
import requests

from config import settings


logger = logging.getLogger(__name__)


class GCPCloudStorage:
    """Manages image storage in GCP Cloud Storage"""
    
    def __init__(self):
        self.client = None
        self.bucket_name = settings.GCP_BUCKET_NAME
        self.bucket = None
        
        if self.bucket_name and settings.GCP_PROJECT_ID:
            try:
                # Use Application Default Credentials (set via config.py)
                self.client = storage.Client(project=settings.GCP_PROJECT_ID)
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"Initialized GCP Cloud Storage for bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize GCP Cloud Storage: {e}")
                logger.warning("Continuing without GCP Cloud Storage. Images will not be stored.")
        else:
            logger.warning("GCP_BUCKET_NAME or GCP_PROJECT_ID not set. Images will not be stored in GCP Cloud Storage.")
    
    def download_image(self, image_url: str, referer: Optional[str] = None) -> Optional[bytes]:
        """
        Download image from URL with retry logic
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Image bytes or None if download fails
        """
        max_retries = 3
        timeout = 30  # seconds

        # Use a session to persist headers
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if referer:
            headers["Referer"] = referer

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading image (attempt {attempt}/{max_retries}): {image_url}")
                resp = session.get(image_url, headers=headers, timeout=timeout, allow_redirects=True)
                # Log status for debugging
                logger.debug(f"Download response status: {getattr(resp, 'status_code', None)}")
                resp.raise_for_status()
                # Basic sanity check for content-type
                content_type = resp.headers.get("Content-Type", "")
                if not content_type.startswith("image"):
                    logger.warning(f"Downloaded resource is not an image (Content-Type: {content_type})")
                logger.info(f"Downloaded image from: {image_url} (bytes: {len(resp.content)})")
                return resp.content
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, 'status_code', None) if getattr(e, 'response', None) else None
                logger.error(f"HTTP error (attempt {attempt}/{max_retries}): {e} (status={status})")
                # If 420 or other client errors from CDN, try small backoff and retry with a cache-busting param
                if attempt < max_retries:
                    import time
                    time.sleep(1 + attempt)
                    # Append a cache-busting query param to try to get a fresh resource
                    if "?" in image_url:
                        image_url = f"{image_url}&_={int(time.time())}"
                    else:
                        image_url = f"{image_url}?_={int(time.time())}"
                    continue
            except Exception as e:
                logger.error(f"Error downloading image from {image_url}: {e}")
                if attempt < max_retries:
                    import time
                    time.sleep(1 + attempt)
                    continue

        logger.error(f"Failed to download after {max_retries} attempts: {image_url}")
        return None
    
    def convert_to_webp(self, image_data: bytes) -> Optional[bytes]:
        """
        Convert image to WebP format
        
        Args:
            image_data: Image bytes in any format
            
        Returns:
            WebP image bytes or None if conversion fails
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert RGBA to RGB if necessary (WebP supports both)
            if image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = background
            elif image.mode not in ('RGB', 'RGBA', 'L'):
                image = image.convert('RGB')
            
            # Convert to WebP
            webp_buffer = io.BytesIO()
            image.save(webp_buffer, format='WEBP', quality=80, method=6)
            webp_data = webp_buffer.getvalue()
            
            logger.info(f"Converted image to WebP format, size: {len(webp_data)} bytes")
            return webp_data
        except Exception as e:
            logger.error(f"Error converting image to WebP: {e}")
            return None
    
    def upload_image(
        self,
        image_data: bytes,
        folder_name: str,
        image_name: str,
        content_type: str = "image/webp"
    ) -> Optional[str]:
        """
        Upload image to GCP Cloud Storage
        
        Args:
            image_data: Image bytes to upload
            folder_name: Folder name in bucket (e.g., "product", "icon")
            image_name: Name of the image file (without extension)
            content_type: Content type of the image (default: image/webp)
            
        Returns:
            GCS object path or None if upload fails
        """
        if not self.bucket:
            logger.warning("GCP Cloud Storage not available. Image not uploaded.")
            return None
        
        try:
            # Construct the blob path: public/products/translated/description/{folder_name}/{image_name}.webp
            blob_path = f"public/products/translated/description/{folder_name}/{image_name}.webp"
            blob = self.bucket.blob(blob_path)
            
            # Set metadata
            blob.content_type = content_type
            blob.cache_control = "public, max-age=604800"  # Cache for 7 days
            
            # Upload image
            blob.upload_from_string(image_data, content_type=content_type)
            
            # Make the blob publicly readable (skip if uniform bucket-level access is enabled)
            try:
                blob.make_public()
                logger.info(f"Made blob publicly readable: {blob_path}")
            except Exception as acl_error:
                # Uniform bucket-level access is enabled, skip setting individual ACLs
                logger.info(f"Skipping ACL setting (uniform bucket-level access enabled): {blob_path}")
            
            logger.info(f"Uploaded image to GCS: {blob_path}")
            return blob_path
            
        except Exception as e:
            logger.error(f"Error uploading image to GCS: {e}")
            return None

    def upload_from_bytes(
        self,
        image_data: bytes,
        folder_name: str,
        image_name: str
    ) -> Optional[str]:
        """
        Convert raw image bytes to WebP and upload to GCS. Returns CDN URL.
        """
        try:
            webp_data = self.convert_to_webp(image_data)
            if not webp_data:
                logger.error("Failed to convert provided bytes to WebP")
                return None

            blob_path = self.upload_image(webp_data, folder_name, image_name)
            if not blob_path:
                logger.error("Failed to upload converted bytes to GCS")
                return None

            public_url = self.get_public_url(blob_path, use_cdn=True)
            if not public_url:
                logger.error("Failed to get public URL after uploading bytes")
                return None

            return public_url
        except Exception as e:
            logger.error(f"Error in upload_from_bytes: {e}", exc_info=True)
            return None
    
    def get_public_url(self, blob_path: str, use_cdn: bool = True) -> Optional[str]:
        """
        Get public URL for the image
        
        Args:
            blob_path: Path to the blob in GCS
            use_cdn: Whether to use CDN URL (default: True)
            
        Returns:
            Public URL of the image or None if blob doesn't exist
        """
        if not self.bucket:
            logger.warning("GCP Cloud Storage not available.")
            return None
        
        try:
            blob = self.bucket.blob(blob_path)
            
            # Check if blob exists (skip if uniform bucket-level access is enabled)
            try:
                if not blob.exists():
                    logger.warning(f"Blob does not exist: {blob_path}")
                    return None
            except Exception as e:
                logger.warning(f"Could not check blob existence: {e}")
            
            if use_cdn and settings.GCP_CDN_URL:
                # Use CDN URL
                cdn_url = settings.GCP_CDN_URL.rstrip('/')
                public_url = f"{cdn_url}/{blob_path}"
            else:
                # Use direct GCS URL
                try:
                    blob.make_public()
                except Exception as acl_error:
                    logger.info(f"Skipping make_public (uniform bucket-level access enabled)")
                public_url = blob.public_url
            
            logger.info(f"Generated public URL: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"Error getting public URL: {e}")
            return None
    
    def process_and_upload_image(
        self,
        image_url: str,
        folder_name: str,
        image_name: str
    ) -> Optional[str]:
        """
        Download image, convert to WebP, and upload to GCS
        
        Args:
            image_url: URL of the image to download
            folder_name: Folder name in bucket (e.g., "product", "icon")
            image_name: Name of the image file (without extension)
            
        Returns:
            Public URL of the uploaded image or None if process fails
        """
        try:
            # Download image
            logger.info(f"Downloading image from: {image_url}")
            image_data = self.download_image(image_url)
            if not image_data:
                logger.error(f"❌ Failed to download image from: {image_url}")
                return None
            
            logger.info(f"✅ Downloaded {len(image_data)} bytes")
            
            # Convert to WebP
            logger.info("Converting image to WebP format")
            webp_data = self.convert_to_webp(image_data)
            if not webp_data:
                logger.error("❌ Failed to convert image to WebP")
                return None
            
            logger.info(f"✅ Converted to WebP: {len(webp_data)} bytes")
            
            # Upload to GCS
            logger.info(f"Uploading image to GCS: public/products/translated/description/{folder_name}/{image_name}.webp")
            blob_path = self.upload_image(webp_data, folder_name, image_name)
            if not blob_path:
                logger.error("❌ Failed to upload image to GCS")
                return None
            
            logger.info(f"✅ Uploaded to GCS: {blob_path}")
            
            # Get public URL
            public_url = self.get_public_url(blob_path, use_cdn=True)
            if not public_url:
                logger.error("❌ Failed to get public URL")
                return None
            
            logger.info(f"✅ Successfully processed and uploaded image: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Error processing and uploading image: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        """Check if GCP Cloud Storage is available"""
        return self.bucket is not None


# Global GCP Cloud Storage instance
gcp_storage = GCPCloudStorage()

