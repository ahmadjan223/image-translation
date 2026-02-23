#!/usr/bin/env python3
"""
Local script to translate product descriptions using FastAPI translate-batch endpoint.
PARALLEL VERSION with INCREMENTAL CSV WRITING.

Features:
  - Hardcoded list of offer IDs (edit OFFER_IDS constant)
  - Appends results to CSV immediately after each completion
  - Skips already processed offers (resume-friendly)
  - Safe to stop mid-way - completed offers are saved

Usage:
    python translate_offer.py

Output:
  - CSV file: translated_offers.csv (appended incrementally) if gcp_storage.is_available():
            blob_path = await loop.run_in_executor(
                None,
                gcp_storage.upload_image,
                img_bytes,
                offer_id,
                f"image_{image_index}",
                "image/webp",
                category  # "description" or "mainImages"
            )
            if blob_path:
                public_url = await loop.run_in_executor(
                    None,
                    gcp_storage.get_public_url,
                    blob_path,
                    True  # use_cdn
                )
                # Delete local file after successful upload to save disk space
                try:
                    await loop.run_in_executor(None, os.remove, local_path)
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to delete local file: {e}")
  - Individual HTML files: translated_offers/{offer_id}.html
"""


import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import httpx
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# MongoDB configuration from .env
MONGODB_CONNECTION_STRING = os.environ.get('mongodb_connection_string')
MONGODB_DATABASE_NAME = os.environ.get('mongodb_database_name')
MONGODB_COLLECTION_NAME = os.environ.get('mongodb_collection_name', 'productsV2')

# FastAPI local endpoint configuration
# Multiple instances for load balancing (ports 8080-8089 by default)
NUM_API_INSTANCES = 1
BASE_PORT = 8080
FASTAPI_URLS = [
    f"http://localhost:{BASE_PORT + i}/translate-batch"
    for i in range(NUM_API_INSTANCES)
]

# Output directory and CSV file
OUTPUT_DIR = Path("translated_offers")
CSV_OUTPUT = "translated_offers.csv"
FAILED_CSV_OUTPUT = "failed.csv"

# Input CSV file with offer IDs (should have a column named "offerId")
OFFER_IDS_INPUT_CSV = "offeridsInputBatch.csv"

# CSV lock for thread-safe writing
csv_lock = asyncio.Lock()
failed_csv_lock = asyncio.Lock()

# Concurrent request limit - one request per API instance
# Each instance handles its own GPU operations internally
MAX_CONCURRENT_REQUESTS = 10


def get_already_processed_offers() -> Set[str]:
    """Load offer IDs that have already been successfully processed from CSV."""
    if not Path(CSV_OUTPUT).exists():
        return set()
    
    processed = set()
    try:
        with open(CSV_OUTPUT, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Only count as processed if HTML is not empty
                if row.get('description'):
                    processed.add(row['offerId'])
        logger.info(f"📋 Found {len(processed)} already processed offers in {CSV_OUTPUT}")
    except Exception as e:
        logger.warning(f"⚠️  Could not read existing CSV: {e}")
    
    return processed


EXPECTED_HEADERS = ['offerId', 'description', 'productImages']


def _ensure_csv_headers(filepath: str, headers: List[str]):
    """Write headers to CSV if file is missing, empty, or has wrong/missing headers."""
    needs_headers = False
    existing_rows = []

    if not Path(filepath).exists() or Path(filepath).stat().st_size == 0:
        needs_headers = True
    else:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                first_row = next(reader)
                if first_row != headers:
                    # Wrong headers - prepend correct ones while keeping data
                    needs_headers = True
                    existing_rows = list(reader)
            except StopIteration:
                needs_headers = True

    if needs_headers:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
            writer.writerows(existing_rows)
        logger.info(f"📄 Wrote headers to {filepath}: {headers}")


def initialize_csv_if_needed():
    """Ensure CSVs exist and have correct headers."""
    _ensure_csv_headers(CSV_OUTPUT, EXPECTED_HEADERS)
    _ensure_csv_headers(FAILED_CSV_OUTPUT, ['offerid', 'error'])


def load_offer_ids_from_csv(csv_file: str) -> List[str]:
    """Load offer IDs from a CSV file with 'offerId' column."""
    if not Path(csv_file).exists():
        logger.error(f"❌ Offer IDs CSV file not found: {csv_file}")
        return []
    
    offer_ids = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                offer_id = row.get('offerId') or row.get('offerid')
                if offer_id and offer_id.strip():
                    offer_ids.append(offer_id.strip())
        logger.info(f"📋 Loaded {len(offer_ids)} offer IDs from {csv_file}")
    except Exception as e:
        logger.error(f"❌ Error reading offer IDs CSV: {e}")
    
    return offer_ids


async def append_result_to_csv(
    offer_id: str,
    translated_html: Optional[str],
    error: Optional[str],
    translated_product_images: Optional[Dict] = None
):
    """Append a single result to CSV immediately (thread-safe)."""
    async with csv_lock:
        try:
            with open(CSV_OUTPUT, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                
                product_images_json = json.dumps(translated_product_images) if translated_product_images else ''
                
                if translated_html:
                    # Escape newlines and quotes for CSV
                    html_escaped = translated_html.replace('\n', '\\n').replace('\r', '\\r')
                    writer.writerow([offer_id, html_escaped, product_images_json])  # offerid, translated_html, translated_mainImages
                else:
                    writer.writerow([offer_id, '', product_images_json])  # offerid, translated_html, translated_mainImages
            
            logger.info(f"💾 [{offer_id}] Appended to CSV")
        except Exception as e:
            logger.error(f"❌ [{offer_id}] Failed to write to CSV: {e}")


async def append_failed_to_csv(offer_id: str, error: str):
    """Append a failed result to failed CSV immediately (thread-safe)."""
    async with failed_csv_lock:
        try:
            with open(FAILED_CSV_OUTPUT, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([offer_id, error or 'Unknown error'])
            
            logger.info(f"💾 [{offer_id}] Appended to failed CSV")
        except Exception as e:
            logger.error(f"❌ [{offer_id}] Failed to write to failed CSV: {e}")


def get_offer_data_from_mongodb(offer_id: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Fetch description and productImages from MongoDB for a single offer_id on-demand.
    Returns: (description, productImages_dict_or_None)
    """
    if not MONGODB_CONNECTION_STRING:
        raise ValueError("mongodb_connection_string environment variable not set")
    if not MONGODB_DATABASE_NAME:
        raise ValueError("mongodb_database_name environment variable not set")
    
    mongo_client = None
    try:
        mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_COLLECTION_NAME]
        
        # Fetch single document
        offer_id_int = int(offer_id)
        document = collection.find_one({"offerId": offer_id_int})
        
        if document:
            description = document.get('description', '')
            product_images = document.get('productImages', None)
            
            if description:
                logger.info(f"✅ [{offer_id}] Retrieved description from MongoDB ({len(description)} chars)")
                if product_images:
                    img_count = len(product_images.get('images', []))
                    has_white = bool(product_images.get('whiteImage'))
                    logger.info(f"   [{offer_id}] productImages: {img_count} images, whiteImage={'yes' if has_white else 'no'}")
                return description, product_images
            else:
                logger.warning(f"⚠️  [{offer_id}] Empty description in MongoDB")
                return None, None
        else:
            logger.warning(f"⚠️  [{offer_id}] Not found in MongoDB")
            return None, None
        
    except Exception as e:
        logger.error(f"❌ [{offer_id}] MongoDB error: {e}")
        return None, None
    finally:
        if mongo_client:
            mongo_client.close()


async def call_translate_api_async(
    client: httpx.AsyncClient,
    offer_id: str,
    semaphore: asyncio.Semaphore,
    api_url: str
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Call FastAPI translate-batch endpoint asynchronously.
    Fetches description from MongoDB on-demand.
    Immediately appends result to CSV upon completion.
    Returns: (offer_id, translated_html, error_message)
    """
    async with semaphore:  # Limit concurrent requests
        port = api_url.split(':')[-1].split('/')[0]
        
        # Fetch description and productImages from MongoDB on-demand
        loop = asyncio.get_event_loop()
        description, product_images = await loop.run_in_executor(None, get_offer_data_from_mongodb, offer_id)
        
        if not description:
            error_msg = "No description found in MongoDB"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_failed_to_csv(offer_id, error_msg)
            return (offer_id, None, error_msg)
        
        logger.info(f"🚀 [{offer_id}] Starting translation... (port {port})")
        
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "description": description,
            "offer_id": str(offer_id),
        }
        if product_images:
            data["productImages"] = product_images
        
        try:
            response = await client.post(api_url, headers=headers, json=data, timeout=600.0)
            
            if response.status_code == 200:
                response_data = response.json()
                translated_html = response_data.get('translated_html', '')
                translated_product_images = response_data.get('translated_product_images', None)
                
                if not translated_html:
                    error_msg = "Empty translated_html in response"
                    logger.error(f"❌ [{offer_id}] {error_msg}")
                    await append_failed_to_csv(offer_id, error_msg)
                    return (offer_id, None, error_msg)
                
                logger.info(f"✅ [{offer_id}] Translation successful ({len(translated_html)} chars) [port {port}]")
                if translated_product_images:
                    img_count = len((translated_product_images.get('images') or []))
                    logger.info(f"   [{offer_id}] translated_product_images: {img_count} images, whiteImage={'yes' if translated_product_images.get('whiteImage') else 'no'}")
                
                # Save individual HTML file immediately
                OUTPUT_DIR.mkdir(exist_ok=True)
                output_path = OUTPUT_DIR / f"{offer_id}.html"
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: output_path.write_text(translated_html, encoding='utf-8'))
                logger.info(f"💾 [{offer_id}] Saved: {output_path}")
                
                await append_result_to_csv(offer_id, translated_html, None, translated_product_images)
                return (offer_id, translated_html, None)
            else:
                error_msg = f"API returned status {response.status_code}"
                logger.error(f"❌ [{offer_id}] {error_msg}")
                await append_failed_to_csv(offer_id, error_msg)
                return (offer_id, None, error_msg)
                
        except httpx.TimeoutException:
            error_msg = "Request timeout"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_failed_to_csv(offer_id, error_msg)
            return (offer_id, None, error_msg)
        except httpx.ConnectError:
            error_msg = f"Connection error - is FastAPI running on {api_url}?"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_failed_to_csv(offer_id, error_msg)
            return (offer_id, None, error_msg)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_failed_to_csv(offer_id, error_msg)
            return (offer_id, None, error_msg)


async def process_all_offers_async(offer_ids: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Process all offers with limited concurrency and load balancing across multiple API instances.
    Fetches descriptions from MongoDB on-demand (one at a time as needed).
    Returns: List of (offer_id, translated_html, error_message)
    """
    logger.info(f"\n🚀 Processing {len(offer_ids)} offers with max {MAX_CONCURRENT_REQUESTS} concurrent requests")
    logger.info(f"⚖️  Load balancing across {len(FASTAPI_URLS)} API instances (ports {BASE_PORT}-{BASE_PORT + NUM_API_INSTANCES - 1})")
    logger.info(f"📊 Fetching descriptions from MongoDB on-demand\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with httpx.AsyncClient() as client:
        tasks = []
        
        for idx, offer_id in enumerate(offer_ids):
            # Round-robin load balancing across API instances
            api_url = FASTAPI_URLS[idx % len(FASTAPI_URLS)]
            
            task = call_translate_api_async(client, offer_id, semaphore, api_url)
            tasks.append(task)
        
        # Run all requests with limited concurrency
        # When one completes, the next one starts automatically
        results = await asyncio.gather(*tasks, return_exceptions=False)
    
    return results



def save_individual_html_files(results: List[Tuple[str, Optional[str], Optional[str]]]):
    """Save individual HTML files for successful translations."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    successful = 0
    for offer_id, translated_html, error in results:
        if translated_html:
            output_path = OUTPUT_DIR / f"{offer_id}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_html)
            logger.info(f"💾 Saved: {output_path}")
            successful += 1
    
    if successful > 0:
        logger.info(f"✅ Saved {successful} individual HTML files to {OUTPUT_DIR}/")


def main():
    """Main function."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 PARALLEL OFFER TRANSLATION")
    logger.info(f"{'='*60}\n")
    
    # Initialize CSV if needed
    initialize_csv_if_needed()
    
    # Load offer IDs from CSV file
    offer_ids_list = load_offer_ids_from_csv(OFFER_IDS_INPUT_CSV)
    
    if not offer_ids_list:
        logger.error(f"❌ No offer IDs loaded from {OFFER_IDS_INPUT_CSV}")
        sys.exit(1)
    
    # Check for already processed offers
    already_processed = get_already_processed_offers()
    
    # Filter out already processed offers
    offer_ids = [oid for oid in offer_ids_list if oid not in already_processed]
    
    if not offer_ids:
        logger.info("✅ All offers already processed! Nothing to do.")
        return
    
    logger.info(f"📋 Total offers in list: {len(offer_ids_list)}")
    logger.info(f"✅ Already processed: {len(already_processed)}")
    logger.info(f"🔄 Remaining to process: {len(offer_ids)}\n")
    
    try:
        # Process all offers in parallel (fetch descriptions on-demand, results appended to CSV incrementally)
        results = asyncio.run(process_all_offers_async(offer_ids))
        
        # Step 3: Save individual HTML files
        # save_individual_html_files(results)
        
        # Summary
        successful = sum(1 for _, html, _ in results if html)
        failed = len(results) - successful
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successful: {successful}/{len(results)}")
        logger.info(f"❌ Failed: {failed}/{len(results)}")
        logger.info(f"📄 Success CSV: {CSV_OUTPUT} (appended incrementally)")
        logger.info(f"📄 Failed CSV: {FAILED_CSV_OUTPUT} (appended incrementally)")
        logger.info(f"📁 HTML Files: {OUTPUT_DIR}/")
        logger.info("=" * 60)
        
        if failed > 0:
            failed_ids = [oid for oid, html, _ in results if not html]
            logger.warning(f"Failed offer IDs: {', '.join(failed_ids)}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Process interrupted by user. Partial results saved to CSV.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()