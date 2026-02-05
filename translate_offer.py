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
  - CSV file: translated_offers.csv (appended incrementally)
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
# Multiple instances for load balancing (ports 8080-8083 by default)
NUM_API_INSTANCES = 4
BASE_PORT = 8080
FASTAPI_URLS = [
    f"http://localhost:{BASE_PORT + i}/translate-batch"
    for i in range(NUM_API_INSTANCES)
]

# Output directory and CSV file
OUTPUT_DIR = Path("translated_offers")
CSV_OUTPUT = "translated_offers.csv"

# Hardcoded list of offer IDs to process
OFFER_IDS = [
    "624730890959",
    "568953817440",
    "621796987802",
    "602766644649",
    "818322307945",
    "732252682536",
    "601747573294",
    # "626589994862",
    # "597042514466",
    # "635994130681",
    # "711710642712",
    # "633573097044",
    # "590409798220",
    # "603450020772",
    # "525006763799",
    # "520829764809",
    # "626622859354",
    # "623409844150",
    # "634123488276",
    # "579325147274",
    # "762709360445",
    # "581477064514",
    # "520431501518",
    # "640911335840",
    # "635094132637",
    # "657563785032",
    # "590228198711",
    # "635181687313",
    # "560116006891",
    # "730244388583",
    # "554992493166",
    # "630198409357",
    # "552155078917",
    # "612899823898",
    # "589587782649",
    # "773013554799",
    # "827887580684",
    # "828240822385",
    # "564151541433",
    # "641430607491",
    # "745554091928",
    # "773124674725",
    # "563864567047",
    # "642071961825",
    # "597339298377",
    # "592211052698",
    # "734140093224",
    # "811517155585",
    # "737582148019",
    # "626199904545",
    # "602485551584",
    # "773053192401",
    # "629609946248",
    # "672160611914",
    # "679126677885",
    # "660206728469",
    # "641303285523",
    # "818819451892",
    # "682741208700",
    # "563377715620",
    # "712258308646",
    # "816086840615",
    # "773348662420",
    # "536577090188",
    # "716252308319",
    # "814597802859",
    # "624595542131",
    # "770053827786",
    # "617806400536",
    # "634542629387",
    # "641971552897",
    # "741803799246",
    # "826569563017",
    # "547896237507",
    # "755271783502",
    # "719950498546",
    # "813922865294",
    # "817763111343",
    # "826249040805",
    # "625921399000",
    # "742841052358",
    # "653479347253",
    # "626239290860",
    # "600116413912",
    # "728995711439",
    # "614254651476",
    # "810369663388",
    # "664330627564",
    # "631169391173",
    # "737831129958",
    # "583174483152",
    # "1086665641",
    # "817675662797",
    # "575114027644",
    # "638554547463",
    # "602634316361",
    # "650748853250",
    # "626583450545",
    # "714638138260",
    # "824023039251",
]

# CSV lock for thread-safe writing
csv_lock = asyncio.Lock()

# Concurrent request limit - optimized for 2x RTX 5060 Ti
# With 8 OCR + 5 inpainting parallel ops, can handle ~35-40 concurrent requests
# Each request processes 5-10 images, so this keeps GPU saturated
MAX_CONCURRENT_REQUESTS = 40


def get_already_processed_offers() -> Set[str]:
    """Load offer IDs that have already been successfully processed from CSV."""
    if not Path(CSV_OUTPUT).exists():
        return set()
    
    processed = set()
    try:
        with open(CSV_OUTPUT, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Only count as processed if HTML is not empty and no error
                if row.get('html') and not row.get('error'):
                    processed.add(row['offerid'])
        logger.info(f"📋 Found {len(processed)} already processed offers in {CSV_OUTPUT}")
    except Exception as e:
        logger.warning(f"⚠️  Could not read existing CSV: {e}")
    
    return processed


def initialize_csv_if_needed():
    """Create CSV with headers if it doesn't exist."""
    if not Path(CSV_OUTPUT).exists():
        with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['offerid', 'html', 'error'])
        logger.info(f"📄 Created new CSV file: {CSV_OUTPUT}")


async def append_result_to_csv(offer_id: str, translated_html: Optional[str], error: Optional[str]):
    """Append a single result to CSV immediately (thread-safe)."""
    async with csv_lock:
        try:
            with open(CSV_OUTPUT, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                
                if translated_html:
                    # Escape newlines and quotes for CSV
                    html_escaped = translated_html.replace('\n', '\\n').replace('\r', '\\r')
                    writer.writerow([offer_id, html_escaped, ''])
                else:
                    writer.writerow([offer_id, '', error or 'Unknown error'])
            
            logger.info(f"💾 [{offer_id}] Appended to CSV")
        except Exception as e:
            logger.error(f"❌ [{offer_id}] Failed to write to CSV: {e}")


def get_descriptions_from_mongodb(offer_ids: List[str]) -> Dict[str, str]:
    """Fetch descriptions from MongoDB for all offer_ids at once."""
    logger.info(f"🔗 Connecting to MongoDB to fetch {len(offer_ids)} descriptions...")
    
    if not MONGODB_CONNECTION_STRING:
        raise ValueError("mongodb_connection_string environment variable not set")
    if not MONGODB_DATABASE_NAME:
        raise ValueError("mongodb_database_name environment variable not set")
    
    mongo_client = None
    try:
        mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_COLLECTION_NAME]
        
        # Convert to integers and fetch all at once
        offer_ids_int = [int(oid) for oid in offer_ids]
        documents = collection.find({"offerId": {"$in": offer_ids_int}})
        
        # Build dictionary: offer_id -> description
        descriptions = {}
        for doc in documents:
            offer_id = str(doc["offerId"])
            description = doc.get('description', '')
            if description:
                descriptions[offer_id] = description
                logger.info(f"✅ Retrieved description for offerId {offer_id} ({len(description)} chars)")
            else:
                logger.warning(f"⚠️  Empty description for offerId {offer_id}")
        
        # Check for missing offer IDs
        missing = set(offer_ids) - set(descriptions.keys())
        if missing:
            logger.warning(f"⚠️  Missing offer IDs: {', '.join(missing)}")
        
        logger.info(f"✅ Fetched {len(descriptions)}/{len(offer_ids)} descriptions from MongoDB")
        return descriptions
        
    finally:
        if mongo_client:
            mongo_client.close()


async def call_translate_api_async(
    client: httpx.AsyncClient,
    description: str,
    offer_id: str,
    semaphore: asyncio.Semaphore,
    api_url: str
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Call FastAPI translate-batch endpoint asynchronously.
    Immediately appends result to CSV upon completion.
    Returns: (offer_id, translated_html, error_message)
    """
    async with semaphore:  # Limit concurrent requests
        port = api_url.split(':')[-1].split('/')[0]
        logger.info(f"🚀 [{offer_id}] Starting translation... (port {port})")
        
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "description": description,
            "offer_id": str(offer_id),
        }
        
        try:
            response = await client.post(api_url, headers=headers, json=data, timeout=300.0)
            
            if response.status_code == 200:
                response_data = response.json()
                translated_html = response_data.get('translated_html', '')
                
                if not translated_html:
                    error_msg = "Empty translated_html in response"
                    logger.error(f"❌ [{offer_id}] {error_msg}")
                    await append_result_to_csv(offer_id, None, error_msg)
                    return (offer_id, None, error_msg)
                
                logger.info(f"✅ [{offer_id}] Translation successful ({len(translated_html)} chars) [port {port}]")
                await append_result_to_csv(offer_id, translated_html, None)
                return (offer_id, translated_html, None)
            else:
                error_msg = f"API returned status {response.status_code}"
                logger.error(f"❌ [{offer_id}] {error_msg}")
                await append_result_to_csv(offer_id, None, error_msg)
                return (offer_id, None, error_msg)
                
        except httpx.TimeoutException:
            error_msg = "Request timeout"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_result_to_csv(offer_id, None, error_msg)
            return (offer_id, None, error_msg)
        except httpx.ConnectError:
            error_msg = f"Connection error - is FastAPI running on {api_url}?"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_result_to_csv(offer_id, None, error_msg)
            return (offer_id, None, error_msg)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ [{offer_id}] {error_msg}")
            await append_result_to_csv(offer_id, None, error_msg)
            return (offer_id, None, error_msg)


async def process_all_offers_async(offer_ids: List[str], descriptions: Dict[str, str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Process all offers with limited concurrency and load balancing across multiple API instances.
    Returns: List of (offer_id, translated_html, error_message)
    """
    logger.info(f"\n🚀 Processing {len(offer_ids)} offers with max {MAX_CONCURRENT_REQUESTS} concurrent requests")
    logger.info(f"⚖️  Load balancing across {len(FASTAPI_URLS)} API instances (ports {BASE_PORT}-{BASE_PORT + NUM_API_INSTANCES - 1})\n")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with httpx.AsyncClient() as client:
        tasks = []
        url_index = 0
        
        for offer_id in offer_ids:
            description = descriptions.get(offer_id)
            if description:
                # Round-robin load balancing across API instances
                api_url = FASTAPI_URLS[url_index % len(FASTAPI_URLS)]
                url_index += 1
                
                task = call_translate_api_async(client, description, offer_id, semaphore, api_url)
                tasks.append(task)
            else:
                logger.warning(f"⚠️  Skipping {offer_id} - no description found")
        
        # Run all requests with limited concurrency
        # When one completes, the next one starts automatically
        results = await asyncio.gather(*tasks, return_exceptions=False)
    
    return results


def save_results_to_csv(results: List[Tuple[str, Optional[str], Optional[str]]]):
    """DEPRECATED - Results are now appended incrementally during processing."""
    pass


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
    
    # Check for already processed offers
    already_processed = get_already_processed_offers()
    
    # Filter out already processed offers
    offer_ids = [oid for oid in OFFER_IDS if oid not in already_processed]
    
    if not offer_ids:
        logger.info("✅ All offers already processed! Nothing to do.")
        return
    
    logger.info(f"📋 Total offers in list: {len(OFFER_IDS)}")
    logger.info(f"✅ Already processed: {len(already_processed)}")
    logger.info(f"🔄 Remaining to process: {len(offer_ids)}\n")
    
    try:
        # Step 1: Fetch all descriptions from MongoDB
        descriptions = get_descriptions_from_mongodb(offer_ids)
        
        if not descriptions:
            logger.error("❌ No descriptions found in MongoDB. Exiting.")
            sys.exit(1)
        
        # Step 2: Process all offers in parallel (results appended to CSV incrementally)
        results = asyncio.run(process_all_offers_async(offer_ids, descriptions))
        
        # Step 3: Save individual HTML files
        save_individual_html_files(results)
        
        # Summary
        successful = sum(1 for _, html, _ in results if html)
        failed = len(results) - successful
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successful: {successful}/{len(results)}")
        logger.info(f"❌ Failed: {failed}/{len(results)}")
        logger.info(f"📄 CSV Output: {CSV_OUTPUT} (appended incrementally)")
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