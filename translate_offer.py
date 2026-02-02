#!/usr/bin/env python3
"""
Local script to translate product descriptions using FastAPI translate-batch endpoint.

Usage:
    python translate_offer.py <offer_id1> <offer_id2> <offer_id3> ...

Examples:
    python translate_offer.py 828176815369
    python translate_offer.py 845889051211 576465233813 694502306556
    python translate_offer.py 694502306556 686907786697 682407541428 722940744852 689643946134 645169076794

Output:
  Translated HTML files will be saved to: translated_offers/{offer_id}.html
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests
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

# FastAPI local endpoint
FASTAPI_URL = "http://localhost:8080/translate-batch"

# Output directory for translated HTML files
OUTPUT_DIR = Path("translated_offers")


def get_description_from_mongodb(offer_id: str) -> str:
    """Fetch description from MongoDB for given offer_id."""
    logger.info(f"🔗 Connecting to MongoDB...")
    
    if not MONGODB_CONNECTION_STRING:
        raise ValueError("mongodb_connection_string environment variable not set")
    if not MONGODB_DATABASE_NAME:
        raise ValueError("mongodb_database_name environment variable not set")
    
    mongo_client = None
    try:
        mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
        db = mongo_client[MONGODB_DATABASE_NAME]
        collection = db[MONGODB_COLLECTION_NAME]
        
        offer_id_int = int(offer_id)
        document = collection.find_one({"offerId": offer_id_int})
        
        if not document:
            raise ValueError(f"No document found for offerId {offer_id}")
        
        description = document.get('description', '')
        if not description:
            raise ValueError(f"Empty description for offerId {offer_id}")
        
        logger.info(f"✅ Retrieved description for offerId {offer_id} ({len(description)} characters)")
        return description
        
    finally:
        if mongo_client:
            mongo_client.close()


def call_translate_api(description: str, offer_id: str) -> str:
    """Call FastAPI translate-batch endpoint."""
    logger.info(f"🔗 Calling FastAPI translate-batch for offerId {offer_id}")
    logger.info(f"📤 API URL: {FASTAPI_URL}")
    
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "description": description,
        "offer_id": str(offer_id),
    }
    
    try:
        logger.info(f"⏳ Sending request (description length: {len(description)} chars)...")
        response = requests.post(FASTAPI_URL, headers=headers, json=data, timeout=300)
        
        logger.info(f"📥 Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            translated_html = response_data.get('translated_html', '')
            
            if not translated_html:
                raise ValueError("Empty translated_html in response")
            
            logger.info(f"✅ Translation successful (output length: {len(translated_html)} chars)")
            return translated_html
        else:
            logger.error(f"❌ API returned status {response.status_code}")
            logger.error(f"Response: {response.text}")
            raise Exception(f"API call failed with status {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Request timeout for offerId {offer_id}")
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection error - is FastAPI running on localhost:8080?")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON response: {e}")
        logger.error(f"Raw response: {response.text}")
        raise


def save_translated_html(translated_html: str, offer_id: str):
    """Save translated HTML to file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{offer_id}.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(translated_html)
    
    logger.info(f"💾 Saved translated HTML to: {output_path}")
    logger.info(f"📏 File size: {len(translated_html)} characters")


def process_single_offer(offer_id: str) -> bool:
    """Process a single offer. Returns True if successful."""
    try:
        logger.info(f"🚀 Starting translation for offerId: {offer_id}")
        logger.info("=" * 60)
        
        # Step 1: Get description from MongoDB
        description = get_description_from_mongodb(offer_id)
        
        # Step 2: Call FastAPI translate-batch
        translated_html = call_translate_api(description, offer_id)
        
        # Step 3: Save to file
        save_translated_html(translated_html, offer_id)
        
        logger.info("=" * 60)
        logger.info(f"✅ Translation completed successfully for offerId {offer_id}\n")
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ Error processing offerId {offer_id}: {e}")
        logger.error("=" * 60 + "\n")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Translate product descriptions using FastAPI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python translate_offer.py 828176815369
  python translate_offer.py 845889051211 576465233813
  python translate_offer.py 694502306556 686907786697 682407541428 722940744852 689643946134 645169076794

Output:
  Translated HTML files will be saved to: translated_offers/{offer_id}.html
        """
    )
    parser.add_argument('offer_ids', nargs='+', type=str, help='One or more offer IDs to translate')
    
    args = parser.parse_args()
    offer_ids = args.offer_ids
    
    logger.info(f"\n🚀 Processing {len(offer_ids)} offer(s) sequentially\n")
    
    successful = 0
    failed = 0
    failed_ids = []
    
    for idx, offer_id in enumerate(offer_ids, 1):
        logger.info(f"\n📍 [{idx}/{len(offer_ids)}] Processing offer: {offer_id}")
        if process_single_offer(offer_id):
            successful += 1
        else:
            failed += 1
            failed_ids.append(offer_id)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {successful}/{len(offer_ids)}")
    logger.info(f"❌ Failed: {failed}/{len(offer_ids)}")
    if failed_ids:
        logger.info(f"Failed offer IDs: {', '.join(failed_ids)}")
    logger.info("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()