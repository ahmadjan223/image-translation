"""
Test script to verify GCP Cloud Storage connection
"""
from gcp_storage import GCPCloudStorage
from PIL import Image
import io

def test_gcp_connection():
    """Test GCP Storage initialization"""
    print("=" * 50)
    print("Testing GCP Cloud Storage Connection")
    print("=" * 50)
    
    # Initialize GCP Storage
    print("\n1. Initializing GCP Storage...")
    gcs = GCPCloudStorage()
    
    if not gcs.client:
        print("❌ FAILED: Could not initialize GCP Storage client")
        return False
    
    print(f"✅ SUCCESS: GCP Storage initialized")
    print(f"   Project: {gcs.client.project}")
    print(f"   Bucket: {gcs.bucket_name}")
    
    # Test bucket access
    print("\n2. Testing bucket access...")
    try:
        bucket_exists = gcs.bucket.exists()
        if bucket_exists:
            print(f"✅ SUCCESS: Bucket '{gcs.bucket_name}' exists and is accessible")
        else:
            print(f"❌ FAILED: Bucket '{gcs.bucket_name}' does not exist")
            return False
    except Exception as e:
        print(f"❌ FAILED: Cannot access bucket - {e}")
        return False
    
    # Test upload with a simple test image
    print("\n3. Testing image upload...")
    try:
        # Create a simple test image (100x100 red square)
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Convert to WebP
        webp_data = gcs.convert_to_webp(img_bytes)
        if not webp_data:
            print("❌ FAILED: Could not convert image to WebP")
            return False
        
        print(f"✅ SUCCESS: Converted test image to WebP ({len(webp_data)} bytes)")
        
        # Upload test image
        blob_path = gcs.upload_image(
            image_data=webp_data,
            folder_name="test",
            image_name="test_image",
            content_type="image/webp"
        )
        
        if not blob_path:
            print("❌ FAILED: Could not upload test image")
            return False
        
        print(f"✅ SUCCESS: Uploaded test image to: {blob_path}")
        
    except Exception as e:
        print(f"❌ FAILED: Upload test failed - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test getting public URL
    print("\n4. Testing public URL generation...")
    try:
        public_url = gcs.get_public_url(blob_path, use_cdn=False)
        if not public_url:
            print("❌ FAILED: Could not generate public URL")
            return False
        
        print(f"✅ SUCCESS: Generated public URL")
        print(f"   URL: {public_url}")
        print(f"\n   🌐 Open this URL in your browser to verify it's publicly accessible!")
        
    except Exception as e:
        print(f"❌ FAILED: URL generation failed - {e}")
        return False
    
    # Test cleanup (delete test image)
    print("\n5. Cleaning up test image...")
    try:
        blob = gcs.bucket.blob(blob_path)
        blob.delete()
        print(f"✅ SUCCESS: Deleted test image")
    except Exception as e:
        print(f"⚠️  WARNING: Could not delete test image - {e}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYour GCP Cloud Storage is configured correctly!")
    return True

if __name__ == "__main__":
    import sys
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    success = test_gcp_connection()
    sys.exit(0 if success else 1)
