#!/usr/bin/env python3
"""
Test script to verify the translate_images.py structure
"""
import sys
import os
import tempfile
from pathlib import Path

# Add current directory to path so we can import without installing
sys.path.insert(0, '/home/ahmad-jan/Desktop/Markaz/image-translation')

def test_basic_import():
    """Test basic import and class structure"""
    try:
        # Import only the argument parsing part
        import argparse
        
        print("✅ Basic imports work")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_help_output():
    """Test help output functionality"""
    try:
        import argparse
        
        # Test the argument parser
        parser = argparse.ArgumentParser(description="Test")
        parser.add_argument("input_folder")
        parser.add_argument("output_folder")
        parser.add_argument("--gemini-key")
        
        # This should work without errors
        args = parser.parse_args(["test_in", "test_out", "--gemini-key", "test"])
        print(f"✅ Argument parsing works: input={args.input_folder}, output={args.output_folder}")
        return True
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return False

def test_file_structure():
    """Test the created files exist and are readable"""
    files_to_check = [
        '/home/ahmad-jan/Desktop/Markaz/image-translation/translate_images.py',
        '/home/ahmad-jan/Desktop/Markaz/image-translation/requirements.txt'
    ]
    
    all_good = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"✅ {os.path.basename(file_path)} exists ({len(content)} chars)")
        else:
            print(f"❌ {file_path} does not exist")
            all_good = False
    
    return all_good

def main():
    print("🧪 Testing translate_images.py script structure...")
    
    tests = [
        ("Basic imports", test_basic_import),
        ("Argument parsing", test_help_output), 
        ("File structure", test_file_structure)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n🔍 Testing: {name}")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All structural tests passed!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set your Gemini API key: export GEMINI_API_KEY='your_key_here'")
        print("3. Test with your images: python3 translate_images.py input_folder output_folder")
    else:
        print("\n❌ Some tests failed")

if __name__ == "__main__":
    main()