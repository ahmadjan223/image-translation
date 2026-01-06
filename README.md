# Image Translation Script

This script converts your Jupyter notebook image translation pipeline into a standalone Python script that can process multiple images from an input folder.

## Features

✅ **Batch Processing**: Process multiple images in a folder  
✅ **Chinese Text Detection**: Uses PaddleOCR to detect Chinese text  
✅ **AI Translation**: Google Gemini API for Chinese-to-English translation  
✅ **Smart Text Fitting**: Optimized font sizing and multi-line wrapping  
✅ **AI Inpainting**: SimpleLama for removing original text  
✅ **Clean Output**: Professional text overlay with shadows and contrast optimization  

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Usage

### Basic Usage
```bash
python3 translate_images.py input_folder output_folder
```

### With API Key
```bash
python3 translate_images.py input_folder output_folder --gemini-key YOUR_API_KEY
```

### Advanced Options
```bash
python3 translate_images.py input_folder output_folder \
    --model gemini-flash-lite-latest \
    --extensions jpg,png,webp
```

## Examples

Process all images in `./images/` and save to `./translated/`:
```bash
python3 translate_images.py ./images ./translated
```

Process only JPG files with a specific model:
```bash
python3 translate_images.py ./input ./output \
    --model gemini-2.0-flash-exp \
    --extensions jpg,jpeg
```

## Configuration

The script includes optimized default parameters:
- **Font sizing**: 12-120px range with 2px steps
- **Text fitting**: Single-line preferred, multi-line fallback
- **Line spacing**: 1.025x for readability
- **Shadow effects**: 1px blur with 1px offset
- **Box padding**: 2px inner padding

## Supported Formats

Input images: JPG, JPEG, PNG, WebP, BMP, TIF, TIFF  
Output images: Same format as input

## Requirements

- Python 3.7+
- Google Gemini API key
- System fonts (DejaVu recommended)

## Error Handling

The script handles:
- Missing fonts (tries multiple candidates)
- Translation failures (graceful fallback)
- Inpainting errors (OpenCV fallback)
- Invalid image formats
- Network timeouts

## Output

Each processed image is saved as `{original_name}_translated.{ext}` in the output folder.

Progress is shown with:
- 🔄 Processing status
- 📝 Detection counts  
- 🔤 Translation counts
- ✅ Success confirmations
- ❌ Error details