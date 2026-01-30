#!/bin/bash
# Quick script to enable GPU mode in config.py

sed -i 's/"use_gpu": False,  # Force CPU/"use_gpu": True,  # GPU mode enabled/' /root/image-translation/config.py
echo "✅ GPU mode enabled in config.py"
echo "💡 Now start with: ./start_with_gpu.sh"
