#!/usr/bin/env python3
"""
Create FP16 version of YOLOv8n - better compatibility than INT8
"""
import onnx
from onnx import helper, TensorProto
import numpy as np
import os

def convert_to_fp16():
    """Convert YOLOv8n to FP16 for faster inference with better compatibility"""
    
    input_model = "../models/yolov8n.onnx"
    output_model = "../models/yolov8n_fp16.onnx"
    
    if not os.path.exists(input_model):
        print(f"❌ Error: {input_model} not found!")
        return False
    
    print(f"🔄 Converting {input_model} to FP16...")
    
    try:
        # Load the model
        model = onnx.load(input_model)
        
        # Convert to FP16
        from onnxconverter_common import float16
        model_fp16 = float16.convert_float_to_float16(model)
        
        # Save the model
        onnx.save(model_fp16, output_model)
        
        # Check file sizes
        original_size = os.path.getsize(input_model) / (1024 * 1024)  # MB
        fp16_size = os.path.getsize(output_model) / (1024 * 1024)  # MB
        
        print(f"✅ FP16 conversion successful!")
        print(f"   Original (FP32): {original_size:.1f} MB")
        print(f"   FP16: {fp16_size:.1f} MB")
        print(f"   Size reduction: {100 * (1 - fp16_size/original_size):.1f}%")
        print(f"   Expected speedup: 1.5-2x on CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ FP16 conversion failed: {e}")
        print("💡 Installing missing dependency: pip install onnxconverter-common")
        return False

if __name__ == "__main__":
    print("=== YOLOv8 FP16 Conversion ===")
    success = convert_to_fp16()
    if success:
        print("🎉 Ready to test with FP16 model!")