# python/yolo_setup.py
import os
import sys
from pathlib import Path
import onnx
import onnxruntime as ort
from ultralytics import YOLO
import numpy as np

def export_yolo_to_onnx():
    """Export YOLOv8 model with OpenCV-compatible settings"""
    
    # Get absolute path to avoid ~ expansion issues
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Models directory: {models_dir}")
    
    try:
        # Load YOLOv8 nano model
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        
        # Export with OpenCV-compatible settings
        print("Exporting to ONNX format...")
        export_path = model.export(
            format='onnx',
            imgsz=640,
            optimize=True,       
            dynamic=False,         # Fixed input size
            simplify=True,         # Simplify model graph
            opset=11,              # OpenCV works best with opset 11
            nms=False,             # Disable built-in NMS for manual control
            agnostic_nms=False,
            half=True,            # Use FP32 for better compatibility
            device='cpu'           # Export on CPU for consistency
        )
        
        # Move the exported file to correct location
        export_file = Path(export_path)
        target_file = models_dir / "yolov8n.onnx"
        
        if export_file.exists():
            if target_file.exists():
                target_file.unlink()  # Remove existing file
            
            export_file.rename(target_file)
            print(f"✅ Model exported successfully to: {target_file}")
        else:
            raise FileNotFoundError(f"Export failed: {export_path} not found")
        
        # Validate the exported model
        validate_onnx_model(target_file)
        
        return target_file
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

def validate_onnx_model(model_path):
    """Validate ONNX model compatibility"""
    print(f"\nValidating ONNX model: {model_path}")
    
    try:
        # Check if file exists and has reasonable size
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        if file_size < 1:  # YOLOv8n should be ~6MB
            raise ValueError(f"Model file too small ({file_size:.2f} MB), likely corrupted")
        
        # Load with ONNX library
        print("Loading with ONNX library...")
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")
        
        # Test with ONNX Runtime
        print("Testing with ONNX Runtime...")
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        # Print input/output info
        print("\nModel Information:")
        for input_meta in session.get_inputs():
            print(f"Input: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")
        
        for output_meta in session.get_outputs():
            print(f"Output: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")
        
        print("✅ ONNX Runtime can load the model")
        
        # Test inference with dummy data
        import numpy as np
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
        print(f"✅ Test inference successful, output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        raise

def test_opencv_compatibility(model_path):
    """Test if OpenCV can load the model"""
    try:
        import cv2
        print(f"\nTesting OpenCV compatibility...")
        net = cv2.dnn.readNetFromONNX(str(model_path))
        print("✅ OpenCV can load the ONNX model")
        
        # Test with dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        blob = cv2.dnn.blobFromImage(dummy_input, 1.0, (640, 640), (0, 0, 0), True, False)
        net.setInput(blob)
        outputs = net.forward()
        print(f"✅ OpenCV inference successful, output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"❌ OpenCV compatibility test failed: {e}")
        print("This model may not be compatible with your OpenCV version")
        raise

if __name__ == "__main__":
    import onnx
    import onnxruntime
    
    # Export and validate model
    model_path = export_yolo_to_onnx()
    
    # Test OpenCV compatibility
    #test_opencv_compatibility(model_path)
    
    print(f"\n🎉 Setup complete! Model ready at: {model_path}")
