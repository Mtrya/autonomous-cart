"""
Qwen2.5-VL Single Object Detection and Annotation Test Script
For autonomous cart project - single class, single object per image
"""

import cv2
import json
import base64
from openai import OpenAI
from PIL import Image
import numpy as np
import re
from typing import Tuple, Optional, Dict
import os

class QwenBBoxDetector:
    def __init__(self, api_key: str, base_url: str, model: str="qwen2.5-vl-7b-instruct"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def detect_single_object(self, image_path: str, object_class: str="white sticker", description: str="white long sticker with yellow and black stripes above and below") -> Dict:
        """
        Args:
            image_path: Path to the input image
            object_class: Description of the object to detect
            description: More detailed description of the object

        Returns:
            Dictionary with detection results and metadata
        """
        base64_image = self.encode_image_to_base64(image_path)

        image = Image.open(image_path)
        img_width, img_height = image.size

        prompt = f"""
        Detect the {description} in this image. There should be exactly one {object_class} visible.

        Return the result in JSON format with absolute pixel coordinates:
        {{
            "bbox_2d": [x1, y1, x2, y2],
            "label": "{object_class}",
            "confidence": "high/medium/low",
            "found": true/false
        }}
        
        Where:
        - x1, y1 are top-left coordinates 
        - x2, y2 are bottom-right coordinates
        - Coordinates are in pixels (0 to {img_width} for x, 0 to {img_height} for y)
        - Only return the JSON, no other text
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url","image_url":{"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]}
            ],
            max_tokens=200,
            temperature=0.1
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        detection_result = self.parse_detection_response(response_text, img_width, img_height)
        detection_result['raw_response'] = response_text
        detection_result['image_dimensions'] = (img_width, img_height)
        
        return detection_result
            
        
    def parse_detection_response(self, response_text: str, img_width: int, img_height: int) -> Dict:
        """Parse and validate the model's JSON response"""
        try:
            # Extract JSON from response (handle cases where model adds extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            detection = json.loads(json_str)
            
            # Validate required fields
            if 'bbox_2d' not in detection or 'found' not in detection:
                raise ValueError("Missing required fields in JSON response")
            
            # Validate bounding box if object was found
            if detection.get('found', False) and detection['bbox_2d']:
                bbox = detection['bbox_2d']
                if len(bbox) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
                
                x1, y1, x2, y2 = bbox
                
                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                # Ensure valid box (x2 > x1, y2 > y1)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Invalid bounding box coordinates")
                
                detection['bbox_2d'] = [int(x1), int(y1), int(x2), int(y2)]
            
            return detection
            
        except Exception as e:
            return {
                'error': f"Failed to parse response: {str(e)}",
                'found': False,
                'bbox_2d': None,
                'raw_response': response_text
            }
        
    def visualize_detection(self, image_path: str, detection_result: Dict, 
                          output_path: Optional[str] = None, show_image: bool = False) -> np.ndarray:
        """
        Draw bounding box on image and display/save result
        
        Args:
            image_path: Path to original image
            detection_result: Result from detect_single_object()
            output_path: Path to save annotated image (optional)
            show_image: Whether to display the image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        annotated_image = image.copy()
        
        # Draw bounding box if object was found
        if detection_result.get('found', False) and detection_result.get('bbox_2d'):
            x1, y1, x2, y2 = detection_result['bbox_2d']
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label
            label = detection_result.get('label', 'Object')
            confidence = detection_result.get('confidence', 'unknown')
            text = f"{label} ({confidence})"
            
            # Calculate text size and background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(annotated_image, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(annotated_image, text, (x1, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
            
            print(f"✅ Detection successful!")
            print(f"   Label: {label}")
            print(f"   Confidence: {confidence}")
            print(f"   Bounding box: [{x1}, {y1}, {x2}, {y2}]")
            print(f"   Box size: {x2-x1}x{y2-y1} pixels")
        else:
            # Add "No object found" text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "No object detected"
            cv2.putText(annotated_image, text, (50, 50), font, 1, (0, 0, 255), 2)
            print("❌ No object detected")
        
        # Print any errors
        if 'error' in detection_result:
            print(f"⚠️ Error: {detection_result['error']}")
        
        # Save image if requested
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Saved annotated image to: {output_path}")
        
        # Display image if requested
        if show_image:
            cv2.imshow('Object Detection Result', annotated_image)
            print("Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_image
    

def main():
    CONFIG = {
        "api_key": os.getenv("SILICONFLOW_API_KEY"),
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
        "image_path": "image.png",
        "object_class": "white sticker",
        "description": "white long sticker with yellow and black stripes above and below",
        "output_path": "../plots/annotated_result.png"
    }

    detector = QwenBBoxDetector(
        api_key=CONFIG['api_key'],
        base_url=CONFIG['base_url'],
        model=CONFIG['model']
    )

    test_image = Image.open(CONFIG['image_path'])
    img_width, img_height = test_image.size

    detection_result = detector.detect_single_object(
        CONFIG['image_path'],
        CONFIG['object_class'],
        CONFIG['description']
    )

    if 'raw_response' in detection_result:
        print(f"\nRaw model response:")
        print(detection_result["raw_response"])
    else:
        print("???")

    print(f"\nVisualizing detection results...")
    try:
        annotated_image = detector.visualize_detection(
            CONFIG['image_path'],
            detection_result,
            output_path=CONFIG['output_path'],
            show_image=False
        )
        print("Visualization complete!")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print(f"\nDetection Summary:")
    print(f"   Object found: {detection_result.get('found', False)}")
    if detection_result.get('bbox_2d'):
        bbox = detection_result['bbox_2d']
        print(f"   YOLO format (normalized): {bbox[0]/img_width:.6f} {bbox[1]/img_height:.6f} {(bbox[2]-bbox[0])/img_width:.6f} {(bbox[3]-bbox[1])/img_height:.6f}")

if __name__ == "__main__":
    main()