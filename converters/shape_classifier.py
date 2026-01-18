import cv2
import numpy as np
from collections import defaultdict

class ShapeClassifier:
    """Classify architectural elements using traditional CV techniques"""
    
    def __init__(self):
        self.element_types = {
            'wall': [],
            'door': [],
            'window': [],
            'furniture': [],
            'unknown': []
        }
    
    def classify_from_image(self, image_path):
        """
        Classify elements from blueprint image
        Returns: dict of classified elements with coordinates
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not read image'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            edges, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Classify each contour
        for i, contour in enumerate(contours):
            # Skip very small contours (noise)
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            # Get contour properties
            classification = self._classify_contour(contour, hierarchy[0][i] if hierarchy is not None else None)
            
            # Extract bounding box and center
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            element = {
                'bbox': [x, y, w, h],
                'center': [center_x, center_y],
                'area': area,
                'contour': contour.tolist()
            }
            
            self.element_types[classification].append(element)
        
        return self.element_types
    
    def _classify_contour(self, contour, hierarchy_info):
        """Classify a single contour based on geometric properties"""
        # Calculate properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 'unknown'
        
        # Circularity: 4π(area/perimeter²)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Approximate polygon
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # Classification rules
        
        # DOOR: Arc/semicircle shape (high circularity) or specific arc pattern
        if circularity > 0.7 and area < 5000:
            return 'door'
        
        # WINDOW: Small rectangle, specific aspect ratio
        if num_vertices == 4 and 200 < area < 3000:
            if 0.3 < aspect_ratio < 3.0:
                return 'window'
        
        # WALL: Long, thin rectangle
        if num_vertices == 4:
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                if area > 1000:
                    return 'wall'
        
        # FURNITURE: Complex shapes with multiple vertices
        if num_vertices > 6 and area > 1000:
            return 'furniture'
        
        # FURNITURE: Circles (chairs, tables from top view)
        if circularity > 0.8 and area > 500:
            return 'furniture'
        
        return 'unknown'
    
    def filter_by_type(self, element_type):
        """Get all elements of a specific type"""
        return self.element_types.get(element_type, [])
    
    def visualize_classification(self, image_path, output_path):
        """Draw classified elements on image with color coding"""
        img = cv2.imread(image_path)
        
        # Color mapping
        colors = {
            'wall': (0, 255, 0),      # Green
            'door': (255, 0, 0),      # Blue
            'window': (0, 255, 255),  # Yellow
            'furniture': (0, 165, 255), # Orange
            'unknown': (128, 128, 128) # Gray
        }
        
        for element_type, elements in self.element_types.items():
            color = colors.get(element_type, (255, 255, 255))
            
            for element in elements:
                x, y, w, h = element['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                cv2.putText(
                    img, 
                    element_type, 
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        cv2.imwrite(output_path, img)
        return output_path
