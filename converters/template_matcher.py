import cv2
import numpy as np

class TemplateMatcher:
    """Match architectural symbols using template matching"""
    
    def __init__(self, template_dir='templates'):
        self.template_dir = template_dir
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load template images for doors, windows, etc."""
        import os
        
        template_types = ['door', 'window', 'furniture']
        
        for t_type in template_types:
            template_path = os.path.join(self.template_dir, f'{t_type}.png')
            if os.path.exists(template_path):
                template = cv2.imread(template_path, 0)
                self.templates[t_type] = template
    
    def find_matches(self, image_path, element_type, threshold=0.7):
        """
        Find all instances of a template in the image
        
        Args:
            image_path: Path to blueprint image
            element_type: Type of element to find ('door', 'window', etc.)
            threshold: Matching threshold (0-1)
        
        Returns:
            List of bounding boxes where matches found
        """
        if element_type not in self.templates:
            return []
        
        img = cv2.imread(image_path, 0)
        template = self.templates[element_type]
        
        # Multiple scale matching
        matches = []
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        for scale in scales:
            # Resize template
            width = int(template.shape[1] * scale)
            height = int(template.shape[0] * scale)
            resized_template = cv2.resize(template, (width, height))
            
            if resized_template.shape[0] > img.shape[0] or resized_template.shape[1] > img.shape[1]:
                continue
            
            # Template matching
            result = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                matches.append({
                    'position': list(pt),
                    'size': [width, height],
                    'confidence': float(result[pt[1], pt[0]]),
                    'type': element_type
                })
        
        # Non-maximum suppression to remove overlapping detections
        matches = self._non_max_suppression(matches)
        
        return matches
    
    def _non_max_suppression(self, matches, overlap_threshold=0.3):
        """Remove overlapping bounding boxes"""
        if len(matches) == 0:
            return []
        
        # Sort by confidence
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        
        for match in matches:
            # Check if overlaps with any kept match
            should_keep = True
            x1, y1 = match['position']
            w1, h1 = match['size']
            
            for kept_match in kept:
                x2, y2 = kept_match['position']
                w2, h2 = kept_match['size']
                
                # Calculate intersection over union
                iou = self._calculate_iou(
                    [x1, y1, w1, h1],
                    [x2, y2, w2, h2]
                )
                
                if iou > overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(match)
        
        return kept
    
    def _calculate_iou(self, box1, box2):
        """Calculate intersection over union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
