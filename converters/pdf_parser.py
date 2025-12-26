import pdfplumber
from PIL import Image
import cv2
import numpy as np

class PDFParser:
    """Parses PDF blueprints and extracts line coordinates"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.walls = []
        
    def parse(self):
        """Main parsing method"""
        try:
            # Try vector extraction first
            result = self._parse_vector_pdf()
            
            if not result['success']:
                # Fall back to image-based extraction
                result = self._parse_image_pdf()
            
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_vector_pdf(self):
        """Extract lines from vector-based PDF"""
        try:
            with pdfplumber.open(self.filepath) as pdf:
                page = pdf.pages[0]
                
                # Extract line objects
                lines = page.lines
                
                for line in lines:
                    self.walls.append({
                        'start': [line['x0'], line['y0']],
                        'end': [line['x1'], line['y1']],
                        'type': 'wall'
                    })
                
                return {
                    'walls': self.walls,
                    'doors': [],
                    'windows': [],
                    'success': True
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_image_pdf(self):
        """Extract lines using OpenCV edge detection"""
        try:
            with pdfplumber.open(self.filepath) as pdf:
                page = pdf.pages[0]
                img = page.to_image(resolution=150)
                
                # Convert to OpenCV format
                img_array = np.array(img.original)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Edge detection
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                
                # Detect lines using Hough transform
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                       minLineLength=50, maxLineGap=10)
                
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        self.walls.append({
                            'start': [float(x1), float(y1)],
                            'end': [float(x2), float(y2)],
                            'type': 'wall'
                        })
                
                return {
                    'walls': self.walls,
                    'doors': [],
                    'windows': [],
                    'success': True
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
