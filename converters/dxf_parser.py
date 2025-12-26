import ezdxf
from ezdxf.math import Vec2

class DXFParser:
    """Parses DXF files and extracts architectural elements"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.doc = None
        self.walls = []
        self.doors = []
        self.windows = []
        
    def parse(self):
        """Main parsing method"""
        try:
            self.doc = ezdxf.readfile(self.filepath)
            self.modelspace = self.doc.modelspace()
            
            self._extract_walls()
            self._extract_doors()
            self._extract_windows()
            
            return {
                'walls': self.walls,
                'doors': self.doors,
                'windows': self.windows,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_walls(self):
        """Extract wall lines from DXF"""
        # Query all LINE entities
        for entity in self.modelspace.query('LINE'):
            start = [entity.dxf.start.x, entity.dxf.start.y]
            end = [entity.dxf.end.x, entity.dxf.end.y]
            layer = entity.dxf.layer
            
            self.walls.append({
                'start': start,
                'end': end,
                'layer': layer,
                'type': 'wall'
            })
    
    def _extract_doors(self):
        """Extract door locations (ARCs typically represent door swings)"""
        for entity in self.modelspace.query('ARC'):
            center = [entity.dxf.center.x, entity.dxf.center.y]
            radius = entity.dxf.radius
            
            self.doors.append({
                'center': center,
                'radius': radius,
                'type': 'door'
            })
    
    def _extract_windows(self):
        """Extract window locations (often as LWPOLYLINE or blocks)"""
        for entity in self.modelspace.query('LWPOLYLINE'):
            points = [[p[0], p[1]] for p in entity.get_points()]
            
            # Simple heuristic: small rectangles might be windows
            if len(points) == 4 or len(points) == 5:
                self.windows.append({
                    'points': points,
                    'type': 'window'
                })
