from utils.geometry import normalize_coordinates
import math

class MeshGenerator:
    """Converts 2D blueprint data to 3D mesh"""
    
    def __init__(self, blueprint_data, wall_height=3.0, wall_thickness=0.15):
        self.blueprint_data = blueprint_data
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.vertices = []
        self.faces = []
        
    def generate(self):
        """Generate 3D mesh from blueprint data"""
        walls = self.blueprint_data.get('walls', [])
        doors = self.blueprint_data.get('doors', [])
        
        # Normalize coordinates - USE 1.0 FOR REALISTIC SIZE
        walls = normalize_coordinates(walls, scale=1.0)  # CHANGE THIS FROM 0.1 TO 1.0
        
        # Generate 3D geometry
        self._extrude_walls(walls)
        
        return {
            'vertices': self.vertices,
            'faces': self.faces,
            'metadata': {
                'wall_count': len(walls),
                'wall_height': self.wall_height
            }
        }

    
    def _extrude_walls(self, walls):
        """Extrude 2D wall lines into 3D boxes"""
        for wall in walls:
            start = wall['start']
            end = wall['end']
            
            # Create bottom vertices
            v0 = [start[0], 0, start[1]]
            v1 = [end[0], 0, end[1]]
            
            # Create top vertices
            v2 = [end[0], self.wall_height, end[1]]
            v3 = [start[0], self.wall_height, start[1]]
            
            # Add vertices
            base_idx = len(self.vertices)
            self.vertices.extend([v0, v1, v2, v3])
            
            # Create faces (two triangles per wall face)
            self.faces.append([base_idx, base_idx + 1, base_idx + 2])
            self.faces.append([base_idx, base_idx + 2, base_idx + 3])
