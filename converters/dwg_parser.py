import subprocess
import os
import platform

class DWGParser:
    """Smart cross-platform DWG parser - uses best method per OS"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.walls = []
        self.doors = []
        self.windows = []
        self.platform = platform.system()
        
    def parse(self):
        """Parse DWG using platform-appropriate method"""
        print(f"Parsing DWG on {self.platform}")
        
        if self.platform == "Windows":
            return self._parse_windows()
        elif self.platform == "Linux":
            return self._parse_linux()
        elif self.platform == "Darwin":
            return self._parse_linux()
        else:
            return {
                'success': False,
                'error': f'DWG parsing not supported on {self.platform}'
            }
    
    def _parse_windows(self):
        """Windows: Use Aspose.CAD"""
        try:
            print("Attempting to use Aspose.CAD (Windows native)...")
            import aspose.cad as cad
            from aspose.cad import Image
            
            image = Image.load(self.filepath)
            print(f"✓ DWG loaded with Aspose.CAD")
            
            if hasattr(image, 'entities'):
                entity_count = 0
                for entity in image.entities:
                    self._extract_entity_aspose(entity)
                    entity_count += 1
                print(f"✓ Processed {entity_count} entities")
            
            print(f"✓ Extracted: {len(self.walls)} walls, {len(self.doors)} doors, {len(self.windows)} windows")
            
            return {
                'walls': self.walls,
                'doors': self.doors,
                'windows': self.windows,
                'success': True,
                'method': 'Aspose.CAD'
            }
            
        except ImportError:
            print("Aspose.CAD not installed")
            return {
                'success': False,
                'error': 'Aspose.CAD not installed',
                'message': 'Install with: pip install aspose-cad'
            }
        except Exception as e:
            print(f"Aspose.CAD error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Aspose.CAD parsing failed: {str(e)}'
            }
    
    def _parse_linux(self):
        """Linux/macOS: Use LibreDWG"""
        from .dxf_parser import DXFParser
        
        try:
            base_name = os.path.splitext(self.filepath)[0]
            dxf_path = f"{base_name}_converted.dxf"
            
            # Delete existing DXF file if it exists
            if os.path.exists(dxf_path):
                print(f"Removing existing file: {dxf_path}")
                os.remove(dxf_path)
            
            print(f"Converting to DXF using LibreDWG...")
            
            # Add -y flag to force overwrite and -v3 for verbose output
            result = subprocess.run(
                ['dwg2dxf', '-y', '-o', dxf_path, self.filepath],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # LibreDWG often shows warnings but still succeeds
            if result.stdout:
                print(f"LibreDWG output: {result.stdout}")
            if result.stderr:
                print(f"LibreDWG warnings: {result.stderr}")
            
            # Check if file was actually created (more reliable than returncode)
            if not os.path.exists(dxf_path):
                return {
                    'success': False,
                    'error': 'DXF file was not created by LibreDWG',
                    'details': result.stderr or result.stdout
                }
            
            file_size = os.path.getsize(dxf_path)
            
            # Check if file has content
            if file_size == 0:
                return {
                    'success': False,
                    'error': 'Generated DXF file is empty'
                }
            
            print(f"✓ Conversion successful: {dxf_path} ({file_size} bytes)")
            
            # Parse the DXF
            print("Parsing converted DXF...")
            dxf_parser = DXFParser(dxf_path)
            result = dxf_parser.parse()
            
            if result.get('success'):
                print(f"✓ Parsing complete!")
                print(f"  - Walls: {len(result.get('walls', []))}")
                print(f"  - Doors: {len(result.get('doors', []))}")
                print(f"  - Windows: {len(result.get('windows', []))}")
                result['method'] = 'LibreDWG'
            
            return result
            
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'LibreDWG not installed',
                'message': 'dwg2dxf command not found in PATH'
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Conversion timed out (file too large)'
            }
        except Exception as e:
            print(f"LibreDWG error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'DWG parsing failed: {str(e)}'
            }
    
    def _extract_entity_aspose(self, entity):
        """Extract entities using Aspose.CAD API"""
        try:
            type_name = str(entity.type_name) if hasattr(entity, 'type_name') else str(type(entity))
            
            if 'LINE' in type_name.upper():
                if hasattr(entity, 'start_point') and hasattr(entity, 'end_point'):
                    start = entity.start_point
                    end = entity.end_point
                    
                    self.walls.append({
                        'start': [start.x, start.y],
                        'end': [end.x, end.y],
                        'layer': getattr(entity, 'layer_name', 'default'),
                        'type': 'wall'
                    })
            
            elif 'ARC' in type_name.upper():
                if hasattr(entity, 'center_point') and hasattr(entity, 'radius'):
                    center = entity.center_point
                    
                    self.doors.append({
                        'center': [center.x, center.y],
                        'radius': entity.radius,
                        'type': 'door'
                    })
            
            elif 'POLYLINE' in type_name.upper() or 'LWPOLYLINE' in type_name.upper():
                if hasattr(entity, 'vertices'):
                    points = []
                    for vertex in entity.vertices:
                        if hasattr(vertex, 'location'):
                            loc = vertex.location
                            points.append([loc.x, loc.y])
                    
                    if len(points) >= 3:
                        self.windows.append({
                            'points': points,
                            'type': 'window'
                        })
        
        except Exception as e:
            print(f"Error extracting entity: {e}")
