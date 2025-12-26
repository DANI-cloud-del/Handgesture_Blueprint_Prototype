"""Geometry utility functions for blueprint processing"""

def normalize_coordinates(walls, scale=1.0):
    """
    Normalize wall coordinates to a reasonable scale
    
    Args:
        walls: List of wall dictionaries with 'start' and 'end' coordinates
        scale: Scale factor to apply
    
    Returns:
        List of walls with normalized coordinates
    """
    if not walls:
        return []
    
    # Find min/max coordinates to center the model
    all_x = []
    all_y = []
    
    for wall in walls:
        all_x.extend([wall['start'][0], wall['end'][0]])
        all_y.extend([wall['start'][1], wall['end'][1]])
    
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    # Center point
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Normalize and scale
    normalized_walls = []
    for wall in walls:
        normalized_wall = {
            'start': [
                (wall['start'][0] - center_x) * scale,
                (wall['start'][1] - center_y) * scale
            ],
            'end': [
                (wall['end'][0] - center_x) * scale,
                (wall['end'][1] - center_y) * scale
            ],
            'layer': wall.get('layer', 'default'),
            'type': wall.get('type', 'wall')
        }
        normalized_walls.append(normalized_wall)
    
    return normalized_walls


def line_intersects_circle(line_start, line_end, circle_center, circle_radius):
    """
    Check if a line segment intersects with a circle (for door detection)
    
    Args:
        line_start: [x, y] start of line
        line_end: [x, y] end of line
        circle_center: [x, y] center of circle
        circle_radius: radius of circle
    
    Returns:
        Boolean indicating intersection
    """
    import math
    
    # Vector from start to end
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    
    # Vector from start to circle center
    fx = line_start[0] - circle_center[0]
    fy = line_start[1] - circle_center[1]
    
    # Quadratic equation coefficients
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - circle_radius * circle_radius
    
    discriminant = b * b - 4 * a * c
    
    # No intersection
    if discriminant < 0:
        return False
    
    # Calculate intersection points
    discriminant = math.sqrt(discriminant)
    
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    
    # Check if intersection is within line segment (t between 0 and 1)
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    return False


def find_wall_for_opening(opening_position, walls, threshold=1.0):
    """
    Find which wall an opening (door/window) belongs to
    
    Args:
        opening_position: [x, y] position of opening
        walls: List of wall dictionaries
        threshold: Distance threshold for matching
    
    Returns:
        Index of matching wall or None
    """
    import math
    
    for i, wall in enumerate(walls):
        # Calculate distance from point to line segment
        start = wall['start']
        end = wall['end']
        
        # Line segment length
        line_length = math.sqrt(
            (end[0] - start[0])**2 + (end[1] - start[1])**2
        )
        
        if line_length == 0:
            continue
        
        # Calculate perpendicular distance
        t = max(0, min(1, (
            (opening_position[0] - start[0]) * (end[0] - start[0]) +
            (opening_position[1] - start[1]) * (end[1] - start[1])
        ) / (line_length ** 2)))
        
        # Nearest point on line
        nearest_x = start[0] + t * (end[0] - start[0])
        nearest_y = start[1] + t * (end[1] - start[1])
        
        # Distance to line
        distance = math.sqrt(
            (opening_position[0] - nearest_x)**2 +
            (opening_position[1] - nearest_y)**2
        )
        
        if distance < threshold:
            return i
    
    return None
