import re
import numpy as np

def geo_reader(filename):
    """
    Parse an Ensight .geo file to extract nodes and elements
    
    Returns:
        nodes: list of tuples (node_id, x, y, z)
        elements: list of tuples (element_id, node1, node2, ...)
    """
    nodes = []
    elements = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    i = 0
    num_lines = len(lines)
    
    # Skip header lines
    while i < num_lines and lines[i].strip() != "coordinates":
        i += 1
    
    # Parse nodes
    if i < num_lines and lines[i].strip() == "coordinates":
        i += 1
        
        # Read number of nodes
        num_nodes = int(lines[i].strip())
        i += 1
        
        # Read node data
        for _ in range(num_nodes):
            line = lines[i].strip()
            i += 1
            
            # Handle cases where numbers might be concatenated (like "2-5.50000E+00")
            # We need to insert spaces before negative signs that follow digits
            line = re.sub(r'(\d)(-)', r'\1 \2', line)
            
            parts = line.split()
            if len(parts) >= 4:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes.append((x, y, z))

    nodes = np.array(nodes)
    
    # Skip to elements section
    while i < num_lines and not (lines[i].strip().startswith("bar") or lines[i].strip().startswith("bar2")):
        i += 1
    
    # Parse elements
    if i < num_lines and (lines[i].strip() == "bar2" or lines[i].strip().startswith("bar")):
        i += 1
        
        # Read number of elements
        num_elements = int(lines[i].strip())
        i += 1
        
        # Read element data
        for _ in range(num_elements):
            line = lines[i].strip()
            i += 1
            
            parts = line.split()
            if len(parts) >= 3:
                # Extract connected nodes (bar2 elements have 2 nodes)
                node1 = int(parts[1])
                node2 = int(parts[2])
                elements.append((node1, node2))
                
    elements = np.array(list(set(map(tuple, map(sorted, elements)))))
    
    
    return nodes, elements
