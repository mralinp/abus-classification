import numpy as np

def find_farthest(points):
    p1 = (0,0)
    p2 = (0,0)
    dm = 0

    for y1,x1 in points:
        for y2,x2 in points:
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if d > dm:
                dm = d
                p1 = (x1,y1)
                p2 = (x2,y2)
    
    return p1, p2

def calculate_slope(points):
    p1, p2 = points
    x1, y1 = p1
    x2, y2 = p2
    return (y2-y1)/(x2-x1)