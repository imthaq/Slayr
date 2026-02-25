import cv2
import numpy as np
import os

def create_glasses_asset():
    # Create a transparent image (BGRA)
    height, width = 150, 400
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Define glasses color (Black)
    color = (0, 0, 0, 255) # B, G, R, A
    thickness = 5

    # Draw Left Lens Frame (Circle)
    cv2.circle(img, (100, 75), 50, color, thickness)
    
    # Draw Right Lens Frame (Circle)
    cv2.circle(img, (300, 75), 50, color, thickness)
    
    # Draw Bridge
    cv2.line(img, (150, 75), (250, 75), color, thickness)
    
    # Ensure directory exists
    output_dir = r"c:\Users\AntiVenom\Desktop\slayr\static\assets"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "glasses.png")
    cv2.imwrite(output_path, img)
    print(f"Generated placeholder glasses at {output_path}")

if __name__ == "__main__":
    create_glasses_asset()
