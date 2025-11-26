import os
import cv2
from pathlib import Path

def crop_and_create_gt(images_dir, gt_dir, output_dir, output_gt):
    """Crop images based on bounding boxes and create line-level gt.txt"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_lines = 0
    processed_files = 0
    
    with open(output_gt, 'w', encoding='utf-8') as f_out:
        gt_files = sorted(Path(gt_dir).glob('*.txt'))
        
        print(f"Found {len(gt_files)} GT files to process...")
        
        for gt_file in gt_files:
            image_name = gt_file.stem + '.jpg'
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image - {image_path}")
                continue
            
            # Read GT file with encoding handling
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            lines = []
            for enc in encodings:
                try:
                    with open(gt_file, 'r', encoding=enc, errors='ignore') as f_in:
                        lines = f_in.readlines()
                    break
                except:
                    continue
            
            if not lines:
                print(f"Warning: Could not read GT file - {gt_file}")
                continue
            
            line_count = 0
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 9:
                    continue
                
                # Extract coordinates
                try:
                    coords = [int(float(x)) for x in parts[:8]]
                except ValueError:
                    continue
                    
                x_coords = coords[::2]  # x1, x2, x3, x4
                y_coords = coords[1::2]  # y1, y2, y3, y4
                
                # Get bounding box
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(img.shape[1], max(x_coords))
                y_max = min(img.shape[0], max(y_coords))
                
                # Extract text (everything after 8 coordinates)
                text = ','.join(parts[8:])
                # Clean text
                text = text.encode('utf-8', errors='ignore').decode('utf-8').strip()
                
                if not text or len(text) == 0:
                    continue
                
                # Skip if invalid coordinates
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Crop image
                cropped = img[y_min:y_max, x_min:x_max]
                
                if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
                    continue
                
                # Save cropped image
                crop_filename = f"{gt_file.stem}_line{idx:03d}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                
                try:
                    cv2.imwrite(crop_path, cropped)
                except:
                    print(f"Warning: Could not save {crop_path}")
                    continue
                
                # Write to gt.txt (relative path)
                relative_path = crop_filename  # Just filename, not full path
                f_out.write(f"{relative_path}\t{text}\n")
                
                line_count += 1
                total_lines += 1
            
            processed_files += 1
            if processed_files % 50 == 0:
                print(f"Processed {processed_files}/{len(gt_files)} files, {total_lines} lines extracted")
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"✓ Processed {processed_files} receipt images")
    print(f"✓ Extracted {total_lines} text lines")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Created: {output_gt}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    # Process training data
    print("Processing TRAINING data...")
    crop_and_create_gt(
        images_dir='data/images',
        gt_dir='data/gt',
        output_dir='data/images_cropped',
        output_gt='data/gt_cropped.txt'
    )
    
    # Process validation data
    print("\nProcessing VALIDATION data...")
    crop_and_create_gt(
        images_dir='val_data/images',
        gt_dir='val_data/gt',
        output_dir='val_data/images_cropped',
        output_gt='val_data/gt_cropped.txt'
    )