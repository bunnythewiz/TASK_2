import os
from pathlib import Path

def convert_gt_format(images_dir, gt_dir, output_file, mode='full_text'):
    """
    Convert individual GT files with bounding boxes to single gt.txt
    
    Args:
        images_dir: Path to images folder
        gt_dir: Path to gt folder with individual .txt files
        output_file: Output gt.txt path
        mode: 'full_text' (concatenate all lines) or 'per_line' (separate crops needed)
    """
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Get all text files from gt directory
        gt_files = sorted(Path(gt_dir).glob('*.txt'))
        
        for gt_file in gt_files:
            # Get corresponding image name
            image_name = gt_file.stem + '.jpg'  # Change extension if needed
            image_path = os.path.join('images', image_name)  # Relative path
            
            # Check if image exists
            full_image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(full_image_path):
                print(f"Warning: Image not found - {full_image_path}")
                continue
            
            # Read GT file
            with open(gt_file, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
            
            # Extract text from each line (after last 8 coordinates)
            texts = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma and get text (last element)
                parts = line.split(',')
                if len(parts) >= 9:  # 8 coords + text
                    text = ','.join(parts[8:])  # Handle text with commas
                    texts.append(text)
            
            if mode == 'full_text':
                # Concatenate all text lines with space
                full_text = ' '.join(texts)
                f_out.write(f"{image_path}\t{full_text}\n")
            
            elif mode == 'per_line':
                # Each line as separate entry (need image cropping)
                for idx, text in enumerate(texts):
                    # You'd need to crop images here based on coordinates
                    crop_name = f"{gt_file.stem}_line{idx}.jpg"
                    crop_path = os.path.join('images_cropped', crop_name)
                    f_out.write(f"{crop_path}\t{text}\n")
        
        print(f"Conversion complete! Created {output_file}")

# Usage
if __name__ == '__main__':
    images_dir = 'data/images'      # Your images folder
    gt_dir = 'data/gt'              # Your gt folder with .txt files
    output_file = 'data/gt.txt'     # Output file
    
    convert_gt_format(images_dir, gt_dir, output_file, mode='full_text')