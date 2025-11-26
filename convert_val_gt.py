import os
from pathlib import Path

def convert_gt_format(images_dir, gt_dir, output_file):
    """
    Convert individual GT files with bounding boxes to single gt.txt
    
    Args:
        images_dir: Path to images folder
        gt_dir: Path to gt folder with individual .txt files
        output_file: Output gt.txt path
    """
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Get all text files from gt directory
        gt_files = sorted(Path(gt_dir).glob('*.txt'))
        
        if len(gt_files) == 0:
            print(f"Warning: No .txt files found in {gt_dir}")
            return
        
        processed = 0
        skipped = 0
        
        for gt_file in gt_files:
            # Get corresponding image name
            image_name = gt_file.stem + '.jpg'  # Change to .png if needed
            image_path = os.path.join('images', image_name)
            
            # Check if image exists
            full_image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(full_image_path):
                print(f"Warning: Image not found - {full_image_path}")
                skipped += 1
                continue
            
            # Read GT file with multiple encoding attempts
            lines = []
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(gt_file, 'r', encoding=encoding) as f_in:
                        lines = f_in.readlines()
                    break  # Success, exit loop
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error reading {gt_file}: {e}")
                    skipped += 1
                    break
            
            if not lines:
                print(f"Warning: Could not decode {gt_file} with any encoding")
                skipped += 1
                continue
            
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
                    # Clean any problematic characters
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
                    texts.append(text)
            
            if not texts:
                print(f"Warning: No valid text found in {gt_file}")
                skipped += 1
                continue
            
            # Concatenate all text lines with space
            full_text = ' '.join(texts)
            f_out.write(f"{image_path}\t{full_text}\n")
            processed += 1
        
        print(f"\nConversion complete!")
        print(f"Processed: {processed} files")
        print(f"Skipped: {skipped} files")
        print(f"Created: {output_file}")

# Usage - VALIDATION DATA
if __name__ == '__main__':
    images_dir = 'val_data/images'      # Validation images folder
    gt_dir = 'val_data/gt'              # Validation gt folder with .txt files
    output_file = 'val_data/gt.txt'     # Output file
    
    convert_gt_format(images_dir, gt_dir, output_file)