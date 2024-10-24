import os
from PIL import Image  # Use PIL for both reading and saving images

def change_tif_to_png(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            # Full path to the image file
            img_path = os.path.join(folder_path, filename)
            
            # Open the .tif image using PIL
            with Image.open(img_path) as img:
                # Convert the filename to have the new .png extension
                new_filename = os.path.splitext(filename)[0] + ".png"
                new_img_path = os.path.join(folder_path, new_filename)
                
                # Save the image with the new .png extension using PIL
                img.save(new_img_path, format="PNG")
                print(f"Converted {filename} to {new_filename}")
    
    print("All .tif files converted to .png.")

# Example usage
# folder_path = "/path/to/your/folder"  # Change this to your folder path
# change_tif_to_png(folder_path)


# Example usage
folder_path = "/scratch/m23csa015/DIBCOSETS/2018/imgs"  # Change this to your folder path
change_tif_to_png(folder_path)
