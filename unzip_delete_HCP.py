import os
import zipfile

def unzip_and_delete_zip(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".zip"):
            zip_path = os.path.join(folder_path, filename)
            
            # Skip empty files (likely partial downloads)
            if os.path.getsize(zip_path) < 100000:
                print(f"Skipping empty file: {zip_path}")
                continue

            # Check if it is a valid zip file
            if not zipfile.is_zipfile(zip_path):
                print(f"Skipping invalid zip: {zip_path}")
                continue
            
            print(f"Extracting: {zip_path}")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)

                # Delete the zip file after successful extraction
                os.remove(zip_path)
                print(f"Deleted: {zip_path}")

            except Exception as e:
                print(f"Error processing {zip_path}: {e}")

if __name__ == "__main__":
    folder = r"D:\HCP_new"
    unzip_and_delete_zip(folder)
