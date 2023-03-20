import os
import shutil
import re

# Define the mapping of old folder names to new folder names
mapping = {
    "Thumbs Up_new": "thumbsUp",
    "Thumbs Down_new": "thumbsDown",
    "Right Swipe_new": "rightSwipe",
    "Left Swipe_new": "leftSwipe",
    "Stop Gesture_new": "stopGesture"
}

def reformat_folder(src_folder, dst_folder):
    for old_name, new_name in mapping.items():
        os.makedirs(os.path.join(dst_folder, new_name), exist_ok=True)

    for folder in os.listdir(src_folder):
        folder_path = os.path.join(src_folder, folder)
        if os.path.isdir(folder_path):
            for old_name, new_name in mapping.items():
                if old_name in folder:
                    # Extract the folder-specific part from the folder name
                    folder_part = re.search(r'(\d{2}_\d{2}_\d{2})', folder).group(1)

                    # Create a enw folder in the destination with the folder-specific part
                    new_folder_path = os.path.join(dst_folder, new_name, folder_part)
                    os.makedirs(new_folder_path, exist_ok=True)

                    for filename in os.listdir(folder_path):
                        number = re.findall(r'\d+', filename)[-1]
                        src_file = os.path.join(folder_path, filename)
                        dst_file = os.path.join(new_folder_path, f'{number}.png')
                        shutil.copyfile(src_file, dst_file)
                    break

root_folder = 'archive'
old_train_folder = os.path.join(root_folder, 'train')
old_val_folder = os.path.join(root_folder, 'val')
new_train_folder = os.path.join(root_folder, 'train_new')
new_val_folder = os.path.join(root_folder, 'validation')

os.makedirs(new_train_folder, exist_ok=True)
os.makedirs(new_val_folder, exist_ok=True)

reformat_folder(old_train_folder, new_train_folder)
reformat_folder(old_val_folder, new_val_folder)