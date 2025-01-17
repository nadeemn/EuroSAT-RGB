import os
import argparse
from sklearn.model_selection import train_test_split

def get_image_path(class_folder, dataset_root):
    image_paths = []

    for root, dirs, files in os.walk(class_folder):
        for file in files:
            if file.endswith(('.jpg', 'jpeg', '.png', '.tif')):
                image_paths.append(os.path.relpath(os.path.join(root, file), start= dataset_root))

    return image_paths

def traintest_split(dataset_root, root, class_names):
    all_train_paths = []
    all_val_paths = []
    all_test_paths = []

    train_size = 2700
    val_size = 1000
    test_size = 2000

    for class_name in class_names:
        class_folder = os.path.join(root, class_name)
        images_paths = get_image_path(class_folder, dataset_root)

        train_paths, temp_paths = train_test_split(images_paths, test_size = (1 - 0.47),
                                                    random_state = 42)

        val_paths, test_paths = train_test_split(temp_paths, test_size = 0.53, random_state = 42)

        all_train_paths.extend(train_paths)
        all_val_paths.extend(val_paths)
        all_test_paths.extend(test_paths)

    return all_train_paths, all_val_paths, all_test_paths

def save_splits_to_files(split_paths, split_name, output_dir):
    with open(os.path.join(output_dir, f"{split_name}.txt"), 'w') as f:
        for path in split_paths:
            f.write(f"{path}\n")


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Train Test Split")
    parser.add_argument('--root_dir', type=str, required=True, help="""Root Directory of the dataset. 
                        (No need to give the entire directory. Only parent directory is enough.)
                        For e.g. if the file path is: D:\EuroSAT_MS\EuroSAT_MS\AnnualCrop\AnnualCrop_1.tif.
                        Give the root as D:\EuroSAT_MS
                        """ )
    args = parser.parse_args()
    dataset_root = args.root_dir

    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    output_dir = os.path.join(project_root, 'splits')

    subdir = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    root = os.path.join(dataset_root, subdir[0])

    class_names = [class_name for class_name in os.listdir(root)
                    if os.path.isdir(os.path.join(root, class_name))]
    
    train_paths, val_paths, test_paths = traintest_split(dataset_root, root, class_names)

    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    assert train_set.isdisjoint(val_set), "Train and Validation sets overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test sets overlap!"
    assert val_set.isdisjoint(test_set), "Validation and Test sets overlap!"

    os.makedirs(output_dir, exist_ok = True)

    save_splits_to_files(train_paths, 'train', output_dir)
    save_splits_to_files(val_paths, 'val', output_dir)
    save_splits_to_files(test_paths, 'test', output_dir)

    print("Splits are disjoint and saved successfully.")
