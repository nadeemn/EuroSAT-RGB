import os
from sklearn.model_selection import train_test_split

dataset_root = r'D:\EuroSAT_RGB'

def get_image_path(class_folder):
    image_paths = []

    for root, dirs, files in os.walk(class_folder):
        print(dirs)
        for file in files:
            if file.endswith(('.jpg', 'jpeg', '.png')):
                image_paths.append(os.path.relpath(os.path.join(root, file), start= dataset_root))

    return image_paths

def traintest_split(dataset_root, class_names):
    all_train_paths = []
    all_val_paths = []
    all_test_paths = []

    train_size = 2700
    val_size = 1000
    test_size = 2000

    for class_name in class_names:
        class_folder = os.path.join(dataset_root, class_name)
        images_paths = get_image_path(class_folder)

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
    current_file = os.path.abspath(__file__)
    project_root_directory = os.path.dirname(current_file)

    root = os.path.join(dataset_root, 'EuroSAT_RGB')

    class_names = [class_name for class_name in os.listdir(root)
                    if os.path.isdir(os.path.join(root, class_name))]
    
    train_paths, val_paths, test_paths = traintest_split(root, class_names)

    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)

    assert train_set.isdisjoint(val_set), "Train and Validation sets overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test sets overlap!"
    assert val_set.isdisjoint(test_set), "Validation and Test sets overlap!"

    output_dir = './splits'
    os.makedirs(output_dir, exist_ok = True)

    save_splits_to_files(train_paths, 'train', output_dir)
    save_splits_to_files(val_paths, 'val', output_dir)
    save_splits_to_files(test_paths, 'test', output_dir)

    print("Splits are disjoint and saved successfully.")
