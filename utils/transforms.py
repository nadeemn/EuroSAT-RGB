from torchvision import transforms

def get_transforms():
    transform_1 = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToTensor()
    ])
    }

    transform_2 = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9,1.1)),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Compose([
                transforms.ToTensor()
            ])
        ])
    } 

    return transform_1, transform_2