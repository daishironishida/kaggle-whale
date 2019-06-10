from torchvision.transforms import (
    CenterCrop, ColorJitter, Compose, Normalize, RandomRotation, 
    Resize, Pad, ToTensor)


def get_transforms(transform_type, image_size):
    train_transform = Compose([
        Resize((160, 320)),
        ColorJitter(),
        Pad(20, padding_mode='symmetric'),
        RandomRotation(10),
        CenterCrop((160, 320))])

    test_transform = Resize((160, 320))

    return train_transform, test_transform

tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
