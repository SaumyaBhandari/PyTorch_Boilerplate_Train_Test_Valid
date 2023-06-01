import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, ToTensor, RandomRotation, RandomAffine, Resize, Normalize
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        with open(csv_file, 'r') as f:
            for row in f:
                file_path, label = row.split(',')
                label = label.strip()  
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)
                label_idx = self.class_to_idx[label]
                self.data.append((file_path, label_idx))

    def __getitem__(self, index):
        file_path, label = self.data[index]
        try:
            image = Image.open(file_path).convert('RGB')
        except:
            return None, label
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)



def get_dataloader(root_dir, batch_size):

    train_transform = transforms.Compose([
        RandomRotation(degrees=45), 
        RandomAffine(degrees=0, shear=10),  # Random skewness and shear up to 10 degrees
        Resize((224, 224)),  # Resize to 224x224
        ToTensor(),  # Convert to tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(os.path.join(root_dir, "train.csv"), transform=train_transform)
    val_dataset = CustomDataset(os.path.join(root_dir, "valid.csv"), transform=val_transform)  # Resize validation images to 224x224
    test_dataset = CustomDataset(os.path.join(root_dir, "test.csv"), transform=val_transform)  # Resize validation images to 224x224

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


#Use the dataloader class like:
'''
train_data, val_data, test_data = get_dataloader("Dataset/Plant_Village/", batch_size=4) 
'''
