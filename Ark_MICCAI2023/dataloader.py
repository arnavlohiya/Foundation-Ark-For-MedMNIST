import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
# MEDMNIST MODIFICATION: Import official MedMNIST library for proper data loading
import medmnist
from medmnist import INFO
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, RandomResizedCrop, Normalize
)
from albumentations.pytorch import ToTensorV2

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    # elif mode == "embedding":
    #   transformations_list.append(transforms.Resize((crop_size, crop_size)))
    #   transformations_list.append(transforms.ToTensor())
    #   if normalize is not None:
    #     transformations_list.append(normalize)

    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_ts_transformations():
    AUGMENTATIONS = Compose([
      RandomResizedCrop(height=224, width=224),
      ShiftScaleRotate(rotate_limit=10),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
    ])
    return AUGMENTATIONS


class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
 
    self.augment = augment
    self.train_augment = build_ts_transformations()
    

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((224,224))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class CheXpert(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class VinDrCXR(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()

    with open(file_path, "r") as fr:
      line = fr.readline().strip()
      while line:
        lineItems = line.split()
        imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
        imageLabel = lineItems[1:]
        imageLabel = [int(i) for i in imageLabel]
        self.img_list.append(imagePath)
        self.img_label.append(imageLabel)
        line = fr.readline()

    if annotation_percent < 100:
      indexes = np.arange(len(self.img_list))
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageLabel = torch.FloatTensor(self.img_label[index])
    imageData = Image.open(imagePath).convert('RGB').resize((224,224))

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          imageLabel = np.zeros(3)
          imageLabel[int(lineItems[-1])] = 1
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class MIMIC(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index): 

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])     

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


# ===================================================================
# MEDMNIST DATASET CLASSES - ADDED FOR MEDMNIST INTEGRATION
# These classes handle NPZ-based datasets with different task types
# ===================================================================

class BaseMedMNIST(Dataset):
    """
    Base class for all MedMNIST datasets using official MedMNIST library
    
    This class provides proper train/val/test splits and handles different task types
    automatically using the official MedMNIST library.
    
    Args:
        dataset_name: Name of MedMNIST dataset (e.g., 'PathMNIST', 'ChestMNIST')
        images_path: Path to data directory (for compatibility with ARK interface)
        file_path: File path containing split information (train_list, val_list, test_list)
        augment: Augmentation transforms (from torchvision)
        size: Image size, default 224 to match ARK's expected input
        download: Whether to download dataset if not found
    """
    def __init__(self, dataset_name, images_path, file_path, augment, size=224, download=True):
        # Determine split from file_path
        if 'train' in file_path.lower():
            split = 'train'
        elif 'val' in file_path.lower():
            split = 'val'
        elif 'test' in file_path.lower():
            split = 'test'
        else:
            # Default to train if ambiguous
            split = 'train'
            
        # Get dataset info from MedMNIST INFO
        dataset_key = dataset_name.lower()
        if dataset_key not in INFO:
            raise ValueError(f"Dataset {dataset_name} not found in MedMNIST INFO")
            
        self.dataset_info = INFO[dataset_key]
        self.task_type = self.dataset_info['task']
        self.num_classes = len(self.dataset_info['label'])
        
        # Get the dataset class from medmnist module
        try:
            DataClass = getattr(medmnist, dataset_name)
        except AttributeError:
            raise ValueError(f"Dataset class {dataset_name} not found in medmnist module")
        
        # Load dataset with proper split
        self.dataset = DataClass(
            split=split,
            transform=None,  # We'll handle transforms in __getitem__
            download=download,
            root=images_path,  # Use your scratch directory
            size=size  # Use 224x224 to match ARK expectations
        )
        
        # Extract data for easier access
        self.images = self.dataset.imgs
        self.labels = self.dataset.labels
        
        # Store augmentation settings
        self.augment = augment
        self.train_augment = build_ts_transformations()
        
        print(f"✅ Loaded {dataset_name} {split} split: {len(self.images)} samples")
        print(f"   Task: {self.task_type}, Classes: {self.num_classes}")
        print(f"   Image shape: {self.images.shape if len(self.images) > 0 else 'N/A'}")
        print(f"   Label shape: {self.labels.shape if len(self.labels) > 0 else 'N/A'}")
    
    def __getitem__(self, index):
        # Get image and label from MedMNIST dataset
        image = self.images[index]  # Already 224x224x3 from MedMNIST with size=224
        label = self.labels[index]
        
        # Handle different label formats based on task type (auto-detected from INFO)
        if self.task_type == 'multi-label':
            # Multi-label: labels are already binary vectors [0,1,0,1,...]
            imageLabel = torch.FloatTensor(label.astype(np.float32))
        elif self.task_type in ['multi-class', 'ordinal']:
            # Multi-class/Ordinal: convert class index to one-hot encoding
            imageLabel = torch.zeros(self.num_classes)
            if isinstance(label, (list, np.ndarray)) and len(label) > 0:
                imageLabel[int(label[0])] = 1.0
            else:
                imageLabel[int(label)] = 1.0
        elif self.task_type == 'binary-class':
            # Binary: already handled by multi-class logic above
            imageLabel = torch.zeros(self.num_classes)
            if isinstance(label, (list, np.ndarray)) and len(label) > 0:
                imageLabel[int(label[0])] = 1.0
            else:
                imageLabel[int(label)] = 1.0
        else:
            # Default: treat as multi-class
            imageLabel = torch.zeros(self.num_classes)
            if isinstance(label, (list, np.ndarray)) and len(label) > 0:
                imageLabel[int(label[0])] = 1.0
            else:
                imageLabel[int(label)] = 1.0
        
        # Ensure image is in correct format (should already be 224x224x3)
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:  # Grayscale with channel dim
            image = np.repeat(image, 3, axis=-1)
        
        # Apply augmentations
        if self.augment is not None:
            # Use torchvision transforms
            image_pil = Image.fromarray(image.astype('uint8'))
            student_img, teacher_img = self.augment(image_pil), self.augment(image_pil)
        else:
            # Use albumentations transforms (same as existing datasets)
            image = image.astype('uint8')
            augmented = self.train_augment(image=image, mask=image)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            
            # Normalize (same as existing datasets)
            student_img = np.array(student_img) / 255.0
            teacher_img = np.array(teacher_img) / 255.0
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img - mean) / std
            teacher_img = (teacher_img - mean) / std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
        
        return student_img, teacher_img, imageLabel
    
    def __len__(self):
        return len(self.images)

# Individual MedMNIST Dataset Classes
class PathMNIST(BaseMedMNIST):
    """PathMNIST: 9-class pathology tissue classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('PathMNIST', images_path, file_path, augment)

class ChestMNIST(BaseMedMNIST):
    """ChestMNIST: 14-label chest X-ray multi-label classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('ChestMNIST', images_path, file_path, augment)

class DermaMNIST(BaseMedMNIST):
    """DermaMNIST: 7-class dermatology classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('DermaMNIST', images_path, file_path, augment)

class OCTMNIST(BaseMedMNIST):
    """OCTMNIST: 4-class OCT image classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('OCTMNIST', images_path, file_path, augment)

class PneumoniaMNIST(BaseMedMNIST):
    """PneumoniaMNIST: Binary pneumonia classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('PneumoniaMNIST', images_path, file_path, augment)

class RetinaMNIST(BaseMedMNIST):
    """RetinaMNIST: 5-level diabetic retinopathy ordinal regression"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('RetinaMNIST', images_path, file_path, augment)

class BreastMNIST(BaseMedMNIST):
    """BreastMNIST: Binary breast cancer classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('BreastMNIST', images_path, file_path, augment)

class BloodMNIST(BaseMedMNIST):
    """BloodMNIST: 8-class blood cell classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('BloodMNIST', images_path, file_path, augment)

class TissueMNIST(BaseMedMNIST):
    """TissueMNIST: 8-class kidney tissue classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('TissueMNIST', images_path, file_path, augment)

class OrganAMNIST(BaseMedMNIST):
    """OrganAMNIST: 11-class abdominal organ classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('OrganAMNIST', images_path, file_path, augment)

class OrganCMNIST(BaseMedMNIST):
    """OrganCMNIST: 11-class coronary organ classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('OrganCMNIST', images_path, file_path, augment)

class OrganSMNIST(BaseMedMNIST):
    """OrganSMNIST: 11-class sagittal organ classification"""
    def __init__(self, images_path, file_path, augment):
        super().__init__('OrganSMNIST', images_path, file_path, augment)

# ===================================================================
# DATASET LOADER DICTIONARY - MODIFIED FOR MEDMNIST-ONLY TRAINING
# Traditional datasets commented out, MedMNIST datasets active
# ===================================================================

dict_dataloarder = {
    # Traditional datasets - COMMENTED OUT for MedMNIST focus
    # Uncomment these when needed for hybrid training
    # "ChestXray14": ChestXray14,
    # "CheXpert": CheXpert,
    # "Shenzhen": ShenzhenCXR,
    # "VinDrCXR": VinDrCXR,
    # "RSNAPneumonia": RSNAPneumonia,
    # "MIMIC": MIMIC,
    
    # MedMNIST 2D Datasets - ACTIVE for ARK training
    "PathMNIST": PathMNIST,
    "ChestMNIST": ChestMNIST,
    "DermaMNIST": DermaMNIST,
    "OCTMNIST": OCTMNIST,
    "PneumoniaMNIST": PneumoniaMNIST,
    "RetinaMNIST": RetinaMNIST,
    "BreastMNIST": BreastMNIST,
    "BloodMNIST": BloodMNIST,
    "TissueMNIST": TissueMNIST,
    "OrganAMNIST": OrganAMNIST,
    "OrganCMNIST": OrganCMNIST,
    "OrganSMNIST": OrganSMNIST,
}
