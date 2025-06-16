import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.cuda.amp as amp
import random
import matplotlib.pyplot as plt

class Emotion6Dataset(Dataset):
    """Dataset loader for Emotion6 dataset"""
    
    def __init__(self, root_dir, gt_file, transform=None, train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        
        # Read probability labels
        self.samples = []
        with open(gt_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                img_path = os.path.join(root_dir, parts[0])
                probs = [float(p) for p in parts[3:9]]  # Get probabilities for 6 emotions
                self.samples.append((img_path, probs))
    
    def __len__(self):
        return len(self.samples)
    
    def augment_image(self, image):
        # Create two versions of the image: original and rotated 30 degrees
        augmented_images = [
            image,
            image.rotate(30)
        ]
        return augmented_images
    
    def __getitem__(self, idx):
        img_path, probs = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.train:
            images = self.augment_image(image)
            processed_images = []
            processed_labels = []
            for img in images:
                # Resize to 512x512
                img = img.resize((512, 512), Image.BILINEAR)
                if self.transform:
                    img = self.transform(img)
                processed_images.append(img)
                processed_labels.append(torch.tensor(probs, dtype=torch.float32))
            # For training set, return augmented images and corresponding probability labels
            # Convert lists to tensors
            processed_images = torch.stack(processed_images)
            processed_labels = torch.stack(processed_labels)
            return processed_images, processed_labels
        else:
            # Modified processing for validation set
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
            if self.transform:
                image = self.transform(image)
            # Return tensor with correct dimensions, no extra dimensions needed
            return image, torch.tensor(probs, dtype=torch.float32)


class EmotionAnalyzer:
    """
    Emotion Analyzer for facial emotion recognition using deep learning models.
    Performs training, evaluation, and inference for emotion classification.
    """
    
    def __init__(self, device=None, pretrained=True):
        """
        Initialize the EmotionAnalyzer.
        
        Args:
            device: Computing device (CPU or GPU)
            pretrained: Whether to use pretrained weights
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        self.pretrained = pretrained
        self.model = self._create_model()
        self.model.eval()
        self.transform = self._get_transform()
    
    def _create_model(self):
        """
        Create a model based on pretrained ResNet50
        
        Returns:
            Initialized model for emotion recognition
        """
        model = models.resnet50(pretrained=self.pretrained)
        # Freeze all pretrained layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Modify the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, len(self.emotions)),
        )
        
        # Only train newly added layers
        for param in model.fc.parameters():
            param.requires_grad = True
            
        model = model.to(self.device)
        return model
    
    def _get_transform(self):
        """
        Create image transformation pipeline
        
        Returns:
            Composition of image transforms
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _prepare_datasets(self, data_dir, gt_file, batch_size):
        """
        Prepare training and validation datasets
        
        Args:
            data_dir: Directory containing image data
            gt_file: Ground truth file with emotion labels
            batch_size: Batch size for data loaders
            
        Returns:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        dataset = Emotion6Dataset(data_dir, gt_file, transform=None)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataset.dataset.transform = self.transform
        train_dataset.dataset.train = True
        val_dataset.dataset.transform = self.transform
        val_dataset.dataset.train = False
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def _calculate_loss(self, outputs, labels):
        """
        Calculate combined loss (KL divergence and cross entropy)
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            
        Returns:
            Combined loss, KL loss component, CE loss component
        """
        log_probs = F.log_softmax(outputs, dim=1)
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        ce_criterion = nn.CrossEntropyLoss()
        
        # KL divergence loss (distribution)
        kl_loss = kl_criterion(log_probs, labels)
        
        # Cross entropy loss (class labels)
        true_classes = torch.argmax(labels, dim=1)
        ce_loss = ce_criterion(outputs, true_classes)
        
        # Combined loss (with adjustable weights)
        combined_loss = 0.7 * kl_loss + 0.3 * ce_loss
        
        return combined_loss, kl_loss, ce_loss
    
    def train(self, data_dir, gt_file, epochs=100, batch_size=32, lr=0.001):
        """
        Train the emotion recognition model
        
        Args:
            data_dir: Directory containing image data
            gt_file: Ground truth file with emotion labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            metrics: Dictionary containing training and validation metrics
            val_loader: Validation data loader
        """
        # Prepare datasets
        train_loader, val_loader = self._prepare_datasets(data_dir, gt_file, batch_size)
        
        # Configure optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        scaler = amp.GradScaler()
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 12
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(
                train_loader, optimizer, scheduler, scaler
            )
            
            # Validation phase
            self.model.eval()
            val_metrics = self._validate_epoch(val_loader)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save model if validation loss improved
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(
                    epoch, optimizer, train_metrics, val_metrics, 'models/best_model.pth'
                )
                print("Saved best model")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        checkpoint = torch.load('models/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return {
            'val_loss': checkpoint['val_loss'],
            'val_acc': checkpoint['val_acc'],
            'train_loss': checkpoint['train_loss'],
            'train_acc': checkpoint['train_acc']
        }, val_loader
    
    def _train_epoch(self, train_loader, optimizer, scheduler, scaler):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader with training data
            optimizer: Optimizer for parameter updates
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision training
            
        Returns:
            Dictionary containing training loss and accuracy
        """
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        for inputs_batch, labels_batch in tqdm(train_loader, desc='Training'):
            # inputs_batch shape: [batch_size, num_augmentations, channels, height, width]
            # labels_batch shape: [batch_size, num_augmentations, num_classes]
            
            batch_size = inputs_batch.size(0)
            num_augmentations = inputs_batch.size(1)
            
            # Reshape tensors to process all augmented images
            inputs = inputs_batch.view(-1, 3, 512, 512).to(self.device)
            labels = labels_batch.view(-1, len(self.emotions)).to(self.device)
            
            optimizer.zero_grad()
            with amp.autocast():
                outputs = self.model(inputs)
                loss, _, _ = self._calculate_loss(outputs, labels)
                
                # Calculate training accuracy
                log_probs = F.log_softmax(outputs, dim=1)
                pred_probs = torch.exp(log_probs)  # Convert to probabilities
                pred_classes = torch.argmax(pred_probs, dim=1)  # Predicted classes
                true_classes = torch.argmax(labels, dim=1)
                train_correct += (pred_classes == true_classes).sum().item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        return {
            'loss': train_loss / total_samples,
            'accuracy': train_correct / total_samples
        }
    
    def _validate_epoch(self, val_loader):
        """
        Validate model on validation set
        
        Args:
            val_loader: DataLoader with validation data
            
        Returns:
            Dictionary containing validation loss and accuracy
        """
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss, _, _ = self._calculate_loss(outputs, labels)
                
                # Calculate validation accuracy
                log_probs = F.log_softmax(outputs, dim=1)
                pred_probs = torch.exp(log_probs)
                pred_classes = torch.argmax(pred_probs, dim=1)
                true_classes = torch.argmax(labels, dim=1)
                val_correct += (pred_classes == true_classes).sum().item()
                
                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)
        
        return {
            'loss': val_loss / val_total,
            'accuracy': val_correct / val_total
        }
    
    def _save_checkpoint(self, epoch, optimizer, train_metrics, val_metrics, filepath):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            optimizer: Optimizer state
            train_metrics: Training metrics
            val_metrics: Validation metrics
            filepath: Path to save checkpoint
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        }, filepath)
    
    def test(self, test_loader):
        """
        Evaluate model on test dataset
        
        Args:
            test_loader: DataLoader with test data
            
        Returns:
            test_loss: Loss on test data
            test_acc: Accuracy on test data
            predictions: List of prediction tuples (predicted, ground truth)
        """
        self.model.eval()
        criterion = nn.KLDivLoss(reduction='batchmean')
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        predictions = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, labels)
                
                # Calculate test accuracy
                pred_probs = torch.exp(log_probs)
                pred_classes = torch.argmax(pred_probs, dim=1)
                true_classes = torch.argmax(labels, dim=1)
                test_correct += (pred_classes == true_classes).sum().item()
                
                test_loss += loss.item() * inputs.size(0)
                test_total += inputs.size(0)
                
                probs = pred_probs.cpu()
                labels = labels.cpu()
                for prob, label in zip(probs, labels):
                    predictions.append((prob, label))
        
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        return test_loss, test_acc, predictions
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to model weights file
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_path):
        """
        Perform emotion prediction on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            predicted_emotion: Predicted emotion category
            emotion_probs: Dictionary of probabilities for each emotion
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Perform prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
            
            # Get prediction results
            emotion_probs = {emotion: prob.item() for emotion, prob in zip(self.emotions, probabilities)}
            predicted_emotion = self.emotions[predicted_idx]
            
            return predicted_emotion, emotion_probs
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def inference_demo():
    """
    Model inference demonstration
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer()
    
    # Load trained model
    model_path = "models/best_model.pth"
    analyzer.load_model(model_path)
    
    # Test image paths
    test_images = [
        "test_images/test1.jpg",
        "test_images/test2.jpg",
        "test_images/test3.jpg",
        "test_images/test4.jpg",
        "test_images/test5.jpg"
    ]
    
    # Directory to save output visualizations
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a figure for all the subplots
    num_images = len(test_images)
    fig, axes = plt.subplots(2, num_images, figsize=(18, 10), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle("Emotion Prediction Results", fontsize=16)

    # Ensure axes is iterable (for cases with one image)
    if num_images == 1:
        axes = [[axes[0]], [axes[1]]]
    
    # Predict each image and create subplots
    for i, image_path in enumerate(test_images):
        print(f"\nProcessing image: {image_path}")
        
        # Perform prediction
        predicted_emotion, emotion_probs = analyzer.predict(image_path)
        
        # Print results
        print(f"Predicted emotion: {predicted_emotion}")
        print("Emotion probabilities:")
        for emotion, prob in emotion_probs.items():
            print(f"{emotion}: {prob:.4f}")
        
        # Load and display the image in the first row
        img = mpimg.imread(image_path)  # Read the image file
        axes[0][i].imshow(img)
        axes[0][i].axis('off')  # Turn off axis for the image
        axes[0][i].set_title(f"Image {i+1}", fontsize=10)
        
        # Visualize the prediction probabilities in the second row
        emotions = list(emotion_probs.keys())
        probabilities = list(emotion_probs.values())

        axes[1][i].bar(emotions, probabilities, color='skyblue', alpha=0.8)
        axes[1][i].set_title(f"Predicted: {predicted_emotion}", fontsize=10)
        axes[1][i].set_xlabel("Emotions", fontsize=8)
        if i == 0:
            axes[1][i].set_ylabel("Probability", fontsize=8)
        axes[1][i].set_ylim(0, 1)
        axes[1][i].tick_params(axis='x', rotation=45, labelsize=8)
    
    # Adjust layout and save the final figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    output_file = os.path.join(output_dir, "all_results_with_images.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved combined visualization with images to {output_file}")

def main():
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    
    # Configuration
    data_dir = "./Emotion6/images"
    gt_file = "./Emotion6/ground_truth.txt"
    model_save_dir = "models"
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    analyzer = EmotionAnalyzer(pretrained=True)
    
    print("\nStarting training...")
    metrics, val_loader = analyzer.train(
        data_dir=data_dir,
        gt_file=gt_file,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate
    )
    
    print("\nTraining completed. Final metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test model
    print("\nTesting model...")
    test_loss, test_acc, predictions = analyzer.test(val_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Other tests
    print("\nTest on sample data...")
    inference_demo()

if __name__ == "__main__":
    main()