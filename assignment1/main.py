import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os
import shutil
import random

# I read online that setting seeds helps with reproducible results
np.random.seed(42)
tf.random.set_seed(42)

class EmojiClassifier:
    def __init__(self, num_classes=None):
        
        self.input_shape = (64, 64, 1)  # 64x64 grayscale images
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None
        
    def build_model(self):
        """
        Building my CNN - optimized for small datasets
        """
        if self.num_classes is None:
            print(" Number of classes not set! Please set num_classes first.")
            return None
            
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First conv block - start small
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block - increase filters
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block - more filters
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth conv block - even more filters
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self):
        """
        Setting up the model for training
        """
        if self.model is None:
            print(" Build the model first!")
            return
            
        # Using Adam optimizer with custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(" Model compiled successfully!")


    def show_model_info(self):
        """
        Display model architecture info
        """
        if self.model is None:
            print("❌ No model built yet!")
            return
            
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        total_params = self.model.count_params()
        print(f"\n📊 Model Info:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Input Shape: {self.input_shape}")
        print(f"   Number of Classes: {self.num_classes}")
    
    def prepare_data(self, train_path, val_path):
        """
        Loading and preparing data with heavy augmentation for small datasets
        """
        # Heavy augmentation for training to increase dataset size
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,          # more rotation
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescale for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Small batch size for small dataset
        batch_size = 8
        
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
        
        # Store class names for later use
        self.class_names = list(train_generator.class_indices.keys())
        print(f" Detected classes: {self.class_names}")
        
        return train_generator, val_generator
    
    def train(self, train_data, val_data, epochs=100):
        """
        Training the model with callbacks for small datasets
        """
        if self.model is None:
            print(" Build and compile the model first!")
            return
        
        # Callbacks optimized for small datasets
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_emoji_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        
        
        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        print(" Training finished!")
        return self.history
    
    def plot_results(self):
        """
        Plot training history
        """
        if self.history is None:
            print("❌ No training history found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print best scores
        best_val_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"🏆 Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    
def check_dataset_structure(dataset_folder):
    """
    Check and display dataset structure
    """
    print(f"🔍 Checking dataset structure in '{dataset_folder}'...")
    
    if not os.path.exists(dataset_folder):
        print(f"❌ Dataset folder '{dataset_folder}' not found!")
        print("Please create your dataset folder with the following structure:")
        
        return False, None, None
    
    # Get all class folders
    class_folders = []
    for item in os.listdir(dataset_folder):
        item_path = os.path.join(dataset_folder, item)
        if os.path.isdir(item_path):
            class_folders.append(item)
    
    if not class_folders:
        print(" No class folders found in dataset!")
        
        return False, None, None
    
    class_folders.sort()  # Sort for consistent ordering
    
    print(f" Found {len(class_folders)} emoji classes:")
    
    total_images = 0
    class_image_counts = {}
    
    for class_name in class_folders:
        class_path = os.path.join(dataset_folder, class_name)
        
        # Count image files
        image_files = []
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(file)
        
        class_image_counts[class_name] = len(image_files)
        total_images += len(image_files)
        
        print(f"  {class_name}: {len(image_files)} images")
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total Classes: {len(class_folders)}")
    print(f"   Total Images: {total_images}")

    
    return True, len(class_folders), class_folders



def split_dataset(original_folder, output_folder='training_data'):
    """
    Split original dataset into train/validation folders
    80% train, 20% validation
    """
    print(f"\n Splitting dataset from '{original_folder}' to '{output_folder}'...")
    
    # Create output directories
    train_dir = os.path.join(output_folder, 'train')
    val_dir = os.path.join(output_folder, 'val')
    
    for directory in [train_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process each class
    for class_name in os.listdir(original_folder):
        class_path = os.path.join(original_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Get all images
        images = []
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(file)
        
        if not images:
            print(f"⚠️  No images found in {class_name}")
            continue
        
        # Shuffle for random split
        random.shuffle(images)
        
        # Calculate split point
        total = len(images)
        train_count = int(total * 0.8)
        
        train_images = images[:train_count]
        val_images = images[train_count:]
        
        print(f"   {class_name}: {len(train_images)} train, {len(val_images)} val")
        
        # Create class directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy files
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)
    
    print(f"✅ Dataset split complete!")
    return output_folder

def predict_single_image(model_path, image_path, class_names):
    """
    Predict a single image
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = keras.preprocessing.image.load_img(
        image_path, 
        target_size=(64, 64),
        color_mode='grayscale'
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, prediction[0]

def test_user_images(model_path='emoji_classifier_final.h5', test_folder='assignment1\test_images', 
                    class_names_file='class_names.txt'):
    
    print(" Testing User Images")
    
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        print("Please train the model first.")
        return
    
    # Check if test folder exists and has images
    if not os.path.exists(test_folder):
        print(f"❌ Test folder '{test_folder}' not found!")
        print("Please create the test folder and add some images.")
        return
    
    # Get test images
    test_images = []
    for file in os.listdir(test_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            test_images.append(file)
    
    if not test_images:
        print(f"❌ No images found in '{test_folder}'!")
        print("Please add some images to test.")
        return
    
    # Load class names
    try:
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"❌ Class names file '{class_names_file}' not found!")
        return
    
    print(f"Available classes: {class_names}")
    print(f"Testing {len(test_images)} images...\n")
    
    # Test each image
    results = []
    for img_file in test_images:
        img_path = os.path.join(test_folder, img_file)
        try:
            predicted_class, confidence, all_predictions = predict_single_image(
                model_path, img_path, class_names
            )
            
            print(f" Prediction: {predicted_class}")
            print(f" Confidence: {confidence:.2%}")
            
            # Show top 3 predictions
            top_3_indices = np.argsort(all_predictions)[-3:][::-1]
            print(" Top 3 predictions are :")
            for i, idx in enumerate(top_3_indices, 1):
                print(f"{i}. {class_names[idx]}: {all_predictions[idx]:.2%}")
            print()
            
            results.append((img_file, predicted_class, confidence))
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Summary
    if results:
        print("Test Summary:")
        
        # Group by predicted class
        class_counts = {}
        for _, pred_class, _ in results:
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        print("   Prediction Distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"{class_name}: {count} images")

def train_model():
    """
    Main training function
    """
    print("Hand-Drawn Emoji Classifier - Training Mode")
    # Step 1: Get dataset path
    dataset_path = 'assignment1\dataset'
    
    # Step 2: Check dataset
    is_valid, num_classes, class_names = check_dataset_structure(dataset_path)
    if not is_valid:
        return
    
    # Step 3: Split dataset
    print(f"\n📁Splitting dataset...")
    split_folder = split_dataset(dataset_path)
    
    # Step 4: Create model
    print(f"\nCreating model for {num_classes} classes...")
    classifier = EmojiClassifier(num_classes=num_classes)
    classifier.build_model()
    classifier.compile_model()
    classifier.show_model_info()
    
    # Step 5: Prepare data
    print(f"\nLoading training data...")
    train_gen, val_gen = classifier.prepare_data(
        os.path.join(split_folder, 'train'),
        os.path.join(split_folder, 'val')
    )
    
    # Step 6: Train
    print(f"\n Starting training...")
    classifier.train(train_gen, val_gen)
    
    # Step 7: Show results
    print(f"\n📈 Training Results:")
    classifier.plot_results()
    
    # Step 8: Save model and class names
    final_model_path = 'emoji_classifier_final.h5'
    classifier.model.save(final_model_path)
    
    # Save class names for testing
    with open('class_names.txt', 'w') as f:
        for class_name in classifier.class_names:
            f.write(f"{class_name}\n")
    
    print(f"\nTraining Complete!")
    print(f"  Model saved: {final_model_path}")
    print(f" Class names saved: class_names.txt")
    print(f"  Ready for testing with user images!")

def main():
    """
    Interactive main function
    """
    while True:
        print("\nWhat would you like to do?")
        print("1. Train a new model")
        print("2.Test with user images")
        
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            test_user_images()
        else:
            print("❌ Invalid choice. Please enter 1-2.")
            break

if __name__ == "__main__":  # Fixed this line!
    main()
