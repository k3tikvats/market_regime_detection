# feature_normalization_validation.py
import pandas as pd
import numpy as np
from data_pipeline.feature_engineering.normalizer import Normalizer
import matplotlib.pyplot as plt

def validate_normalization_pipeline():
    """
    Validate feature normalization consistency between training and inference
    """
    print("Validating normalization pipeline...")
    
    # Generate mock feature data for training
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create example data with varied scales
    train_data = pd.DataFrame({
        f'feature_{i}': np.random.normal(scale=10**i, size=n_samples)
        for i in range(n_features)
    })
    
    # Create additional test data with similar distribution
    test_data = pd.DataFrame({
        f'feature_{i}': np.random.normal(scale=10**i, size=int(n_samples * 0.2))
        for i in range(n_features)
    })

    # Test each normalization method
    methods = ['standard', 'minmax', 'robust']
    
    fig, axes = plt.subplots(len(methods), 2, figsize=(15, 5 * len(methods)))
    
    for i, method in enumerate(methods):
        print(f"\nTesting {method.upper()} normalization")
        
        # Initialize normalizer with the current method
        normalizer = Normalizer(method=method)
        
        # Fit and transform the training data
        train_norm = normalizer.fit_transform(train_data)
        
        # Transform the test data using the same normalizer
        test_norm = normalizer.transform(test_data)
        
        # Verify statistics
        print(f"Training data stats after normalization:")
        print(f"Mean: {train_norm.mean().mean():.4f}")
        print(f"Std: {train_norm.std().mean():.4f}")
        print(f"Min: {train_norm.min().min():.4f}")
        print(f"Max: {train_norm.max().max():.4f}")
        
        print(f"Test data stats after normalization:")
        print(f"Mean: {test_norm.mean().mean():.4f}")
        print(f"Std: {test_norm.std().mean():.4f}")
        print(f"Min: {test_norm.min().min():.4f}")
        print(f"Max: {test_norm.max().max():.4f}")
        
        # Test inverse transform for reconstruction
        train_reconstructed = normalizer.inverse_transform(train_norm)
        reconstruction_error = np.mean(np.abs(train_data.values - train_reconstructed.values))
        print(f"Reconstruction error: {reconstruction_error:.4f}")
        
        # Plot the distribution of normalized features
        for j in range(2):
            ax = axes[i, j]
            if j == 0:
                # Plot distribution of normalized training data
                ax.hist(train_norm.iloc[:, 1].values, bins=30, alpha=0.5, label='Train')
                ax.hist(test_norm.iloc[:, 1].values, bins=30, alpha=0.5, label='Test')
                ax.set_title(f"{method.capitalize()} - Feature Distribution")
                ax.legend()
            else:
                # Plot original vs reconstructed values for a selected feature
                feature_idx = 3  # Choose a representative feature
                ax.scatter(train_data.iloc[:100, feature_idx], 
                          train_reconstructed.iloc[:100, feature_idx], alpha=0.5)
                ax.plot([train_data.iloc[:, feature_idx].min(), train_data.iloc[:, feature_idx].max()], 
                       [train_data.iloc[:, feature_idx].min(), train_data.iloc[:, feature_idx].max()], 
                       'r--')
                ax.set_title(f"{method.capitalize()} - Reconstruction")
                ax.set_xlabel("Original Values")
                ax.set_ylabel("Reconstructed Values")
    
    plt.tight_layout()
    plt.savefig('normalization_validation.png')
    print("Validation plots saved to 'normalization_validation.png'")
    
    # Test PCA dimensionality reduction
    print("\nTesting PCA dimensionality reduction")
    pca_components = 5
    pca_normalizer = Normalizer(method='standard', pca_components=pca_components)
    
    # Fit and transform with PCA
    train_pca = pca_normalizer.fit_transform(train_data)
    test_pca = pca_normalizer.transform(test_data)
    
    # Verify PCA reduced dimensions
    print(f"Original dimensions: {train_data.shape}")
    print(f"PCA reduced dimensions: {train_pca.shape}")
    
    # Check explained variance
    explained_variance = pca_normalizer.pca.explained_variance_ratio_
    print(f"Explained variance per component: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.4f}")
    
    # Reconstruct from PCA
    train_pca_reconstructed = pca_normalizer.inverse_transform(train_pca)
    pca_error = np.mean(np.abs(train_data.values - train_pca_reconstructed.values))
    print(f"PCA reconstruction error: {pca_error:.4f}")
    
    return {
        "normalizer": normalizer,
        "pca_normalizer": pca_normalizer,
        "train_data": train_data,
        "test_data": test_data
    }

if __name__ == "__main__":
    validate_normalization_pipeline()