import os
import torch
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from pathlib import Path
import click
from torchvision import transforms
import pickle
import sys
import types
import warnings

def load_tf_model(model_path):
    """
    Load a TensorFlow-based StyleGAN2 model and prepare it for PyTorch usage.
    This function creates a compatibility layer for TensorFlow pickled models.
    """
    # Create a simple custom module to handle TF model structure
    class TFLibStub:
        pass

    class NetworkStub:
        def __init__(self):
            self.input_shape = [None, 3, None, None]
            self.output_shape = [None, 3, None, None]
            self.num_channels = 3
            self.resolution = 1024
            self.label_size = 0
            self.label_dim = 0
            self.z_dim = 512

    # Set up the stub module
    sys.modules['dnnlib'] = types.ModuleType('dnnlib')
    sys.modules['dnnlib.tflib'] = types.ModuleType('dnnlib.tflib')
    sys.modules['dnnlib.tflib'].Network = NetworkStub

    # Load the pickle file with the custom stub
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

class StyleGANArtProcessor:
    def __init__(self, model_path):
        """Initialize StyleGAN processor with TensorFlow model compatibility."""
        self.device = torch.device('cuda')
        print(f'Loading StyleGAN model from "{model_path}"...')
        
        try:
            # Load the TensorFlow model
            data = load_tf_model(model_path)
            
            # Extract the generator
            if isinstance(data, dict):
                if 'G_ema' in data:
                    self.G = data['G_ema']
                elif 'G' in data:
                    self.G = data['G']
                else:
                    self.G = data
            else:
                self.G = data
            
            # Set basic parameters if not available
            if not hasattr(self.G, 'z_dim'):
                self.G.z_dim = 512
            if not hasattr(self.G, 'img_resolution'):
                self.G.img_resolution = 1024
            
            print(f'Successfully loaded model with resolution {self.G.img_resolution}')
        except Exception as e:
            print(f'Error loading model: {str(e)}')
            raise
        
        # Initialize image processing transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.G.img_resolution, self.G.img_resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def load_and_preprocess_image(self, image_path):
        """Load and prepare an input image for processing."""
        image = PIL.Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Apply transforms with error handling
        try:
            image_tensor = self.transform(image)
            return image, image_tensor, original_size
        except Exception as e:
            print(f'Error preprocessing image: {str(e)}')
            raise

    def create_warm_direction(self, strength=1.0):
        """Create a latent direction for warmth adjustment."""
        warm = torch.zeros(1, self.G.z_dim, device=self.device)
        
        # Enhanced warmth values for better effect
        color_region = self.G.z_dim // 3
        warm[0, :color_region] = 0.5 * strength          # Reds
        warm[0, color_region:2*color_region] = 0.3 * strength    # Yellows
        warm[0, 2*color_region:] = -0.2 * strength       # Cool colors
        
        return warm

    def generate_variations(self, input_image, num_variations=5, warmth_range=(0.2, 1.2)):
        """Generate variations using the TensorFlow-compatible model."""
        variations = []
        warmth_values = np.linspace(warmth_range[0], warmth_range[1], num_variations)
        
        try:
            # Generate base image
            seed = np.random.randint(0, 100000)
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
            
            # Handle potential differences in model forward signatures
            try:
                base_img = self.G.forward(z, None)  # Try without label first
            except:
                try:
                    base_img = self.G.forward(z)  # Try without any additional arguments
                except Exception as e:
                    print(f'Error in model forward pass: {str(e)}')
                    raise
            
            base_img = (base_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            variations.append(("Original", base_img[0].cpu()))
            
            # Generate warm variations
            warm_direction = self.create_warm_direction()
            for idx, warmth in enumerate(warmth_values[1:], 1):
                modified_z = z + (warm_direction * warmth)
                try:
                    img = self.G.forward(modified_z, None)
                except:
                    img = self.G.forward(modified_z)
                
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = self.apply_color_warmth(img[0], warmth)
                variations.append((f'Warm {idx}', img.cpu()))
            
            return variations
            
        except Exception as e:
            print(f'Error generating variations: {str(e)}')
            raise

    def apply_color_warmth(self, img_tensor, intensity=1.0):
        """Apply color adjustments to enhance warm tones."""
        img = img_tensor.float() / 255.0 if img_tensor.dtype == torch.uint8 else img_tensor.float()
        
        # Enhanced color manipulation
        img[:, :, 0] *= (1 + 0.2 * intensity)  # Boost red
        img[:, :, 1] *= (1 + 0.1 * intensity)  # Boost green (for yellow tones)
        img[:, :, 2] *= (1 - 0.15 * intensity)  # Reduce blue
        
        return (img * 255).clamp(0, 255).to(torch.uint8)

    def create_comparison_plot(self, input_image, variations, output_path, title):
        """Create a visual comparison of all variations."""
        total_images = len(variations) + 1
        n_cols = min(5, total_images)
        n_rows = (total_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.suptitle(title, fontsize=16)
        
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        # Plot input image first
        axes[0][0].imshow(input_image)
        axes[0][0].axis('off')
        axes[0][0].set_title('Input Image')
        
        # Plot variations
        for idx, (label, img) in enumerate(variations):
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols
            axes[row][col].imshow(img.numpy())
            axes[row][col].axis('off')
            axes[row][col].set_title(label)
        
        # Hide empty subplots
        for idx in range(total_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def process_directory(processor, input_dir, output_dir, num_variations):
    """Process all images in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in input_path.glob('*.[jp][pn][gf]'):
        try:
            print(f'Processing {img_path.name}...')
            img_output_dir = output_path / img_path.stem
            img_output_dir.mkdir(exist_ok=True)
            
            input_image, image_tensor, original_size = processor.load_and_preprocess_image(img_path)
            variations = processor.generate_variations(image_tensor, num_variations)
            
            plot_path = img_output_dir / 'comparison.png'
            processor.create_comparison_plot(
                input_image, 
                variations, 
                plot_path, 
                f'Warm Variations - {img_path.stem}'
            )
            
            for label, img in variations:
                output_file = img_output_dir / f'{label.lower().replace(" ", "_")}.png'
                PIL.Image.fromarray(img.numpy()).save(output_file)
                
            print(f'Successfully processed {img_path.name}')
            
        except Exception as e:
            print(f'Error processing {img_path.name}: {str(e)}')
            continue

@click.command()
@click.option('--input-dir', help='Directory containing input images', type=str, required=True)
@click.option('--output-dir', help='Where to save the variations', type=str, required=True)
@click.option('--num-variations', type=int, default=5, help='Number of warm variations to generate')
def main(input_dir, output_dir, num_variations):
    """Generate warm variations of artwork using StyleGAN2."""
    processor = StyleGANArtProcessor('network-snapshot-012052.pkl')
    process_directory(processor, input_dir, output_dir, num_variations)

if __name__ == '__main__':
    main()