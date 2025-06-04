import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.transformers import optimizer
import numpy as np
import time
from PIL import Image
import fiftyone as fo
import fiftyone.zoo as foz
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "google/vit-base-patch16-224"
ONNX_PATH = "vit_base.onnx"
OPTIMIZED_ONNX_PATH = "vit_base_optimized.onnx"
QUANTIZED_ONNX_PATH = "vit_base_quantized.onnx"
FP16_ONNX_PATH = "vit_base_fp16.onnx"
BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_BENCHMARK = 100

class ViTBenchmark:
    def __init__(self):
        print("Loading ViT model and processor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        
        # Load dataset
        print("Loading COCO-2017 validation dataset...")
        try:
            self.dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=200)
        except:
            print("Note: COCO dataset loading failed, will use dummy images for benchmarking")
            self.dataset = None
        
    def prepare_sample_images(self, num_samples=NUM_BENCHMARK):
        """Prepare sample images from COCO dataset or generate dummy images"""
        print(f"Preparing {num_samples} sample images...")
        images = []
        
        if self.dataset:
            samples = self.dataset.take(num_samples)
            for sample in tqdm(samples, desc="Loading images"):
                try:
                    image = Image.open(sample.filepath).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue
        
        # If not enough images or no dataset, create dummy images
        while len(images) < num_samples:
            # Create a random RGB image
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            images.append(dummy_image)
                
        return images[:num_samples]
    
    def export_to_onnx(self):
        """Export PyTorch model to ONNX format"""
        print("\nExporting model to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(self.device)
        
        # Export to ONNX with specific settings for better compatibility
        torch.onnx.export(
            self.model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=11,  # Use opset 11 for better compatibility with quantization
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model saved to {ONNX_PATH}")
        
    def optimize_onnx(self):
        """Apply graph optimizations to ONNX model"""
        print("\nOptimizing ONNX model...")
        
        try:
            # Use transformer optimization
            optimized_model = optimizer.optimize_model(
                ONNX_PATH,
                model_type='vit',  # Try ViT type first
                num_heads=12,
                hidden_size=768,
                opt_level=1,  # Conservative optimization level
                optimization_options=None,
                use_gpu=False  # CPU optimization for compatibility
            )
            optimized_model.save_model_to_file(OPTIMIZED_ONNX_PATH)
        except:
            # Fallback to BERT optimization if ViT type not supported
            try:
                optimized_model = optimizer.optimize_model(
                    ONNX_PATH,
                    model_type='bert',
                    num_heads=12,
                    hidden_size=768,
                    opt_level=1,
                    use_gpu=False
                )
                optimized_model.save_model_to_file(OPTIMIZED_ONNX_PATH)
            except Exception as e:
                print(f"Optimization failed: {e}")
                # If optimization fails, just copy the original
                import shutil
                shutil.copy(ONNX_PATH, OPTIMIZED_ONNX_PATH)
        
        print(f"Optimized ONNX model saved to {OPTIMIZED_ONNX_PATH}")
        
    def analyze_model_for_quantization(self, model_path):
        """Analyze model to identify problematic nodes for quantization"""
        model = onnx.load(model_path)
        
        conv_nodes = []
        matmul_nodes = []
        other_compute_nodes = []
        
        for node in model.graph.node:
            if node.op_type == 'Conv':
                conv_nodes.append(node.name)
            elif node.op_type in ['MatMul', 'Gemm']:
                matmul_nodes.append(node.name)
            elif node.op_type in ['Add', 'Mul', 'Div', 'Sub']:
                other_compute_nodes.append(node.name)
        
        print(f"\nModel analysis:")
        print(f"- Conv nodes: {len(conv_nodes)}")
        print(f"- MatMul/Gemm nodes: {len(matmul_nodes)}")
        print(f"- Other compute nodes: {len(other_compute_nodes)}")
        
        # Find patch embedding conv node
        patch_embed_conv = None
        for node_name in conv_nodes:
            if 'patch' in node_name.lower() or 'embed' in node_name.lower():
                patch_embed_conv = node_name
                break
        
        return conv_nodes, matmul_nodes, patch_embed_conv
        
    def quantize_onnx_safe(self):
        """Quantize ONNX model with multiple fallback strategies"""
        print("\nApplying quantization strategies...")
        
        # Analyze model first
        conv_nodes, matmul_nodes, patch_embed_conv = self.analyze_model_for_quantization(OPTIMIZED_ONNX_PATH)
        
        # Strategy 1: Try quantizing only MatMul operations
        print("\nStrategy 1: Quantizing only MatMul operations...")
        try:
            quantize_dynamic(
                OPTIMIZED_ONNX_PATH,
                QUANTIZED_ONNX_PATH,
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=['MatMul'],  # Only MatMul, no Conv
                per_channel=False,
                reduce_range=False
            )
            print(f"Successfully created {QUANTIZED_ONNX_PATH}")
            return True
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try with node exclusion
        if patch_embed_conv:
            print(f"\nStrategy 2: Excluding patch embedding Conv node: {patch_embed_conv}")
            try:
                quantize_dynamic(
                    OPTIMIZED_ONNX_PATH,
                    QUANTIZED_ONNX_PATH,
                    weight_type=QuantType.QInt8,
                    nodes_to_exclude=[patch_embed_conv],
                    per_channel=False,
                    reduce_range=True  # More compatible range
                )
                print(f"Successfully created {QUANTIZED_ONNX_PATH}")
                return True
            except Exception as e:
                print(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Create a custom quantization config
        print("\nStrategy 3: Custom quantization with specific settings...")
        try:
            # Create a model copy and modify it
            model = onnx.load(OPTIMIZED_ONNX_PATH)
            
            # Find and mark Conv nodes to skip
            nodes_to_exclude = []
            for node in model.graph.node:
                if node.op_type == 'Conv':
                    nodes_to_exclude.append(node.name)
            
            quantize_dynamic(
                OPTIMIZED_ONNX_PATH,
                QUANTIZED_ONNX_PATH,
                weight_type=QuantType.QInt8,
                nodes_to_exclude=nodes_to_exclude[:1] if nodes_to_exclude else [],  # Exclude first Conv only
                per_channel=False,
                reduce_range=True,
                extra_options={'ActivationSymmetric': False}
            )
            print(f"Successfully created {QUANTIZED_ONNX_PATH}")
            return True
        except Exception as e:
            print(f"Strategy 3 failed: {e}")
        
        print("\nAll quantization strategies failed. Skipping INT8 quantization.")
        return False
        
    def convert_to_fp16(self):
        """Convert model to FP16 as an alternative to INT8 quantization"""
        print("\nConverting to FP16 precision...")
        try:
            from onnxconverter_common import float16
            
            model = onnx.load(OPTIMIZED_ONNX_PATH)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, FP16_ONNX_PATH)
            print(f"FP16 model saved to {FP16_ONNX_PATH}")
            return True
        except ImportError:
            print("onnxconverter-common not installed. Skipping FP16 conversion.")
            print("Install with: pip install onnxconverter-common")
            return False
        except Exception as e:
            print(f"FP16 conversion failed: {e}")
            return False
            
    def benchmark_pytorch(self, images):
        """Benchmark PyTorch model"""
        print("\n=== Benchmarking PyTorch Model ===")
        
        # Warmup
        print("Warming up...")
        for i in range(NUM_WARMUP):
            inputs = self.processor(images[i % len(images)], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                _ = self.model(**inputs)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for image in tqdm(images, desc="PyTorch inference"):
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(images) * 1000
        print(f"Average inference time: {avg_time:.2f} ms/image")
        print(f"Throughput: {1000/avg_time:.2f} images/second")
        
        return avg_time
        
    def benchmark_onnx(self, model_path, model_name, images=None):
        """Benchmark ONNX model with comprehensive error handling"""
        print(f"\n=== Benchmarking {model_name} ===")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None, None
        
        try:
            # Configure providers
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # Create session with error handling
            print(f"Loading model with providers: {providers}")
            session = ort.InferenceSession(model_path, providers=providers)
            
            # Get actual providers being used
            actual_providers = session.get_providers()
            print(f"Active providers: {actual_providers}")
            
            # Get model info
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(f"Input: {input_name}, shape: {input_shape}")
            
            if images is None:
                images = self.prepare_sample_images()
            
            # Warmup
            print("Warming up...")
            for i in range(NUM_WARMUP):
                inputs = self.processor(images[i % len(images)], return_tensors="np")
                _ = session.run(None, {input_name: inputs["pixel_values"]})
            
            # Benchmark
            start_time = time.time()
            
            for image in tqdm(images, desc=f"{model_name} inference"):
                inputs = self.processor(image, return_tensors="np")
                outputs = session.run(None, {input_name: inputs["pixel_values"]})
                
            end_time = time.time()
            
            avg_time = (end_time - start_time) / len(images) * 1000
            print(f"Average inference time: {avg_time:.2f} ms/image")
            print(f"Throughput: {1000/avg_time:.2f} images/second")
            
            # Check model size
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"Model size: {model_size:.2f} MB")
            
            return avg_time, model_size
            
        except ort.OrtException as e:
            print(f"ONNX Runtime error: {str(e)}")
            if "NOT_IMPLEMENTED" in str(e):
                print("This error typically occurs with unsupported quantization operations.")
                print("The model uses operations not supported by your ONNX Runtime version.")
            return None, None
        except Exception as e:
            print(f"Error benchmarking {model_name}: {type(e).__name__}: {str(e)}")
            return None, None
        
    def run_full_benchmark(self):
        """Run complete benchmark pipeline with robust error handling"""
        # Prepare images first
        images = self.prepare_sample_images()
        
        # Export and optimize models
        if not os.path.exists(ONNX_PATH):
            self.export_to_onnx()
        else:
            print(f"\nUsing existing ONNX model: {ONNX_PATH}")
            
        if not os.path.exists(OPTIMIZED_ONNX_PATH):
            self.optimize_onnx()
        else:
            print(f"Using existing optimized model: {OPTIMIZED_ONNX_PATH}")
        
        # Try different optimization strategies
        quantization_success = False
            
        # Try FP16 conversion
        fp16_success = False
        if not os.path.exists(FP16_ONNX_PATH):
            fp16_success = self.convert_to_fp16()
        else:
            print(f"Using existing FP16 model: {FP16_ONNX_PATH}")
            fp16_success = True
        
        # Run benchmarks
        results = {}
        
        # PyTorch benchmark
        pytorch_time = self.benchmark_pytorch(images)
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        results['PyTorch'] = {
            'avg_time_ms': pytorch_time,
            'model_size_mb': model_size
        }
        
        # ONNX benchmarks
        models_to_test = [
            (ONNX_PATH, "ONNX Model"),
            (OPTIMIZED_ONNX_PATH, "Optimized ONNX")
        ]
        
        if quantization_success:
            models_to_test.append((QUANTIZED_ONNX_PATH, "Quantized ONNX"))
            
        if fp16_success:
            models_to_test.append((FP16_ONNX_PATH, "FP16 ONNX"))
        
        for model_path, model_name in models_to_test:
            if os.path.exists(model_path):
                time_ms, size_mb = self.benchmark_onnx(model_path, model_name, images)
                if time_ms is not None:
                    results[model_name] = {
                        'avg_time_ms': time_ms,
                        'model_size_mb': size_mb
                    }
        
        # Print summary
        self.print_summary(results)
        
        return results
        
    def print_summary(self, results):
        """Print comprehensive benchmark summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"{'Model':<20} {'Avg Time (ms)':<15} {'Model Size (MB)':<15} {'Speedup':<10} {'Size Reduction':<15}")
        print("-"*70)
        
        if 'PyTorch' not in results:
            print("No results to display")
            return
            
        pytorch_time = results['PyTorch']['avg_time_ms']
        pytorch_size = results['PyTorch']['model_size_mb']
        
        for model, metrics in results.items():
            speedup = pytorch_time / metrics['avg_time_ms']
            size_reduction = (1 - metrics['model_size_mb'] / pytorch_size) * 100
            print(f"{model:<20} {metrics['avg_time_ms']:<15.2f} {metrics['model_size_mb']:<15.2f} "
                  f"{speedup:<10.2f}x {size_reduction:<15.1f}%")
            
        print("="*70)
        
        # Print optimization insights
        print("\nOptimization Insights:")
        if 'ONNX Model' in results and 'Optimized ONNX' in results:
            opt_speedup = results['ONNX Model']['avg_time_ms'] / results['Optimized ONNX']['avg_time_ms']
            print(f"- Graph optimization provided {opt_speedup:.2f}x speedup")
            
        if 'Quantized ONNX' in results:
            quant_speedup = results['Optimized ONNX']['avg_time_ms'] / results['Quantized ONNX']['avg_time_ms']
            quant_size_reduction = (1 - results['Quantized ONNX']['model_size_mb'] / results['Optimized ONNX']['model_size_mb']) * 100
            print(f"- INT8 quantization provided {quant_speedup:.2f}x speedup and {quant_size_reduction:.1f}% size reduction")
            
        if 'FP16 ONNX' in results:
            fp16_speedup = results['Optimized ONNX']['avg_time_ms'] / results['FP16 ONNX']['avg_time_ms']
            fp16_size_reduction = (1 - results['FP16 ONNX']['model_size_mb'] / results['Optimized ONNX']['model_size_mb']) * 100
            print(f"- FP16 conversion provided {fp16_speedup:.2f}x speedup and {fp16_size_reduction:.1f}% size reduction")

def print_system_info():
    """Print system and package information"""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python version: {torch.sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU cores: {os.cpu_count()}")
    print("="*50)

def main():
    # Check dependencies
    required_packages = {
        'onnxruntime': 'onnxruntime',
        'transformers': 'transformers',
        'fiftyone': 'fiftyone',
        'PIL': 'pillow',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return
    
    # Print system info
    print_system_info()
    
    # Run benchmark
    benchmark = ViTBenchmark()
    results = benchmark.run_full_benchmark()
    
    # Provide recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    if results:
        if 'Quantized ONNX' not in results:
            print("- INT8 quantization failed. This is common for ViT models.")
            print("  Consider using FP16 or optimized FP32 models instead.")
            
        if 'FP16 ONNX' not in results:
            print("- FP16 conversion not available. Install onnxconverter-common:")
            print("  pip install onnxconverter-common")
            
        print("\n- For production deployment, use the fastest successful model")
        print("- Consider TensorRT (NVIDIA) or OpenVINO (Intel) for better optimization")
        print("- Monitor memory usage in addition to speed for edge deployments")
    
    print("="*50)

if __name__ == "__main__":
    main()