# Image Quality Assessment UDTF for ODPS

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Java](https://img.shields.io/badge/Java-8%2B-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)

A **modular and extensible** Java framework for **Image Quality Assessment (IQA)** as User Defined Table Functions (UDTF) for **Alibaba Cloud MaxCompute (ODPS)**. This project provides a production-ready solution for large-scale image quality assessment with support for multiple IQA models.

**Currently Supported Models:**
- ✅ **LIQE** (Blind Image Quality Assessment via Vision-Language Correspondence) - [CVPR 2023](https://github.com/zwx8981/LIQE)
- 🔄 **Future Models**: DBCNN, HyperIQA, MANIQA, and more from [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Conversion](#model-conversion)
- [Building and Packaging](#building-and-packaging)
- [Local Testing](#local-testing)
- [ODPS Deployment](#odps-deployment)
- [Usage Examples](#usage-examples)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

This project is a **comprehensive framework** for image quality assessment in ODPS environments. Unlike single-model implementations, this framework:

- **Supports Multiple IQA Models**: LIQE (current), with extensible architecture for DBCNN, HyperIQA, MANIQA, and more
- **Modular Design**: Easy to add new models without modifying existing code
- **ONNX Runtime Integration**: Efficient inference using ONNX Runtime for all models
- **Factory Pattern**: Dynamic model selection at runtime
- **Production-Ready**: Comprehensive error handling, batch processing, and resource management
- **ODPS Native**: Seamless integration with Alibaba Cloud MaxCompute

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture design.

### Why This Project Matters

In the era of big data and cloud computing, **large-scale image quality assessment** has become a critical component in industrial data processing pipelines. This project addresses the unique challenges of:

1. **Industrial-Scale Processing**: ODPS (MaxCompute) is Alibaba Cloud's petabyte-scale data processing platform, serving millions of users and processing billions of images daily. Traditional single-machine solutions cannot meet the scalability requirements.

2. **Production-Ready Quality Control**: Automated image quality assessment at scale enables:
   - **Content Moderation**: Filtering low-quality or inappropriate images before they reach users
   - **Data Quality Assurance**: Ensuring data quality in machine learning pipelines
   - **Cost Optimization**: Reducing storage and processing costs by filtering unusable images early
   - **User Experience**: Maintaining high-quality content standards across platforms

3. **Real-World Necessity**: 
   - **E-commerce Platforms**: Quality assessment for product images (millions of images daily)
   - **Social Media**: Content quality filtering at upload time
   - **Cloud Storage Services**: Automated quality checks for user-uploaded images
   - **Computer Vision Pipelines**: Pre-filtering before expensive ML model inference

This Java UDTF implementation bridges the gap between cutting-edge research (LIQE model) and industrial-scale deployment (ODPS), making state-of-the-art image quality assessment accessible to production systems.

**Original Paper Citation:**
```bibtex
@inproceedings{zhang2023liqe,
  title={Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},
  author={Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14071--14081},
  year={2023}
}
```

**Original Repository:** [https://github.com/zwx8981/LIQE](https://github.com/zwx8981/LIQE)

### Model Files Source

The PyTorch model weights required for this project can be obtained from the following sources:

1. **LIQE Model Weights** (`liqe_mix.pt` and `liqe_text_feat_mix.pt`):
   - Download from the [original LIQE repository](https://github.com/zwx8981/LIQE)
   - Pre-trained weights are available in the repository or via Google Drive/Baidu Pan links in the README
   - The `liqe_mix.pt` model is trained on multiple datasets (as described in the paper)
   - The `liqe_text_feat_mix.pt` contains pre-computed text features for efficient inference

2. **CLIP Model Weights** (`ViT-B-32.pt`):
   - This is the Vision Transformer (ViT-B/32) backbone from OpenAI's CLIP model
   - Can be downloaded from:
     - [OpenAI CLIP repository](https://github.com/openai/CLIP)
     - Or automatically loaded via `torch.hub.load('openai/clip-vit-base-patch32')`
   - The ViT-B/32 model is the standard CLIP image encoder used in LIQE

**Note:** After downloading the PyTorch model files, place them in `weights/torch/` directory and use the conversion script (`scripts/liqe_torch2onnx.py`) to generate ONNX models for Java inference.

## ✨ Features

- 🏗️ **Extensible Architecture**: Plugin-based design for easy model integration
- 🤖 **Multiple Model Support**: LIQE (current), with framework ready for DBCNN, HyperIQA, MANIQA, etc.
- ⚡ **ONNX Runtime Integration**: Efficient inference using ONNX Runtime
- 📦 **Batch Processing**: Process multiple images in a single UDTF call
- 🌐 **URL Support**: Direct image URL processing (no local file storage needed)
- 🛡️ **Error Handling**: Robust error handling with graceful degradation
- 🐧 **Cross-Platform**: Supports Linux (ODPS) and macOS (local testing)
- 💾 **Memory Efficient**: Optimized for large-scale data processing
- 🚀 **Production Ready**: Comprehensive error handling and logging
- 🔧 **Configurable**: JSON-based model configuration

## 🎯 Use Cases

This UDTF is designed for **Alibaba Cloud MaxCompute (ODPS)** environments and is ideal for:

1. **Large-Scale Image Quality Assessment**
   - Batch processing of images from OSS (Object Storage Service)
   - Quality filtering for image datasets
   - Image quality monitoring in production pipelines

2. **Data Quality Control**
   - Automated quality checks for uploaded images
   - Quality-based filtering in data pipelines
   - Image quality metrics for analytics

3. **Content Moderation**
   - Pre-filtering low-quality images before processing
   - Quality-based image selection
   - Batch quality assessment workflows

## 📁 Project Structure

```
├── src/main/java/com/autonavi/iqa/
│   ├── common/                     # Common interfaces and utilities
│   │   ├── IImageQualityModel.java    # Model interface
│   │   ├── IModelManager.java         # ONNX manager interface
│   │   ├── ModelConfig.java           # Model configuration
│   │   └── QualityScore.java          # Score result wrapper
│   ├── models/                     # Model implementations
│   │   ├── base/                    # Base classes
│   │   │   └── BaseIQAModel.java        # Abstract base class
│   │   └── liqe/                    # LIQE model (current)
│   │       ├── LIQEModel.java
│   │       └── LIQEModelManager.java
│   ├── factory/                    # Factory pattern
│   │   ├── ModelFactory.java          # Creates model instances
│   │   └── ModelRegistry.java         # Model registry
│   ├── udtf/                       # UDTF implementation
│   │   └── ImageQualityUDTF.java      # Generic UDTF
│   └── utils/                      # Utilities
│       ├── ImageDownloader.java
│       └── ImagePreprocessor.java
├── scripts/
│   ├── liqe_torch2onnx.py         # Convert PyTorch models to ONNX
│   ├── udtf_liqe.py               # Python UDTF implementation (reference)
│   └── liqe.py                    # Original LIQE model implementation
├── weights/
│   ├── torch/                     # PyTorch model weights
│   │   ├── liqe_mix.pt
│   │   ├── ViT-B-32.pt
│   │   └── liqe_text_feat_mix.pt
│   └── onnx/                      # ONNX model files (generated)
│       ├── clip_model.onnx
│       ├── liqe_model.onnx
│       └── text_features.json
├── pom.xml                        # Maven configuration
├── test_with_url.sh               # Test script with URL
├── test_udtf.sh                  # Test UDTF implementation
├── compile_and_test.sh            # Compile and test script
├── package_jar.sh                 # Package JAR script
└── README.md                      # This file
```

## 🔧 Prerequisites

### For Model Conversion (Python)
- Python 3.7+
- PyTorch 1.8+
- torchvision
- onnx
- onnxruntime
- Pillow
- requests

### For Java Development
- Java 8+ (JDK 8 for ODPS compatibility)
- Maven 3.6+ (optional, for automated building)
- Access to ODPS SDK (for UDTF deployment)

### For ODPS Deployment
- Alibaba Cloud MaxCompute (ODPS) account
- ODPS CLI tools configured
- Access to upload resources and create UDTF functions

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/86MaxCao/iqa-odps-udtf.git
cd iqa-odps-udtf
```

### 2. Convert PyTorch Models to ONNX

```bash
# Install Python dependencies
pip install torch torchvision onnx onnxruntime pillow requests

# Convert models
python scripts/liqe_torch2onnx.py \
    --liqe_model_path weights/torch/liqe_mix.pt \
    --clip_model_path weights/torch/ViT-B-32.pt \
    --text_feat_path weights/torch/liqe_text_feat_mix.pt \
    --output_dir weights/onnx
```

This will generate:
- `weights/onnx/clip_model.onnx`
- `weights/onnx/liqe_model.onnx`
- `weights/onnx/text_features.json`

### 3. Compile Java Code

```bash
# Using Maven (recommended)
mvn clean compile

# OR using manual script
./compile_and_test.sh
```

### 4. Test Locally

```bash
# Test with default URLs
./test_with_url.sh

# Test with custom URLs
./test_with_url.sh "http://example.com/image1.jpg" "http://example.com/image2.jpg"

# Test UDTF implementation
./test_udtf.sh
```

## 📦 Model Conversion

### Step-by-Step Conversion

1. **Download PyTorch Models**
   
   First, download the required model files:
   
   - **LIQE Models** (`liqe_mix.pt` and `liqe_text_feat_mix.pt`):
     - Download from: [https://github.com/zwx8981/LIQE](https://github.com/zwx8981/LIQE)
     - Google Drive: [Link in original repository](https://github.com/zwx8981/LIQE)
     - Baidu Pan: [Link in original repository](https://github.com/zwx8981/LIQE)
   
   - **CLIP Model** (`ViT-B-32.pt`):
     - Download from: [OpenAI CLIP](https://github.com/openai/CLIP)
     - Or use: `torch.hub.load('openai/clip-vit-base-patch32')`
     - This is the standard ViT-B/32 CLIP model

2. **Place Models in Directory**
   
   Place all downloaded PyTorch model files in `weights/torch/`:
   ```
   weights/torch/
   ├── liqe_mix.pt              # LIQE model weights (from LIQE repo)
   ├── ViT-B-32.pt              # CLIP model weights (from OpenAI CLIP)
   └── liqe_text_feat_mix.pt    # Text features (from LIQE repo)
   ```

3. **Run Conversion Script**
   ```bash
   python scripts/liqe_torch2onnx.py \
       --liqe_model_path weights/torch/liqe_mix.pt \
       --clip_model_path weights/torch/ViT-B-32.pt \
       --text_feat_path weights/torch/liqe_text_feat_mix.pt \
       --output_dir weights/onnx \
       --device cpu  # or 'cuda' for GPU
   ```

4. **Verify Output**
   - Check that all three files are generated in `weights/onnx/`:
     - `clip_model.onnx`
     - `liqe_model.onnx`
     - `text_features.json`
   - Test the ONNX models using `SimpleONNXTest.java`

## 🏗️ Building and Packaging

### Using Maven (Recommended)

```bash
mvn clean package
```

This creates:
- `target/liqe-udtf-1.0.0.jar` - Main JAR
- `target/liqe-udtf-1.0.0-jar-with-dependencies.jar` - Fat JAR (for ODPS)

### Manual Packaging

```bash
./package_jar.sh
```

The fat JAR includes:
- All Java classes
- All dependencies (ONNX Runtime, Jackson, JavaCV, etc.)
- Linux native libraries (OpenCV, OpenBLAS) for ODPS

**Note:** The JAR is compiled for Java 8 compatibility to work with ODPS runtime.

### Pre-built JAR Files (Recommended)

**✅ Best Practice**: Pre-built JAR files are distributed via [GitHub Releases](https://github.com/86MaxCao/iqa-odps-udtf/releases) rather than committed to the repository.

**Why GitHub Releases?**
- ✅ Keeps repository size small (JAR is ~200MB)
- ✅ Standard industry practice
- ✅ Version control and download tracking
- ✅ Easy for users to get started quickly

**Download Latest Release:**
```bash
# Download latest release JAR
wget https://github.com/86MaxCao/iqa-odps-udtf/releases/latest/download/iqa-udtf-1.0.0-jar-with-dependencies.jar

# Or download specific version
wget https://github.com/86MaxCao/iqa-odps-udtf/releases/download/v1.0.0/iqa-udtf-1.0.0-jar-with-dependencies.jar
```

**What's Included:**
- All Java classes
- All dependencies (ONNX Runtime, Jackson, JavaCV, etc.)
- Linux native libraries (OpenCV, OpenBLAS) for ODPS
- Ready for immediate ODPS deployment

**Note:** If you prefer to build from source, follow the instructions above. See [RELEASES.md](RELEASES.md) for details on creating releases.

## 🧪 Local Testing

### Test with Image URLs

```bash
# Test with default test URLs
./test_with_url.sh

# Test with custom URLs
./test_with_url.sh \
    "http://example.com/image1.jpg" \
    "http://example.com/image2.jpg"
```

### Test UDTF Implementation

```bash
# Test UDTF process method
./test_udtf.sh

# Test with custom URLs
./test_udtf.sh "http://example.com/image1.jpg" "http://example.com/image2.jpg"
```

### Python Script Testing

```bash
# Test Python UDTF implementation
python scripts/udtf_liqe.py \
    --model_path weights/torch/liqe_mix.pt \
    --clip_model_path weights/torch/ViT-B-32.pt \
    --image_path "http://example.com/image.jpg"
```

## ☁️ ODPS Deployment

### Step 1: Upload Resources

```sql
-- Upload JAR file
ADD JAR target/liqe-udtf-1.0.0-jar-with-dependencies.jar;

-- Upload ONNX model files
ADD FILE weights/onnx/clip_model.onnx AS clip_model.onnx;
ADD FILE weights/onnx/liqe_model.onnx AS liqe_model.onnx;
ADD FILE weights/onnx/text_features.json AS text_features.json;
```

### Step 2: Create UDTF Function

**Option 1: Generic UDTF (Recommended - supports multiple models)**
```sql
CREATE FUNCTION udtf_iqa AS 'com.autonavi.iqa.udtf.ImageQualityUDTF'
USING 'iqa-udtf-1.0.0-jar-with-dependencies.jar',
      'models/liqe/config.json',
      'models/liqe/clip_model.onnx',
      'models/liqe/liqe_model.onnx',
      'models/liqe/text_features.json';
```

**Option 2: LIQE-specific UDTF (Backward compatible)**
```sql
CREATE FUNCTION udtf_liqe AS 'com.autonavi.iqa.models.liqe.LIQEUDTF'
USING 'iqa-udtf-1.0.0-jar-with-dependencies.jar',
      'models/liqe/clip_model.onnx',
      'models/liqe/liqe_model.onnx',
      'models/liqe/text_features.json';
```

### Step 3: Use in SQL Queries

```sql
-- Example: Process image URLs from a table (using generic UDTF)
SELECT 
    image_url,
    quality_score
FROM (
    SELECT 
        COLLECT_LIST(oss_url) AS urls
    FROM your_image_table
    GROUP BY some_group_key
) t
LATERAL VIEW udtf_iqa(urls) AS image_url, quality_score;
```

**Note:** To use a specific model, set the model name in the configuration file. The framework will automatically load the appropriate model.

### Step 4: Verify Deployment

```sql
-- Test with sample URLs
SELECT 
    image_url,
    quality_score
FROM (
    SELECT ARRAY(
        'http://example.com/image1.jpg',
        'http://example.com/image2.jpg'
    ) AS urls
) t
LATERAL VIEW udtf_liqe(urls) AS image_url, quality_score;
```

## 💡 Usage Examples

### Example 1: Batch Quality Assessment

```sql
-- Process multiple images from a table
SELECT 
    batch_id,
    image_url,
    quality_score,
    CASE 
        WHEN quality_score < 2.0 THEN 'low'
        WHEN quality_score < 3.5 THEN 'medium'
        ELSE 'high'
    END AS quality_level
FROM (
    SELECT 
        batch_id,
        COLLECT_LIST(oss_url) AS urls
    FROM image_uploads
    WHERE date = '2024-01-01'
    GROUP BY batch_id
) t
LATERAL VIEW udtf_liqe(urls) AS image_url, quality_score
WHERE quality_score >= 2.0;  -- Filter low-quality images
```

### Example 2: Quality Filtering

```sql
-- Filter images by quality threshold
WITH quality_assessed AS (
    SELECT 
        image_id,
        image_url,
        quality_score
    FROM (
        SELECT 
            image_id,
            COLLECT_LIST(oss_url) AS urls
        FROM image_table
        GROUP BY image_id
    ) t
    LATERAL VIEW udtf_liqe(urls) AS image_url, quality_score
)
SELECT 
    image_id,
    image_url,
    quality_score
FROM quality_assessed
WHERE quality_score >= 3.0;  -- Only keep good quality images
```

### Example 3: Quality Statistics

```sql
-- Calculate quality statistics per batch
SELECT 
    batch_id,
    COUNT(*) AS total_images,
    AVG(quality_score) AS avg_quality,
    MIN(quality_score) AS min_quality,
    MAX(quality_score) AS max_quality,
    SUM(CASE WHEN quality_score >= 3.0 THEN 1 ELSE 0 END) AS high_quality_count
FROM (
    SELECT 
        batch_id,
        COLLECT_LIST(oss_url) AS urls
    FROM image_table
    GROUP BY batch_id
) t
LATERAL VIEW udtf_liqe(urls) AS image_url, quality_score
GROUP BY batch_id;
```

## ⚡ Performance

- **Batch Processing**: Processes multiple images in a single UDTF call
- **Memory Efficient**: Processes images in batches of 8 (configurable)
- **Error Handling**: Failed images return error score (-1.0) without blocking
- **Native Libraries**: Optimized OpenCV/OpenBLAS for image processing

**Performance Tips:**
- Process images in batches for better throughput
- Use appropriate batch sizes based on available memory
- Monitor ODPS resource usage and adjust accordingly

## 🔍 Troubleshooting

### Common Issues

#### 1. `UnsatisfiedLinkError: no jniopenblas_nolapack`

**Solution:** Ensure Linux native libraries are included in the JAR:
- Download `opencv-4.7.0-1.5.9-linux-x86_64.jar`
- Download `openblas-0.3.23-1.5.9-linux-x86_64.jar`
- Place them in `target/lib/` before packaging

#### 2. Model Files Not Found

**Solution:** Verify model files are uploaded to ODPS:
```sql
LIST RESOURCES;
```

#### 3. Java Version Mismatch

**Solution:** Ensure JAR is compiled for Java 8:
```xml
<!-- In pom.xml -->
<maven.compiler.source>8</maven.compiler.source>
<maven.compiler.target>8</maven.compiler.target>
```

#### 4. Image Download Failures

**Solution:** The UDTF handles download failures gracefully:
- Returns error score (-1.0) for failed downloads
- Logs warnings for debugging
- Continues processing remaining images

### Debug Mode

Enable debug logging:
```sql
SET odps.udf.log.level=DEBUG;
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Model Licenses

- **LIQE Model**: The original LIQE model is licensed under MIT License (see [original repository](https://github.com/zwx8981/LIQE))
- **CLIP Model**: OpenAI CLIP is licensed under MIT License (see [OpenAI CLIP](https://github.com/openai/CLIP))

## 🙏 Acknowledgments

- **Original LIQE Implementation**: [https://github.com/zwx8981/LIQE](https://github.com/zwx8981/LIQE) - The research paper and original PyTorch implementation
- **OpenAI CLIP**: [https://github.com/openai/CLIP](https://github.com/openai/CLIP) - Vision-language model backbone
- **ONNX Runtime**: [https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) - Efficient inference engine
- **JavaCV**: [https://github.com/bytedeco/javacv](https://github.com/bytedeco/javacv) - Java bindings for OpenCV
- **ODPS SDK**: Alibaba Cloud MaxCompute - Big data processing platform

### Citation

If you use this project in your research or work, please cite:

1. **Original LIQE Paper:**
```bibtex
@inproceedings{zhang2023liqe,
  title={Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},
  author={Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14071--14081},
  year={2023}
}
```

2. **This Implementation:**
```bibtex
@software{liqe_odps_udtf,
  title={LIQE ODPS UDTF: Java Implementation for Large-Scale Image Quality Assessment},
  author={Your Name},
  year={2024},
          url={https://github.com/86MaxCao/iqa-odps-udtf}
}
```

## 📧 Contact

For issues and questions, please open an issue on GitHub.

---

**Note:** This UDTF is optimized for ODPS (MaxCompute) environments. For other environments, you may need to adjust native library dependencies and deployment procedures.
