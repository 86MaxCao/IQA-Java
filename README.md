# IQA-Java

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Java](https://img.shields.io/badge/Java-8%2B-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ONNX%20Models-yellow.svg)](https://huggingface.co/86Cao/IQA-ONNX-Models)

A modular Java framework for **No-Reference Image Quality Assessment (NR-IQA)** with ONNX Runtime inference. Supports 7 state-of-the-art blind IQA models. Includes an ODPS/MaxCompute UDTF adapter for large-scale batch processing.

## Supported Models

| Model | Paper | Preprocessing | Crops | Score Range |
|-------|-------|---------------|-------|-------------|
| **LIQE** | [CVPR 2023](https://github.com/zwx8981/LIQE) | CLIP norm, 15 patches | 15 | 1.0 - 5.0 |
| **DBCNN** | [IEEE TCSVT 2020](https://github.com/zwx8981/DBCNN-PyTorch) | ImageNet norm, full image | 1 | continuous |
| **HyperIQA** | [CVPR 2020](https://github.com/SSL92/hyperIQA) | ImageNet norm | 25 | continuous |
| **MANIQA** | [CVPRW 2022](https://github.com/IIGROUP/MANIQA) | Inception norm (0.5) | 20 | continuous |
| **MUSIQ** | [ICCV 2021](https://github.com/google-research/google-research/tree/master/musiq) | [-1, 1] norm | 1 | continuous |
| **TReS** | [WACV 2022](https://github.com/isalirezag/TReS) | ImageNet norm | 50 | continuous |
| **CLIPIQA** | [AAAI 2023](https://github.com/IceClear/CLIP-IQA) | CLIP norm | 1 | 0.0 - 1.0 |

All models are converted from [IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch) to ONNX format for Java inference. Pre-converted ONNX weights are available at [86Cao/IQA-ONNX-Models](https://huggingface.co/86Cao/IQA-ONNX-Models) on HuggingFace.

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Conversion](#model-conversion)
- [Building](#building)
- [Testing](#testing)
- [ODPS Deployment](#odps-deployment)
- [Usage Examples](#usage-examples)
- [Adding New Models](#adding-new-models)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Architecture

```
                    ┌─────────────────────────┐
                    │   ImageQualityUDTF      │
                    │   (ODPS entry point)    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     ModelFactory        │
                    │  (creates by name)      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │    ModelRegistry        │
                    │  (name → class map)     │
                    └───────────┬─────────────┘
                                │
        ┌───────┬───────┬───────┼───────┬───────┬───────┐
        ▼       ▼       ▼       ▼       ▼       ▼       ▼
      LIQE    DBCNN  HyperIQA MANIQA  MUSIQ   TReS  CLIPIQA
        │       │       │       │       │       │       │
        └───────┴───────┴───┬───┴───────┴───────┴───────┘
                            │
                    ┌───────▼─────────────┐
                    │   BaseIQAModel      │
                    │ + BaseONNXManager   │
                    └───────┬─────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ImageDownloader  ImagePreprocessor  ONNX Runtime
```

Each model follows the same pattern:
- `XxxModel.java` extends `BaseIQAModel` — handles preprocessing and multi-crop averaging
- `XxxModelManager.java` extends `BaseONNXModelManager` — manages ONNX session lifecycle

## Project Structure

```
├── src/main/java/com/autonavi/iqa/
│   ├── common/                          # Interfaces and shared types
│   │   ├── IImageQualityModel.java         # Model interface
│   │   ├── IModelManager.java              # ONNX manager interface
│   │   ├── ModelConfig.java                # Configuration POJO
│   │   ├── ModelException.java             # Domain exception
│   │   └── QualityScore.java               # Score result wrapper
│   ├── models/
│   │   ├── base/
│   │   │   ├── BaseIQAModel.java           # Abstract base for all models
│   │   │   └── BaseONNXModelManager.java   # ONNX session management
│   │   ├── liqe/                           # LIQE implementation
│   │   ├── dbcnn/                          # DBCNN implementation
│   │   ├── hyperiqa/                       # HyperIQA implementation
│   │   ├── maniqa/                         # MANIQA implementation
│   │   ├── musiq/                          # MUSIQ implementation
│   │   ├── tres/                           # TReS implementation
│   │   └── clipiqa/                        # CLIPIQA implementation
│   ├── factory/
│   │   ├── ModelFactory.java               # Creates model by name
│   │   └── ModelRegistry.java              # Static model registry
│   ├── udtf/
│   │   └── ImageQualityUDTF.java           # ODPS UDTF entry point
│   └── utils/
│       ├── ImageDownloader.java            # HTTP image fetcher
│       ├── ImagePreprocessor.java          # Resize, crop, normalize
│       └── TextFeatureLoader.java          # JSON text features (LIQE)
├── src/test/java/com/autonavi/iqa/        # Unit tests (JUnit 5)
├── scripts/
│   ├── liqe_torch2onnx.py                 # LIQE conversion
│   ├── dbcnn_torch2onnx.py                # DBCNN conversion
│   ├── hyperiqa_torch2onnx.py             # HyperIQA conversion
│   ├── maniqa_torch2onnx.py               # MANIQA conversion
│   ├── musiq_torch2onnx.py                # MUSIQ conversion
│   ├── tres_torch2onnx.py                 # TReS conversion
│   └── clipiqa_torch2onnx.py              # CLIPIQA conversion
├── pom.xml
└── README.md
```

## Prerequisites

### Python (for model conversion)
- Python 3.7+
- PyTorch 1.8+
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) cloned locally
- onnx, onnxruntime

### Java (for development and deployment)
- JDK 8+ (Java 8 for ODPS compatibility)
- Maven 3.6+

### ODPS (for deployment)
- Alibaba Cloud MaxCompute account
- ODPS CLI configured

## Quick Start

### 1. Clone

```bash
git clone https://github.com/86Cao/IQA-Java.git
cd IQA-Java
```

### 2. Convert a model to ONNX

Each model has a dedicated conversion script. Example for LIQE:

```bash
python scripts/liqe_torch2onnx.py \
    --liqe_model_path /path/to/liqe_mix.pt \
    --clip_model_path /path/to/ViT-B-32.pt \
    --text_feat_path /path/to/liqe_text_feat_mix.pt \
    --output_dir weights/onnx
```

For other models, conversion scripts load pretrained weights via `pyiqa`:

```bash
# Example: DBCNN
python scripts/dbcnn_torch2onnx.py

# Example: HyperIQA
python scripts/hyperiqa_torch2onnx.py
```

All scripts save ONNX files and verify them with onnxruntime before exiting.

### 3. Build

```bash
mvn clean package
```

### 4. Deploy to ODPS

See [ODPS Deployment](#odps-deployment) below.

## Model Conversion

Each conversion script wraps the PyTorch model to simplify the ONNX graph (eval-mode only, no data augmentation) and exports with:
- `opset_version=14`
- `dynamic_axes` for batch dimension
- `do_constant_folding=True`

### Conversion Details

| Model | Script | Input Shape | ONNX Files | Notes |
|-------|--------|-------------|------------|-------|
| LIQE | `liqe_torch2onnx.py` | (1,3,224,224) | [`clip_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/clip_model.onnx) [`liqe_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/liqe_model.onnx) [`text_features.json`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/text_features.json) | Requires separate CLIP + LIQE weights, exports text features as JSON |
| DBCNN | `dbcnn_torch2onnx.py` | (1,3,512,384) | [`dbcnn_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/dbcnn_model.onnx) | Wraps VGG16 + SCNN + bilinear pooling |
| HyperIQA | `hyperiqa_torch2onnx.py` | (1,3,224,224) | [`hyperiqa_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/hyperiqa_model.onnx) | Exports `forward_patch` only (no multi-crop logic) |
| MANIQA | `maniqa_torch2onnx.py` | (1,3,224,224) | [`maniqa_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/maniqa_model.onnx) | Replaces ViT hooks with explicit layer indexing |
| MUSIQ | `musiq_torch2onnx.py` | (1,3,224,224) | [`musiq_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/musiq_model.onnx) | Simplifies multi-scale to single-scale fixed-size |
| TReS | `tres_torch2onnx.py` | (1,3,224,224) | [`tres_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/tres_model.onnx) | Eval path only (no flipped image / consistency loss) |
| CLIPIQA | `clipiqa_torch2onnx.py` | (1,3,224,224) | [`clipiqa_model.onnx`](https://huggingface.co/86Cao/IQA-ONNX-Models/blob/main/clipiqa_model.onnx) | Pre-encodes text features as buffer, exports image encoder + scoring head |

### Where to put model files

The conversion scripts save ONNX files to the output directory you specify. For ODPS deployment, upload them as resources. For local testing, configure paths via `ModelConfig`:

```java
ModelConfig config = new ModelConfig("liqe");
config.setModelPath("clip_onnx", "/path/to/clip_model.onnx");
config.setModelPath("liqe_onnx", "/path/to/liqe_model.onnx");
config.setModelPath("text_features", "/path/to/text_features.json");
```

## Building

```bash
# Compile and run tests
mvn clean verify

# Package fat JAR for ODPS (includes all dependencies)
mvn clean package
```

Output:
- `target/iqa-udtf-1.0.0.jar` — main JAR
- `target/iqa-udtf-1.0.0-jar-with-dependencies.jar` — fat JAR for ODPS deployment

The fat JAR includes ONNX Runtime, Jackson, JavaCV (with Linux native libraries), and Apache HttpClient.

## Testing

Unit tests cover core components without requiring ONNX model files or native libraries:

```bash
mvn test
```

Test classes:
- `QualityScoreTest` — score wrapper, error handling, constants
- `ModelConfigTest` — configuration POJO, typed parameter retrieval
- `ModelRegistryTest` — model registration, case-insensitive lookup, validation
- `ModelFactoryTest` — factory creation, error messages
- `ImagePreprocessorTest` — normalization, patch selection, ONNX input formatting

## ODPS Deployment

### 1. Upload resources

```sql
-- Upload fat JAR
ADD JAR iqa-udtf-1.0.0-jar-with-dependencies.jar;

-- Upload model files (example for LIQE)
ADD FILE clip_model.onnx;
ADD FILE liqe_model.onnx;
ADD FILE text_features.json;
```

### 2. Create UDTF

```sql
CREATE FUNCTION image_quality AS 'com.autonavi.iqa.udtf.ImageQualityUDTF'
USING 'iqa-udtf-1.0.0-jar-with-dependencies.jar,clip_model.onnx,liqe_model.onnx,text_features.json';
```

### 3. Query

```sql
SELECT T.url, T.score
FROM (
    SELECT COLLECT_LIST(image_url) AS urls
    FROM image_table
    WHERE dt = '20260601'
) t1
LATERAL VIEW image_quality('liqe', urls) T AS url, score;
```

The first argument selects the model: `liqe`, `dbcnn`, `hyperiqa`, `maniqa`, `musiq`, `tres`, or `clipiqa`.

## Usage Examples

### Batch quality assessment with filtering

```sql
SELECT url, score,
    CASE
        WHEN score < 2.0 THEN 'low'
        WHEN score < 3.5 THEN 'medium'
        ELSE 'high'
    END AS quality_level
FROM (
    SELECT COLLECT_LIST(oss_url) AS urls
    FROM image_uploads
    WHERE dt = '20260601'
    GROUP BY batch_id
) t
LATERAL VIEW image_quality('liqe', urls) T AS url, score
WHERE score >= 0;  -- exclude errors (score = -1.0)
```

### Compare models

```sql
-- Run two models on the same images
SELECT a.url, a.score AS liqe_score, b.score AS dbcnn_score
FROM (
    SELECT COLLECT_LIST(image_url) AS urls FROM sample_images
) t
LATERAL VIEW image_quality('liqe', urls) a AS url, score
JOIN (
    SELECT url, score FROM (
        SELECT COLLECT_LIST(image_url) AS urls FROM sample_images
    ) t2
    LATERAL VIEW image_quality('dbcnn', urls) b AS url, score
) b ON a.url = b.url;
```

### Quality statistics

```sql
SELECT
    COUNT(*) AS total,
    AVG(score) AS avg_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score,
    SUM(CASE WHEN score >= 3.0 THEN 1 ELSE 0 END) AS high_quality
FROM (
    SELECT COLLECT_LIST(image_url) AS urls
    FROM image_table
) t
LATERAL VIEW image_quality('liqe', urls) T AS url, score
WHERE score >= 0;
```

## Adding New Models

1. **Create model classes** in `src/main/java/com/autonavi/iqa/models/yourmodel/`:
   - `YourModel.java` extending `BaseIQAModel`
   - `YourModelManager.java` extending `BaseONNXModelManager`

2. **Register** in `ModelRegistry.java`:
   ```java
   registerModel("yourmodel", YourModel.class);
   ```

3. **Create conversion script** in `scripts/yourmodel_torch2onnx.py`

4. **Build and test**:
   ```bash
   mvn clean verify
   ```

The factory and UDTF will automatically pick up the new model by name.

## Troubleshooting

### `UnsatisfiedLinkError: no jniopenblas_nolapack`

Ensure the fat JAR includes Linux native libraries. The `javacv-platform` dependency bundles them automatically via Maven.

### Model files not found in ODPS

Verify resources are uploaded:
```sql
LIST RESOURCES;
```

### Java version mismatch

The project targets Java 8. Verify in `pom.xml`:
```xml
<maven.compiler.source>8</maven.compiler.source>
<maven.compiler.target>8</maven.compiler.target>
```

### Image download failures

The UDTF returns `-1.0` for failed downloads and continues processing remaining images. Check ODPS logs for details:
```sql
SET odps.udf.log.level=DEBUG;
```

## License

MIT License. See [LICENSE](LICENSE).

### Model Licenses

All models are adapted from open-source implementations:
- [LIQE](https://github.com/zwx8981/LIQE) — MIT
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) — MIT (DBCNN, HyperIQA, MANIQA, MUSIQ, TReS, CLIPIQA)
- [OpenAI CLIP](https://github.com/openai/CLIP) — MIT

## Acknowledgments

- [IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch) — Unified IQA toolkit
- [LIQE](https://github.com/zwx8981/LIQE) — Vision-language IQA
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Inference engine
- [JavaCV](https://github.com/bytedeco/javacv) — Java OpenCV bindings
- [Alibaba Cloud MaxCompute](https://www.alibabacloud.com/product/maxcompute) — ODPS platform

### Citation

```bibtex
@software{iqa_java,
  title={IQA-Java: Multi-Model Image Quality Assessment with ONNX Runtime},
  url={https://github.com/86Cao/IQA-Java},
  year={2024}
}
```
