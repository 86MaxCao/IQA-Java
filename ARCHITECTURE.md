# Architecture Design

## Overview

This project is designed as a **modular and extensible framework** for image quality assessment in ODPS environments. The architecture supports multiple IQA models and can be easily extended to include new models from IQA-PyTorch or other sources.

## Design Principles

1. **Modularity**: Each IQA model is a separate, self-contained module
2. **Extensibility**: Easy to add new models without modifying existing code
3. **Abstraction**: Common interfaces for all models
4. **Factory Pattern**: Dynamic model creation based on configuration
5. **Separation of Concerns**: Image processing, model inference, and UDTF logic are separated

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   UDTF Layer                             │
│  ImageQualityUDTF (Generic UDTF for all models)        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Factory Layer                               │
│  ModelFactory (Creates model instances by name)         │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│            Model Interface Layer                         │
│  IImageQualityModel (Common interface)                  │
│  IModelManager (ONNX model management)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│            Model Implementation Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ LIQEModel   │  │ DBCNNModel  │  │ HyperIQAModel│     │
│  │ (Current)   │  │ (Future)    │  │ (Future)    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│          Infrastructure Layer                            │
│  ImagePreprocessor, ImageDownloader, TextFeatureLoader  │
└─────────────────────────────────────────────────────────┘
```

## Package Structure

```
com.autonavi.iqa
├── common/                    # Common utilities and interfaces
│   ├── IImageQualityModel.java      # Model interface
│   ├── IModelManager.java           # ONNX model manager interface
│   ├── ModelConfig.java             # Model configuration
│   └── QualityScore.java            # Score result wrapper
│
├── models/                    # Model implementations
│   ├── base/                  # Base classes
│   │   ├── BaseIQAModel.java        # Abstract base class
│   │   └── BaseONNXModelManager.java # Base ONNX manager
│   │
│   ├── liqe/                  # LIQE model implementation
│   │   ├── LIQEModel.java           # LIQE-specific logic
│   │   ├── LIQEModelManager.java    # LIQE ONNX manager
│   │   └── LIQETextFeatureLoader.java
│   │
│   └── future/               # Future model implementations
│       ├── dbcnn/            # DBCNN model (example)
│       └── hyperiqa/         # HyperIQA model (example)
│
├── factory/                   # Factory pattern
│   ├── ModelFactory.java            # Creates model instances
│   └── ModelRegistry.java           # Model registry
│
├── udtf/                      # UDTF implementation
│   ├── ImageQualityUDTF.java       # Generic UDTF
│   └── ModelConfigLoader.java      # Loads model config from ODPS
│
└── utils/                     # Utilities
    ├── ImageDownloader.java
    ├── ImagePreprocessor.java
    └── ONNXModelLoader.java
```

## Key Interfaces

### IImageQualityModel

```java
public interface IImageQualityModel {
    /**
     * Initialize the model
     */
    void initialize(ModelConfig config) throws ModelException;
    
    /**
     * Assess image quality from URL
     * @param imageUrl Image URL
     * @return Quality score (typically 1-5 range)
     */
    QualityScore assessQuality(String imageUrl) throws ModelException;
    
    /**
     * Batch assess multiple images
     * @param imageUrls Array of image URLs
     * @return Array of quality scores (same length as input)
     */
    QualityScore[] assessQualityBatch(String[] imageUrls) throws ModelException;
    
    /**
     * Get model name
     */
    String getModelName();
    
    /**
     * Get model version
     */
    String getModelVersion();
    
    /**
     * Cleanup resources
     */
    void close();
}
```

### IModelManager

```java
public interface IModelManager {
    /**
     * Load model from path or ODPS resource
     */
    void loadModel(String modelPath, ExecutionContext ctx) throws ModelException;
    
    /**
     * Run inference
     */
    float[] runInference(float[] input) throws ModelException;
    
    /**
     * Close model resources
     */
    void close();
}
```

## Model Configuration

Models are configured via JSON files or ODPS resources:

```json
{
  "model": "liqe",
  "version": "1.0",
  "onnx_models": {
    "clip_model": "clip_model.onnx",
    "main_model": "liqe_model.onnx"
  },
  "text_features": "text_features.json",
  "preprocessing": {
    "target_width": 640,
    "target_height": 360,
    "patch_size": 224,
    "num_patches": 15
  }
}
```

## Adding New Models

To add a new model (e.g., DBCNN):

1. **Create model package**: `com.autonavi.iqa.models.dbcnn`

2. **Implement interfaces**:
   ```java
   public class DBCNNModel extends BaseIQAModel implements IImageQualityModel {
       // Implement model-specific logic
   }
   
   public class DBCNNModelManager extends BaseONNXModelManager implements IModelManager {
       // Implement ONNX inference
   }
   ```

3. **Register in factory**:
   ```java
   ModelRegistry.register("dbcnn", DBCNNModel.class);
   ```

4. **Add model configuration**:
   - Create `weights/onnx/dbcnn/` directory
   - Add model files and config.json

5. **Update UDTF**:
   - No code changes needed! Just use different model name in SQL

## Benefits

1. **Easy Extension**: Add new models without touching existing code
2. **Runtime Selection**: Choose model at runtime via configuration
3. **Maintainability**: Clear separation of concerns
4. **Testability**: Each component can be tested independently
5. **Future-Proof**: Ready for IQA-PyTorch model integration

