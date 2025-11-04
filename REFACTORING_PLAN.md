# Refactoring Plan: From LIQE-Specific to Multi-Model Framework

## Current State

- Single model implementation (LIQE)
- Model-specific code in UDTF
- Hard-coded model paths and logic

## Target State

- Multi-model framework
- Configurable model selection
- Extensible architecture
- Backward compatible (LIQE still works)

## Refactoring Steps

### Phase 1: Create Core Interfaces and Base Classes

1. **Create common package structure**
   ```
   src/main/java/com/autonavi/iqa/
   ├── common/
   ├── models/
   ├── factory/
   ├── udtf/
   └── utils/
   ```

2. **Define interfaces**
   - `IImageQualityModel.java`
   - `IModelManager.java`
   - `ModelConfig.java`
   - `QualityScore.java`

3. **Create base classes**
   - `BaseIQAModel.java`
   - `BaseONNXModelManager.java`

### Phase 2: Refactor LIQE as First Model

1. **Move LIQE to models package**
   - `LIQEModel.java` (implements IImageQualityModel)
   - `LIQEModelManager.java` (implements IModelManager)
   - Keep existing logic, just wrap in interfaces

2. **Create factory**
   - `ModelFactory.java` - Creates model instances
   - `ModelRegistry.java` - Registers available models

### Phase 3: Create Generic UDTF

1. **Create ImageQualityUDTF**
   - Generic UDTF that works with any model
   - Loads model config from ODPS resources
   - Uses factory to create model instance

2. **Maintain backward compatibility**
   - Keep old `UDTFLIQE` class (deprecated)
   - Or make it a wrapper around new generic UDTF

### Phase 4: Update Configuration

1. **Model configuration files**
   - JSON-based config for each model
   - Support ODPS resource loading

2. **Update documentation**
   - New architecture docs
   - How to add new models

## Migration Path

### For Existing Users

1. **Option 1**: Keep using `udtf_liqe` (backward compatible)
2. **Option 2**: Migrate to `udtf_iqa` with model="liqe"

### SQL Migration Example

**Old (LIQE-specific):**
```sql
CREATE FUNCTION udtf_liqe AS 'com.autonavi.liqe.UDTFLIQE'
USING 'liqe-udtf.jar', 'clip_model.onnx', 'liqe_model.onnx', 'text_features.json';
```

**New (Generic):**
```sql
CREATE FUNCTION udtf_iqa AS 'com.autonavi.iqa.udtf.ImageQualityUDTF'
USING 'iqa-udtf.jar', 'models/liqe/config.json', 'models/liqe/clip_model.onnx', ...;
```

## File Structure Changes

### Before
```
com.autonavi.liqe/
├── UDTFLIQE.java
├── ONNXModelManager.java
├── ImagePreprocessor.java
└── ...
```

### After
```
com.autonavi.iqa/
├── common/
│   ├── IImageQualityModel.java
│   └── ...
├── models/
│   ├── base/
│   └── liqe/
│       ├── LIQEModel.java
│       └── LIQEModelManager.java
├── factory/
│   └── ModelFactory.java
├── udtf/
│   └── ImageQualityUDTF.java
└── utils/
    ├── ImageDownloader.java
    └── ImagePreprocessor.java
```

## Timeline

- **Week 1**: Create interfaces and base classes
- **Week 2**: Refactor LIQE to new structure
- **Week 3**: Create generic UDTF and factory
- **Week 4**: Testing and documentation

## Risk Mitigation

1. **Backward Compatibility**: Keep old classes as wrappers
2. **Gradual Migration**: Support both old and new APIs
3. **Testing**: Comprehensive tests before each phase
4. **Documentation**: Update docs as we go

