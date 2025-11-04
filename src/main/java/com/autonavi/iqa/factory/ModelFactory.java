package com.autonavi.iqa.factory;

import com.autonavi.iqa.common.IImageQualityModel;
import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating IQA model instances
 * 
 * Supports dynamic model creation based on configuration
 */
public class ModelFactory {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelFactory.class);
    
    /**
     * Create a model instance based on configuration
     * 
     * @param config Model configuration containing model name
     * @return Initialized model instance
     * @throws ModelException if model creation fails
     */
    public static IImageQualityModel createModel(ModelConfig config) throws ModelException {
        if (config == null) {
            throw new ModelException("Model configuration cannot be null");
        }
        
        String modelName = config.getModelName();
        if (modelName == null || modelName.trim().isEmpty()) {
            throw new ModelException("Model name not specified in configuration");
        }
        
        // Get model class from registry
        Class<? extends IImageQualityModel> modelClass = ModelRegistry.getModelClass(modelName);
        if (modelClass == null) {
            throw new ModelException("Unknown model: " + modelName + ". Available models: " + 
                ModelRegistry.getAvailableModels());
        }
        
        try {
            // Create instance
            IImageQualityModel model = modelClass.getDeclaredConstructor().newInstance();
            
            // Initialize with config
            model.initialize(config);
            
            logger.info("Created and initialized model: {}", modelName);
            return model;
            
        } catch (Exception e) {
            logger.error("Failed to create model: {}", modelName, e);
            throw new ModelException("Failed to create model " + modelName + ": " + e.getMessage(), e);
        }
    }
    
    /**
     * Create a model instance by name
     * 
     * @param modelName Model name (e.g., "liqe", "dbcnn")
     * @param config Model configuration (paths, etc.)
     * @return Initialized model instance
     * @throws ModelException if model creation fails
     */
    public static IImageQualityModel createModel(String modelName, ModelConfig config) throws ModelException {
        if (config == null) {
            config = new ModelConfig();
        }
        config.setModelName(modelName);
        return createModel(config);
    }
}

