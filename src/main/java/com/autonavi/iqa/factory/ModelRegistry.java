package com.autonavi.iqa.factory;

import com.autonavi.iqa.common.IImageQualityModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Registry for IQA models
 * 
 * Maps model names to their implementation classes
 */
public class ModelRegistry {
    
    private static final Logger logger = LoggerFactory.getLogger(ModelRegistry.class);
    
    private static final Map<String, Class<? extends IImageQualityModel>> models = new HashMap<>();
    
    static {
        registerModel("liqe", com.autonavi.iqa.models.liqe.LIQEModel.class);
        registerModel("dbcnn", com.autonavi.iqa.models.dbcnn.DBCNNModel.class);
        registerModel("hyperiqa", com.autonavi.iqa.models.hyperiqa.HyperIQAModel.class);
        registerModel("maniqa", com.autonavi.iqa.models.maniqa.MANIQAModel.class);
        registerModel("musiq", com.autonavi.iqa.models.musiq.MUSIQModel.class);
        registerModel("tres", com.autonavi.iqa.models.tres.TReSModel.class);
        registerModel("clipiqa", com.autonavi.iqa.models.clipiqa.CLIPIQAModel.class);
    }
    
    /**
     * Register a model class
     * 
     * @param modelName Model identifier (e.g., "liqe", "dbcnn")
     * @param modelClass Model implementation class
     */
    public static void registerModel(String modelName, Class<? extends IImageQualityModel> modelClass) {
        if (modelName == null || modelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Model name cannot be null or empty");
        }
        if (modelClass == null) {
            throw new IllegalArgumentException("Model class cannot be null");
        }
        
        models.put(modelName.toLowerCase(), modelClass);
        logger.info("Registered model: {} -> {}", modelName, modelClass.getName());
    }
    
    /**
     * Get model class by name
     * 
     * @param modelName Model identifier
     * @return Model class or null if not found
     */
    public static Class<? extends IImageQualityModel> getModelClass(String modelName) {
        if (modelName == null) {
            return null;
        }
        return models.get(modelName.toLowerCase());
    }
    
    /**
     * Check if a model is registered
     * 
     * @param modelName Model identifier
     * @return true if registered
     */
    public static boolean isRegistered(String modelName) {
        return modelName != null && models.containsKey(modelName.toLowerCase());
    }
    
    /**
     * Get all registered model names
     * 
     * @return Set of model names
     */
    public static Set<String> getAvailableModels() {
        return models.keySet();
    }
    
    /**
     * Get number of registered models
     * 
     * @return Number of models
     */
    public static int getModelCount() {
        return models.size();
    }
}

