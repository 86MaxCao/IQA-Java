package com.autonavi.iqa.common;

import java.util.Map;
import java.util.HashMap;

/**
 * Configuration for an IQA model
 * 
 * Contains model paths, parameters, and other settings
 */
public class ModelConfig {
    
    private String modelName;
    private String modelVersion;
    private Map<String, String> modelPaths;
    private Map<String, Object> parameters;
    private com.aliyun.odps.udf.ExecutionContext executionContext;
    
    public ModelConfig() {
        this.modelPaths = new HashMap<>();
        this.parameters = new HashMap<>();
    }
    
    public ModelConfig(String modelName) {
        this();
        this.modelName = modelName;
    }
    
    // Getters and setters
    public String getModelName() {
        return modelName;
    }
    
    public void setModelName(String modelName) {
        this.modelName = modelName;
    }
    
    public String getModelVersion() {
        return modelVersion;
    }
    
    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }
    
    public Map<String, String> getModelPaths() {
        return modelPaths;
    }
    
    public void setModelPaths(Map<String, String> modelPaths) {
        this.modelPaths = modelPaths;
    }
    
    public String getModelPath(String key) {
        return modelPaths.get(key);
    }
    
    public void setModelPath(String key, String path) {
        modelPaths.put(key, path);
    }
    
    public Map<String, Object> getParameters() {
        return parameters;
    }
    
    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }
    
    @SuppressWarnings("unchecked")
    public <T> T getParameter(String key, Class<T> type) {
        Object value = parameters.get(key);
        if (value != null && type.isInstance(value)) {
            return (T) value;
        }
        return null;
    }
    
    public void setParameter(String key, Object value) {
        parameters.put(key, value);
    }
    
    public com.aliyun.odps.udf.ExecutionContext getExecutionContext() {
        return executionContext;
    }
    
    public void setExecutionContext(com.aliyun.odps.udf.ExecutionContext executionContext) {
        this.executionContext = executionContext;
    }
}

