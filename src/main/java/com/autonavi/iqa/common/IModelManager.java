package com.autonavi.iqa.common;

/**
 * Interface for ONNX model management
 * 
 * Handles loading and running ONNX models for inference
 */
public interface IModelManager {
    
    /**
     * Load model from path or ODPS resource
     * 
     * @param modelPath Path to model file or ODPS resource name
     * @param ctx ODPS ExecutionContext (may be null for local testing)
     * @throws ModelException if model loading fails
     */
    void loadModel(String modelPath, com.aliyun.odps.udf.ExecutionContext ctx) throws ModelException;
    
    /**
     * Run inference on input data
     * 
     * @param input Input data (shape depends on model)
     * @return Inference output
     * @throws ModelException if inference fails
     */
    float[] runInference(float[] input) throws ModelException;
    
    /**
     * Run batch inference
     * 
     * @param inputs Array of input arrays
     * @return Array of output arrays
     * @throws ModelException if inference fails
     */
    float[][] runBatchInference(float[][] inputs) throws ModelException;
    
    /**
     * Check if model is loaded
     * 
     * @return true if model is loaded and ready
     */
    boolean isLoaded();
    
    /**
     * Close model and release resources
     */
    void close();
}

