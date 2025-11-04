package com.autonavi.iqa.models.liqe;

import com.autonavi.iqa.common.ModelException;
import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * LIQE-specific ONNX model manager
 * Handles CLIP and LIQE model inference
 * 
 * Note: This class manages two ONNX models (CLIP and LIQE), so it doesn't
 * directly implement IModelManager which assumes a single model.
 * Instead, it provides LIQE-specific methods.
 */
public class LIQEModelManager {
    
    private static final Logger logger = LoggerFactory.getLogger(LIQEModelManager.class);
    
    private OrtEnvironment env;
    private OrtSession clipSession;
    private OrtSession liqeSession;
    private float logitScale = 4.6052f; // Default CLIP logit scale
    
    /**
     * Initialize ONNX Runtime environment
     */
    public LIQEModelManager() {
        this.env = OrtEnvironment.getEnvironment();
        logger.info("ONNX Runtime environment initialized");
    }
    
    /**
     * Load CLIP model
     */
    public void loadClipModel(String modelPath, com.aliyun.odps.udf.ExecutionContext ctx) throws ModelException {
        try {
            // Try ODPS resource first
            if (ctx != null) {
                try {
                    java.io.InputStream stream = ctx.readResourceFileAsStream(modelPath);
                    if (stream != null) {
                        loadClipModelFromStream(stream, modelPath);
                        return;
                    }
                } catch (Exception e) {
                    logger.debug("Could not load from ODPS resource: {}", e.getMessage());
                }
            }
            
            // Try file system
            loadClipModelFromFile(modelPath);
        } catch (Exception e) {
            throw new ModelException("Failed to load CLIP model: " + e.getMessage(), e);
        }
    }
    
    private void loadClipModelFromFile(String filePath) throws OrtException {
        logger.info("Loading CLIP model from: {}", filePath);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            opts.setIntraOpNumThreads(1);
        } catch (Exception e) {
            // Ignore if method doesn't exist
        }
        try {
            opts.setInterOpNumThreads(1);
        } catch (Exception e) {
            // Ignore if method doesn't exist
        }
        this.clipSession = env.createSession(filePath, opts);
        logger.info("CLIP model loaded successfully");
    }
    
    private void loadClipModelFromStream(java.io.InputStream inputStream, String modelName) 
            throws OrtException, java.io.IOException {
        java.io.File tempFile = java.io.File.createTempFile("clip_model", ".onnx");
        tempFile.deleteOnExit();
        java.nio.file.Files.copy(inputStream, tempFile.toPath(), 
            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        loadClipModelFromFile(tempFile.getAbsolutePath());
    }
    
    /**
     * Load LIQE model
     */
    public void loadLiqeModel(String modelPath, com.aliyun.odps.udf.ExecutionContext ctx) throws ModelException {
        try {
            // Try ODPS resource first
            if (ctx != null) {
                try {
                    java.io.InputStream stream = ctx.readResourceFileAsStream(modelPath);
                    if (stream != null) {
                        loadLiqeModelFromStream(stream, modelPath);
                        return;
                    }
                } catch (Exception e) {
                    logger.debug("Could not load from ODPS resource: {}", e.getMessage());
                }
            }
            
            // Try file system
            loadLiqeModelFromFile(modelPath);
        } catch (Exception e) {
            throw new ModelException("Failed to load LIQE model: " + e.getMessage(), e);
        }
    }
    
    private void loadLiqeModelFromFile(String filePath) throws OrtException {
        logger.info("Loading LIQE model from: {}", filePath);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            opts.setIntraOpNumThreads(1);
        } catch (Exception e) {
            // Ignore if method doesn't exist
        }
        try {
            opts.setInterOpNumThreads(1);
        } catch (Exception e) {
            // Ignore if method doesn't exist
        }
        this.liqeSession = env.createSession(filePath, opts);
        logger.info("LIQE model loaded successfully");
    }
    
    private void loadLiqeModelFromStream(java.io.InputStream inputStream, String modelName) 
            throws OrtException, java.io.IOException {
        java.io.File tempFile = java.io.File.createTempFile("liqe_model", ".onnx");
        tempFile.deleteOnExit();
        java.nio.file.Files.copy(inputStream, tempFile.toPath(), 
            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        loadLiqeModelFromFile(tempFile.getAbsolutePath());
    }
    
    /**
     * Run CLIP model inference
     */
    public float[][] runClipModel(float[] imagePatches, int batchSize, int numPatches) throws OrtException {
        if (clipSession == null) {
            throw new IllegalStateException("CLIP model not loaded");
        }
        
        int totalPatches = batchSize * numPatches;
        long[] shape = {totalPatches, 3, 224, 224};
        
        FloatBuffer buffer = FloatBuffer.wrap(imagePatches);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, buffer, shape);
        
        try (OrtSession.Result result = clipSession.run(
                Collections.singletonMap("image", inputTensor)
        )) {
            OnnxValue output = result.get(0);
            float[][] outputArray = (float[][]) output.getValue();
            return outputArray;
        } finally {
            inputTensor.close();
        }
    }
    
    /**
     * Run LIQE model inference
     */
    public float[] runLiqeModel(float[] imagePatches, int batchSize, int numPatches) throws OrtException {
        if (liqeSession == null) {
            throw new IllegalStateException("LIQE model not loaded");
        }
        
        long[] patchesShape = {batchSize, numPatches, 3, 224, 224};
        long[] logitShape = {1};
        
        FloatBuffer patchesBuffer = FloatBuffer.wrap(imagePatches);
        FloatBuffer logitBuffer = FloatBuffer.wrap(new float[]{logitScale});
        
        OnnxTensor patchesTensor = OnnxTensor.createTensor(env, patchesBuffer, patchesShape);
        OnnxTensor logitTensor = OnnxTensor.createTensor(env, logitBuffer, logitShape);
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("image_patches", patchesTensor);
        inputs.put("logit_scale", logitTensor);
        
        try (OrtSession.Result result = liqeSession.run(inputs)) {
            OnnxValue output = result.get(0);
            float[] outputArray = (float[]) output.getValue();
            return outputArray;
        } finally {
            patchesTensor.close();
            logitTensor.close();
        }
    }
    
    public float getLogitScale() {
        return logitScale;
    }
    
    public void setLogitScale(float logitScale) {
        this.logitScale = logitScale;
    }
    
    /**
     * Close all resources
     */
    public void close() {
        try {
            if (clipSession != null) {
                clipSession.close();
                clipSession = null;
            }
            if (liqeSession != null) {
                liqeSession.close();
                liqeSession = null;
            }
            logger.info("LIQE model manager closed");
        } catch (OrtException e) {
            logger.error("Error closing LIQE model sessions", e);
        }
    }
    
    /**
     * Check if both models are loaded
     */
    public boolean isLoaded() {
        return clipSession != null && liqeSession != null;
    }
}

