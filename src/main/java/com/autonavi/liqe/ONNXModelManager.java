package com.autonavi.liqe;

import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * Manages ONNX models for LIQE inference
 * Handles loading and running CLIP and LIQE models
 */
public class ONNXModelManager {
    
    private static final Logger logger = LoggerFactory.getLogger(ONNXModelManager.class);
    
    private OrtEnvironment env;
    private OrtSession clipSession;
    private OrtSession liqeSession;
    private float logitScale;
    
    /**
     * Initialize ONNX Runtime environment
     */
    public ONNXModelManager() {
        this.env = OrtEnvironment.getEnvironment();
        logger.info("ONNX Runtime environment initialized");
    }
    
    /**
     * Load CLIP model from file path
     */
    public void loadClipModel(String modelPath) throws OrtException {
        logger.info("Loading CLIP model from: {}", modelPath);
        
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // Note: Thread configuration may vary by ONNX Runtime version
        // Try-catch to handle different API versions
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
        
        this.clipSession = env.createSession(modelPath, opts);
        logger.info("CLIP model loaded successfully");
        
        // Get logit scale from model (if available)
        // For now, we'll use a default value or extract from model
        this.logitScale = 4.6052f; // Default CLIP logit scale (exp(4.6052) ≈ 100)
    }
    
    /**
     * Load CLIP model from InputStream (for JAR resources)
     */
    public void loadClipModel(InputStream inputStream) throws OrtException, IOException {
        logger.info("Loading CLIP model from InputStream");
        
        // Copy to temporary file
        File tempFile = File.createTempFile("clip_model", ".onnx");
        tempFile.deleteOnExit();
        Files.copy(inputStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        
        loadClipModel(tempFile.getAbsolutePath());
    }
    
    /**
     * Load LIQE model from file path
     */
    public void loadLiqeModel(String modelPath) throws OrtException {
        logger.info("Loading LIQE model from: {}", modelPath);
        
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // Note: Thread configuration may vary by ONNX Runtime version
        // Try-catch to handle different API versions
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
        
        this.liqeSession = env.createSession(modelPath, opts);
        logger.info("LIQE model loaded successfully");
    }
    
    /**
     * Load LIQE model from InputStream (for JAR resources)
     */
    public void loadLiqeModel(InputStream inputStream) throws OrtException, IOException {
        logger.info("Loading LIQE model from InputStream");
        
        // Copy to temporary file
        File tempFile = File.createTempFile("liqe_model", ".onnx");
        tempFile.deleteOnExit();
        Files.copy(inputStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        
        loadLiqeModel(tempFile.getAbsolutePath());
    }
    
    /**
     * Run CLIP model inference on image patches
     * Input: [batch_size * num_patches, 3, 224, 224]
     * Output: [batch_size * num_patches, feature_dim]
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
     * Input: image_patches [batch_size, num_patches, 3, 224, 224], logit_scale [1]
     * Output: quality_score [batch_size]
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
    
    /**
     * Get logit scale value
     */
    public float getLogitScale() {
        return logitScale;
    }
    
    /**
     * Set logit scale value
     */
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
            }
            if (liqeSession != null) {
                liqeSession.close();
            }
        } catch (OrtException e) {
            logger.error("Error closing ONNX sessions: {}", e.getMessage());
        }
    }
}

