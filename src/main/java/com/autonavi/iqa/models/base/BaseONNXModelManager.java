package com.autonavi.iqa.models.base;

import com.autonavi.iqa.common.IModelManager;
import com.autonavi.iqa.common.ModelException;
import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Base class for ONNX model managers
 * Provides common functionality for loading and running ONNX models
 */
public abstract class BaseONNXModelManager implements IModelManager {
    
    protected static final Logger logger = LoggerFactory.getLogger(BaseONNXModelManager.class);
    
    protected OrtEnvironment env;
    protected OrtSession session;
    protected boolean loaded = false;
    
    /**
     * Initialize ONNX Runtime environment
     */
    public BaseONNXModelManager() {
        this.env = OrtEnvironment.getEnvironment();
        logger.info("ONNX Runtime environment initialized");
    }
    
    @Override
    public void loadModel(String modelPath, com.aliyun.odps.udf.ExecutionContext ctx) throws ModelException {
        try {
            // Try to load from ODPS resource first
            if (ctx != null) {
                try {
                    InputStream stream = ctx.readResourceFileAsStream(modelPath);
                    if (stream != null) {
                        loadModelFromStream(stream, modelPath);
                        return;
                    }
                } catch (Exception e) {
                    logger.debug("Could not load from ODPS resource: {}", e.getMessage());
                }
            }
            
            // Try file system paths
            File file = new File(modelPath);
            if (file.exists()) {
                loadModelFromFile(file.getAbsolutePath());
                return;
            }
            
            // Try common locations
            String[] commonPaths = {
                "weights/onnx/" + modelPath,
                "../weights/onnx/" + modelPath,
                "resources/" + modelPath
            };
            
            for (String path : commonPaths) {
                file = new File(path);
                if (file.exists()) {
                    loadModelFromFile(file.getAbsolutePath());
                    return;
                }
            }
            
            throw new ModelException("Model file not found: " + modelPath);
            
        } catch (Exception e) {
            logger.error("Failed to load model: {}", modelPath, e);
            throw new ModelException("Failed to load model: " + e.getMessage(), e);
        }
    }
    
    /**
     * Load model from file path
     */
    protected void loadModelFromFile(String filePath) throws OrtException {
        logger.info("Loading model from: {}", filePath);
        
        OrtSession.SessionOptions opts = createSessionOptions();
        this.session = env.createSession(filePath, opts);
        this.loaded = true;
        
        logger.info("Model loaded successfully");
    }
    
    /**
     * Load model from InputStream
     */
    protected void loadModelFromStream(InputStream inputStream, String modelName) throws OrtException, IOException {
        logger.info("Loading model from InputStream: {}", modelName);
        
        File tempFile = File.createTempFile("model_" + modelName.replace(".", "_"), ".onnx");
        tempFile.deleteOnExit();
        Files.copy(inputStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        
        loadModelFromFile(tempFile.getAbsolutePath());
    }
    
    /**
     * Create session options
     * Subclasses can override to customize
     */
    protected OrtSession.SessionOptions createSessionOptions() {
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
        return opts;
    }
    
    @Override
    public boolean isLoaded() {
        return loaded && session != null;
    }
    
    @Override
    public void close() {
        if (session != null) {
            try {
                session.close();
                session = null;
                loaded = false;
                logger.info("Model closed");
            } catch (Exception e) {
                logger.error("Error closing model", e);
            }
        }
    }
}

