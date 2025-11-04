package com.autonavi.iqa.models.base;

import com.autonavi.iqa.common.IImageQualityModel;
import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.QualityScore;
import com.autonavi.iqa.common.ModelException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for all IQA models
 * 
 * Provides common functionality and default implementations
 */
public abstract class BaseIQAModel implements IImageQualityModel {
    
    protected static final Logger logger = LoggerFactory.getLogger(BaseIQAModel.class);
    
    protected ModelConfig config;
    protected boolean initialized = false;
    
    @Override
    public void initialize(ModelConfig config) throws ModelException {
        if (config == null) {
            throw new ModelException("Model configuration cannot be null");
        }
        
        this.config = config;
        
        try {
            initializeModel(config);
            this.initialized = true;
            logger.info("Model {} initialized successfully", getModelName());
        } catch (Exception e) {
            logger.error("Failed to initialize model {}", getModelName(), e);
            throw new ModelException("Model initialization failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * Model-specific initialization
     * Subclasses should implement this to load models, setup resources, etc.
     */
    protected abstract void initializeModel(ModelConfig config) throws ModelException;
    
    @Override
    public QualityScore assessQuality(String imageUrl) throws ModelException {
        if (!initialized) {
            throw new ModelException("Model not initialized. Call initialize() first.");
        }
        
        if (imageUrl == null || imageUrl.trim().isEmpty()) {
            return new QualityScore(imageUrl, "Empty image URL");
        }
        
        try {
            double score = assessImageQuality(imageUrl);
            return new QualityScore(score, imageUrl);
        } catch (Exception e) {
            logger.warn("Failed to assess quality for image: {}", imageUrl, e);
            return new QualityScore(imageUrl, e.getMessage());
        }
    }
    
    /**
     * Model-specific quality assessment
     * Subclasses should implement the actual inference logic
     */
    protected abstract double assessImageQuality(String imageUrl) throws ModelException;
    
    @Override
    public QualityScore[] assessQualityBatch(String[] imageUrls) throws ModelException {
        if (!initialized) {
            throw new ModelException("Model not initialized. Call initialize() first.");
        }
        
        if (imageUrls == null || imageUrls.length == 0) {
            return new QualityScore[0];
        }
        
        QualityScore[] results = new QualityScore[imageUrls.length];
        
        // Process in batches to optimize memory usage
        int batchSize = getBatchSize();
        for (int i = 0; i < imageUrls.length; i += batchSize) {
            int end = Math.min(i + batchSize, imageUrls.length);
            processBatch(imageUrls, results, i, end);
        }
        
        return results;
    }
    
    /**
     * Process a batch of images
     */
    protected void processBatch(String[] imageUrls, QualityScore[] results, int start, int end) {
        for (int i = start; i < end; i++) {
            try {
                results[i] = assessQuality(imageUrls[i]);
            } catch (ModelException e) {
                logger.error("Error processing image {}: {}", imageUrls[i], e.getMessage());
                results[i] = new QualityScore(imageUrls[i], e.getMessage());
            }
        }
    }
    
    /**
     * Get batch size for processing
     * Default is 8, subclasses can override
     */
    protected int getBatchSize() {
        Object batchSize = config != null ? config.getParameter("batch_size", Integer.class) : null;
        return batchSize != null ? (Integer) batchSize : 8;
    }
    
    @Override
    public boolean isInitialized() {
        return initialized;
    }
    
    @Override
    public void close() {
        if (initialized) {
            try {
                cleanup();
                initialized = false;
                logger.info("Model {} closed", getModelName());
            } catch (Exception e) {
                logger.error("Error closing model {}", getModelName(), e);
            }
        }
    }
    
    /**
     * Model-specific cleanup
     * Subclasses should override to release resources
     */
    protected void cleanup() {
        // Default: do nothing
    }
}

