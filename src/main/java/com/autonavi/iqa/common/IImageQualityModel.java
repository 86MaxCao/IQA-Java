package com.autonavi.iqa.common;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.QualityScore;
import com.autonavi.iqa.common.ModelException;

/**
 * Interface for all Image Quality Assessment models
 * 
 * This interface allows the framework to support multiple IQA models
 * (LIQE, DBCNN, HyperIQA, etc.) through a unified API.
 * 
 * @author Auto-generated
 * @version 1.0
 */
public interface IImageQualityModel {
    
    /**
     * Initialize the model with configuration
     * 
     * @param config Model configuration (paths, parameters, etc.)
     * @throws ModelException if initialization fails
     */
    void initialize(ModelConfig config) throws ModelException;
    
    /**
     * Assess quality of a single image
     * 
     * @param imageUrl Image URL to assess
     * @return Quality score result
     * @throws ModelException if assessment fails
     */
    QualityScore assessQuality(String imageUrl) throws ModelException;
    
    /**
     * Assess quality of multiple images in batch
     * 
     * This method ensures output length matches input length,
     * even if some images fail to process.
     * 
     * @param imageUrls Array of image URLs
     * @return Array of quality scores (same length as input)
     * @throws ModelException if batch processing fails
     */
    QualityScore[] assessQualityBatch(String[] imageUrls) throws ModelException;
    
    /**
     * Get the model name (e.g., "liqe", "dbcnn", "hyperiqa")
     * 
     * @return Model identifier
     */
    String getModelName();
    
    /**
     * Get the model version
     * 
     * @return Version string
     */
    String getModelVersion();
    
    /**
     * Check if model is initialized
     * 
     * @return true if initialized, false otherwise
     */
    boolean isInitialized();
    
    /**
     * Cleanup model resources
     * Should be called when done with the model
     */
    void close();
}

