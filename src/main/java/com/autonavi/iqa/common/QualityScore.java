package com.autonavi.iqa.common;

/**
 * Quality score result wrapper
 * 
 * Contains the score value and metadata about the assessment
 */
public class QualityScore {
    
    private final double score;
    private final String imageUrl;
    private final boolean success;
    private final String errorMessage;
    
    // Quality score range constants
    public static final double ERROR_SCORE = -1.0;
    public static final double MIN_SCORE = 1.0;
    public static final double MAX_SCORE = 5.0;
    
    /**
     * Create successful quality score result
     * 
     * @param score Quality score (typically 1.0-5.0)
     * @param imageUrl Image URL that was assessed
     */
    public QualityScore(double score, String imageUrl) {
        this.score = score;
        this.imageUrl = imageUrl;
        this.success = true;
        this.errorMessage = null;
    }
    
    /**
     * Create error result
     * 
     * @param imageUrl Image URL that failed
     * @param errorMessage Error description
     */
    public QualityScore(String imageUrl, String errorMessage) {
        this.score = ERROR_SCORE;
        this.imageUrl = imageUrl;
        this.success = false;
        this.errorMessage = errorMessage;
    }
    
    /**
     * Get quality score
     * Returns ERROR_SCORE (-1.0) if assessment failed
     * 
     * @return Quality score
     */
    public double getScore() {
        return score;
    }
    
    /**
     * Get image URL
     * 
     * @return Image URL
     */
    public String getImageUrl() {
        return imageUrl;
    }
    
    /**
     * Check if assessment was successful
     * 
     * @return true if successful, false if error occurred
     */
    public boolean isSuccess() {
        return success;
    }
    
    /**
     * Get error message (if any)
     * 
     * @return Error message or null if successful
     */
    public String getErrorMessage() {
        return errorMessage;
    }
    
    /**
     * Check if score indicates an error
     * 
     * @return true if score is ERROR_SCORE
     */
    public boolean isError() {
        return score == ERROR_SCORE;
    }
    
    @Override
    public String toString() {
        if (success) {
            return String.format("QualityScore{score=%.4f, url=%s}", score, imageUrl);
        } else {
            return String.format("QualityScore{error=%s, url=%s}", errorMessage, imageUrl);
        }
    }
}

