package com.autonavi.iqa.common;

/**
 * Exception thrown by IQA models
 */
public class ModelException extends Exception {
    
    public ModelException(String message) {
        super(message);
    }
    
    public ModelException(String message, Throwable cause) {
        super(message, cause);
    }
}

