package com.autonavi.liqe;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * Load text features from JSON file
 */
public class TextFeatureLoader {
    
    private static final Logger logger = LoggerFactory.getLogger(TextFeatureLoader.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    
    /**
     * Load text features from JSON file
     * JSON format: {"shape": [rows, cols], "dtype": "float32", "data": [[...], [...]]}
     */
    public static float[][] loadTextFeatures(String jsonPath) throws IOException {
        logger.info("Loading text features from: {}", jsonPath);
        
        JsonNode root = objectMapper.readTree(new File(jsonPath));
        
        // Get shape
        JsonNode shapeNode = root.get("shape");
        int rows = shapeNode.get(0).asInt();
        int cols = shapeNode.get(1).asInt();
        
        logger.info("Text features shape: {} x {}", rows, cols);
        
        // Get data
        JsonNode dataNode = root.get("data");
        float[][] textFeatures = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            JsonNode rowNode = dataNode.get(i);
            for (int j = 0; j < cols; j++) {
                textFeatures[i][j] = (float) rowNode.get(j).asDouble();
            }
        }
        
        logger.info("Successfully loaded text features");
        return textFeatures;
    }
    
    /**
     * Load text features from InputStream (for loading from JAR resources)
     */
    public static float[][] loadTextFeatures(InputStream inputStream) throws IOException {
        logger.info("Loading text features from InputStream");
        
        JsonNode root = objectMapper.readTree(inputStream);
        
        // Get shape
        JsonNode shapeNode = root.get("shape");
        int rows = shapeNode.get(0).asInt();
        int cols = shapeNode.get(1).asInt();
        
        logger.info("Text features shape: {} x {}", rows, cols);
        
        // Get data
        JsonNode dataNode = root.get("data");
        float[][] textFeatures = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            JsonNode rowNode = dataNode.get(i);
            for (int j = 0; j < cols; j++) {
                textFeatures[i][j] = (float) rowNode.get(j).asDouble();
            }
        }
        
        logger.info("Successfully loaded text features");
        return textFeatures;
    }
    
    /**
     * Convert text features to ONNX input format (transpose for matrix multiplication)
     * Input: [rows, cols] - text features
     * Output: [cols, rows] - transposed for ONNX MatMul
     */
    public static float[] textFeaturesToOnnxInput(float[][] textFeatures) {
        int rows = textFeatures.length;
        int cols = textFeatures[0].length;
        
        // Transpose: [rows, cols] -> [cols, rows]
        float[] result = new float[cols * rows];
        int idx = 0;
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                result[idx++] = textFeatures[i][j];
            }
        }
        
        return result;
    }
}

