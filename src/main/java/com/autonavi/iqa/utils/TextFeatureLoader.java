package com.autonavi.iqa.utils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * Loads precomputed text features from JSON files for models like LIQE and CLIPIQA.
 *
 * <p>Expected JSON format:
 * <pre>
 * {
 *   "shape": [rows, cols],
 *   "dtype": "float32",
 *   "data": [[...], [...], ...]
 * }
 * </pre>
 */
public class TextFeatureLoader {

    private static final Logger logger = LoggerFactory.getLogger(TextFeatureLoader.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Load text features from a JSON file on disk.
     *
     * @param jsonPath path to the JSON file
     * @return 2D float array with shape [rows][cols]
     * @throws IOException if the file cannot be read or parsed
     */
    public static float[][] loadTextFeatures(String jsonPath) throws IOException {
        logger.info("Loading text features from: {}", jsonPath);
        JsonNode root = objectMapper.readTree(new File(jsonPath));
        return parseTextFeatures(root);
    }

    /**
     * Load text features from an InputStream (e.g. ODPS volume resource).
     *
     * @param inputStream input stream containing JSON data
     * @return 2D float array with shape [rows][cols]
     * @throws IOException if the stream cannot be read or parsed
     */
    public static float[][] loadTextFeatures(InputStream inputStream) throws IOException {
        logger.info("Loading text features from InputStream");
        JsonNode root = objectMapper.readTree(inputStream);
        return parseTextFeatures(root);
    }

    /**
     * Transpose text features for ONNX MatMul: [rows, cols] → flattened [cols * rows] in column-major order.
     *
     * @param textFeatures 2D array with shape [rows][cols]
     * @return flattened 1D array in column-major order, length = rows * cols
     */
    public static float[] textFeaturesToOnnxInput(float[][] textFeatures) {
        int rows = textFeatures.length;
        int cols = textFeatures[0].length;
        float[] result = new float[cols * rows];
        int idx = 0;
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                result[idx++] = textFeatures[i][j];
            }
        }
        return result;
    }

    private static float[][] parseTextFeatures(JsonNode root) {
        JsonNode shapeNode = root.get("shape");
        int rows = shapeNode.get(0).asInt();
        int cols = shapeNode.get(1).asInt();
        logger.info("Text features shape: {} x {}", rows, cols);

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
}
