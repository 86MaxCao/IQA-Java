package com.autonavi.iqa.models.liqe;

import com.autonavi.iqa.models.base.BaseIQAModel;
import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.common.QualityScore;
import com.autonavi.iqa.utils.ImageDownloader;
import com.autonavi.iqa.utils.ImagePreprocessor;
import com.autonavi.iqa.utils.TextFeatureLoader;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * LIQE model implementation
 * Implements the IImageQualityModel interface for LIQE-specific logic
 */
public class LIQEModel extends BaseIQAModel {
    
    private static final Logger logger = LoggerFactory.getLogger(LIQEModel.class);
    
    private LIQEModelManager modelManager;
    private ImageDownloader imageDownloader;
    private float[][] textFeatures;
    
    @Override
    protected void initializeModel(ModelConfig config) throws ModelException {
        try {
            // Initialize model manager
            modelManager = new LIQEModelManager();
            
            // Load models from config
            String clipModelPath = config.getModelPath("clip_model");
            String liqeModelPath = config.getModelPath("liqe_model");
            String textFeaturesPath = config.getModelPath("text_features");
            
            if (clipModelPath == null || liqeModelPath == null || textFeaturesPath == null) {
                throw new ModelException("Missing required model paths in config: clip_model, liqe_model, text_features");
            }
            
            // Load ONNX models
            modelManager.loadClipModel(clipModelPath, config.getExecutionContext());
            modelManager.loadLiqeModel(liqeModelPath, config.getExecutionContext());
            
            // Load text features
            textFeatures = TextFeatureLoader.loadTextFeatures(textFeaturesPath);
            
            // Initialize image downloader
            imageDownloader = new ImageDownloader();
            
            logger.info("LIQE model initialized successfully");
            
        } catch (Exception e) {
            logger.error("Failed to initialize LIQE model", e);
            throw new ModelException("LIQE model initialization failed: " + e.getMessage(), e);
        }
    }
    
    @Override
    protected double assessImageQuality(String imageUrl) throws ModelException {
        Mat image = null;
        try {
            // Download image
            image = imageDownloader.downloadImage(imageUrl);
            if (image == null || image.empty()) {
                throw new ModelException("Failed to download or decode image: " + imageUrl);
            }
            
            // Resize image
            Mat resized = ImagePreprocessor.resizeImage(image);
            image.close();
            
            try {
                // Extract patches
                List<float[][][][]> allPatches = ImagePreprocessor.extractPatches(resized);
                List<float[][][][]> selectedPatches = ImagePreprocessor.selectPatches(allPatches);
                
                if (selectedPatches.isEmpty()) {
                    throw new ModelException("No patches extracted from image");
                }
                
                // Normalize patches
                float[][][][] patchesArray = new float[selectedPatches.size()][][][];
                for (int j = 0; j < selectedPatches.size(); j++) {
                    patchesArray[j] = selectedPatches.get(j)[0];
                }
                float[][][][] normalizedPatches = ImagePreprocessor.normalizePatches(patchesArray);
                
                // Convert to ONNX input format
                List<float[][][][]> normalizedList = new ArrayList<>();
                for (int p = 0; p < normalizedPatches.length; p++) {
                    float[][][][] wrappedPatch = new float[1][][][];
                    wrappedPatch[0] = normalizedPatches[p];
                    normalizedList.add(wrappedPatch);
                }
                float[] onnxInput = ImagePreprocessor.patchesToOnnxInput(normalizedList);
                
                // Run inference
                int numPatches = selectedPatches.size();
                float[] qualityScores = modelManager.runLiqeModel(onnxInput, 1, numPatches);
                
                if (qualityScores.length > 0) {
                    float score = qualityScores[0];
                    if (score >= 1.0f && score <= 5.0f) {
                        return score;
                    } else {
                        throw new ModelException("Score out of range: " + score);
                    }
                } else {
                    throw new ModelException("Empty inference result");
                }
                
            } finally {
                if (resized != null && !resized.empty()) {
                    resized.close();
                }
            }
            
        } catch (Exception e) {
            logger.error("Error assessing image quality: {}", imageUrl, e);
            throw new ModelException("Failed to assess image quality: " + e.getMessage(), e);
        } finally {
            if (image != null && !image.empty()) {
                image.close();
            }
        }
    }
    
    @Override
    public String getModelName() {
        return "liqe";
    }
    
    @Override
    public String getModelVersion() {
        return "1.0";
    }
    
    @Override
    protected void cleanup() {
        if (modelManager != null) {
            modelManager.close();
            modelManager = null;
        }
        if (imageDownloader != null) {
            imageDownloader.close();
            imageDownloader = null;
        }
    }
}

