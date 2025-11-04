package com.autonavi.liqe;

import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

/**
 * Test class for LIQE model inference with URL support
 * Can be used for local testing without ODPS UDTF framework
 */
public class LIQETest {
    
    private static final Logger logger = LoggerFactory.getLogger(LIQETest.class);
    
    // Default test URL
    private static final String DEFAULT_TEST_URL = 
        "http://dzlk-cloud.oss-cn-zhangjiakou-gd.aliyuncs.com/zlkoss/socol/20250507/fc673e41c5384cdeb23b3cfd21ff2657/52196787/2025050714/1746598676496_143014e2-2b80-4124-b1c3-e57da2c43373.jpg";
    
    public static void main(String[] args) {
        String[] imageInputs;
        
        if (args.length >= 4) {
            // Full arguments: clip_model.onnx liqe_model.onnx text_features.json image_url
            String clipModelPath = args[0];
            String liqeModelPath = args[1];
            String textFeaturesPath = args[2];
            imageInputs = Arrays.copyOfRange(args, 3, args.length);
            
            runTest(clipModelPath, liqeModelPath, textFeaturesPath, imageInputs);
        } else if (args.length == 1) {
            // Single argument: assume it's a model directory, use default test URL
            String modelDir = args[0];
            String clipModelPath = modelDir + "/clip_model.onnx";
            String liqeModelPath = modelDir + "/liqe_model.onnx";
            String textFeaturesPath = modelDir + "/text_features.json";
            
            imageInputs = new String[]{DEFAULT_TEST_URL};
            
            runTest(clipModelPath, liqeModelPath, textFeaturesPath, imageInputs);
        } else {
            System.out.println("Usage:");
            System.out.println("  LIQETest <clip_model.onnx> <liqe_model.onnx> <text_features.json> <image_url> [image_url2 ...]");
            System.out.println("  OR");
            System.out.println("  LIQETest <model_directory>  # Uses default test URL");
            System.out.println("");
            System.out.println("Example:");
            System.out.println("  LIQETest weights/onnx/clip_model.onnx weights/onnx/liqe_model.onnx weights/onnx/text_features.json http://example.com/image.jpg");
            System.out.println("  OR");
            System.out.println("  LIQETest weights/onnx  # Uses default test URL");
            System.exit(1);
        }
    }
    
    private static void runTest(String clipModelPath, String liqeModelPath, 
                                String textFeaturesPath, String[] imageInputs) {
        ONNXModelManager modelManager = null;
        ImageDownloader imageDownloader = null;
        
        try {
            System.out.println(repeatString("=", 60));
            System.out.println("LIQE Model Test");
            System.out.println(repeatString("=", 60));
            System.out.println();
            
            logger.info("Initializing LIQE model...");
            
            // Load models
            System.out.println("Loading models...");
            modelManager = new ONNXModelManager();
            modelManager.loadClipModel(clipModelPath);
            System.out.println("✓ CLIP model loaded");
            
            modelManager.loadLiqeModel(liqeModelPath);
            System.out.println("✓ LIQE model loaded");
            
            // Load text features (for verification)
            float[][] textFeatures = TextFeatureLoader.loadTextFeatures(textFeaturesPath);
            System.out.println("✓ Text features loaded: " + textFeatures.length + " x " + textFeatures[0].length);
            System.out.println();
            
            // Initialize image downloader
            imageDownloader = new ImageDownloader();
            
            // Process each image
            System.out.println("Processing " + imageInputs.length + " image(s)...");
            System.out.println();
            
            for (int i = 0; i < imageInputs.length; i++) {
                String imageInput = imageInputs[i];
                System.out.println("[" + (i + 1) + "/" + imageInputs.length + "] " + imageInput);
                
                try {
                    Mat image = null;
                    
                    // Check if it's a URL or local file path
                    if (imageInput.startsWith("http://") || imageInput.startsWith("https://")) {
                        System.out.println("  Downloading image...");
                        image = imageDownloader.downloadImage(imageInput);
                    } else {
                        System.out.println("  Loading image from file...");
                        image = imread(imageInput);
                    }
                    
                    if (image == null || image.empty()) {
                        System.out.println("  ✗ Failed to load image");
                        System.out.println("  Quality Score: -1.0 (ERROR)");
                        System.out.println();
                        continue;
                    }
                    
                    System.out.println("  Image size: " + image.cols() + " x " + image.rows());
                    
                    // Process image
                    System.out.println("  Processing image...");
                    double score = processImage(image, modelManager);
                    
                    System.out.println("  Quality Score: " + String.format("%.4f", score));
                    if (score < 0) {
                        System.out.println("  Status: ERROR");
                    } else if (score >= 1.0 && score <= 5.0) {
                        System.out.println("  Status: SUCCESS");
                    } else {
                        System.out.println("  Status: WARNING (score out of range)");
                    }
                    System.out.println();
                    
                    image.close();
                    
                } catch (Exception e) {
                    System.out.println("  ✗ Error: " + e.getMessage());
                    e.printStackTrace();
                    System.out.println("  Quality Score: -1.0 (ERROR)");
                    System.out.println();
                }
            }
            
            System.out.println(repeatString("=", 60));
            System.out.println("Test completed!");
            System.out.println(repeatString("=", 60));
            
        } catch (Exception e) {
            System.err.println("\n✗ Error in test: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } finally {
            if (modelManager != null) {
                modelManager.close();
            }
            if (imageDownloader != null) {
                imageDownloader.close();
            }
        }
    }
    
    /**
     * Process a single image and return quality score
     */
    private static double processImage(Mat image, ONNXModelManager modelManager) throws Exception {
        // Resize image
        Mat resized = ImagePreprocessor.resizeImage(image);
        
        try {
            // Extract patches
            List<float[][][][]> allPatches = ImagePreprocessor.extractPatches(resized);
            List<float[][][][]> selectedPatches = ImagePreprocessor.selectPatches(allPatches);
            
            if (selectedPatches.isEmpty()) {
                throw new RuntimeException("No patches extracted from image");
            }
            
            System.out.println("    Extracted " + allPatches.size() + " patches, selected " + selectedPatches.size());
            
            // Normalize patches
            float[][][][] patchesArray = new float[selectedPatches.size()][][][];
            for (int j = 0; j < selectedPatches.size(); j++) {
                patchesArray[j] = selectedPatches.get(j)[0]; // Get [3, 224, 224]
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
            System.out.println("    Running LIQE model inference...");
            float[] qualityScores = modelManager.runLiqeModel(onnxInput, 1, numPatches);
            
            if (qualityScores.length > 0) {
                double score = qualityScores[0];
                if (score >= 1.0 && score <= 5.0) {
                    return score;
                } else {
                    logger.warn("Score {} out of valid range [1.0, 5.0]", score);
                    return -1.0;
                }
            } else {
                throw new RuntimeException("No output from model");
            }
            
        } finally {
            resized.close();
        }
    }
    
    /**
     * Repeat a string n times (Java 8 compatible replacement for String.repeat())
     */
    private static String repeatString(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
}
