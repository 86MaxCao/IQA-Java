package com.autonavi.liqe;

import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.UDTFCollector;
import com.aliyun.odps.udf.annotation.Resolve;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.*;

/**
 * LIQE UDTF for ODPS
 * 
 * Input: array<string> - array of image URLs
 * Output: string (URL), double (quality score)
 * 
 * Quality score range: 1.0 - 5.0
 * Error indicator: -1.0 (for failed cases)
 */
@Resolve("array<string> -> string, double")
public class UDTFLIQE extends UDTF {
    
    private static final Logger logger = LoggerFactory.getLogger(UDTFLIQE.class);
    private static final double ERROR_SCORE = -1.0;
    private static final int BATCH_SIZE = 10;
    
    private ONNXModelManager modelManager;
    private ImageDownloader imageDownloader;
    private boolean initialized = false;
    private int counter = 0;
    
    /**
     * Initialize native libraries (OpenCV, OpenBLAS) for ODPS environment
     * This ensures native libraries are loaded before any OpenCV operations
     * 
     * JavaCPP/JavaCV should automatically extract and load native libraries from JAR,
     * but we need to ensure this happens early and handle ODPS-specific issues.
     * 
     * NOTE: In ODPS, native libraries may fail to load due to restrictions.
     * We use lazy loading strategy: initialize here but don't fail if it doesn't work,
     * let the actual usage fail with a clearer error message.
     */
    private static void initializeNativeLibraries() throws com.aliyun.odps.udf.UDFException {
        try {
            // Set JavaCPP library path to use temp directory
            // This is important for ODPS where native libraries need to be extracted
            String tempDir = System.getProperty("java.io.tmpdir");
            if (tempDir != null) {
                System.setProperty("org.bytedeco.javacpp.cachedir", tempDir);
                logger.debug("Set JavaCPP cache directory to: {}", tempDir);
            }
            
            // Set library path to include temp directory
            String currentLibPath = System.getProperty("java.library.path", "");
            if (tempDir != null && !currentLibPath.contains(tempDir)) {
                String newLibPath = currentLibPath + (currentLibPath.isEmpty() ? "" : ":") + tempDir;
                System.setProperty("java.library.path", newLibPath);
                logger.debug("Updated library path to include: {}", tempDir);
            }
            
            // Try to load OpenBLAS first (dependency of OpenCV)
            // Use lazy loading - don't fail immediately, let it fail when actually used
            boolean openblasLoaded = false;
            try {
                // Try to trigger OpenBLAS loading
                Class<?> openblasClass = Class.forName("org.bytedeco.openblas.global.openblas_nolapack");
                // Try to access a static field to trigger initialization
                try {
                    java.lang.reflect.Field field = openblasClass.getDeclaredField("INSTANCE");
                    field.setAccessible(true);
                    field.get(null); // Access to trigger static initialization
                    openblasLoaded = true;
                    logger.info("OpenBLAS loaded successfully");
                } catch (NoSuchFieldException | IllegalAccessException e) {
                    // Field doesn't exist or can't access, try Class.forName only
                    Class.forName("org.bytedeco.openblas.global.openblas_nolapack");
                    openblasLoaded = true;
                    logger.info("OpenBLAS loaded via Class.forName");
                }
            } catch (ClassNotFoundException e) {
                logger.warn("OpenBLAS classes not found in classpath: {}", e.getMessage());
            } catch (UnsatisfiedLinkError e) {
                // Native library failed to load - this is OK for now, will fail when actually used
                logger.warn("OpenBLAS native library could not be loaded during initialization: {}", e.getMessage());
                logger.warn("This is OK - will try lazy loading when actually needed.");
                logger.warn("Library path: {}", System.getProperty("java.library.path"));
                // Don't throw here - let it fail when actually used with a clearer error
            }
            
            // Try to load OpenCV core (depends on OpenBLAS)
            // Again, use lazy loading strategy
            boolean opencvLoaded = false;
            try {
                Class.forName("org.bytedeco.opencv.global.opencv_core");
                opencvLoaded = true;
                logger.info("OpenCV core classes loaded successfully");
            } catch (ClassNotFoundException e) {
                logger.warn("OpenCV core classes not found: {}", e.getMessage());
            } catch (UnsatisfiedLinkError e) {
                logger.warn("OpenCV native library could not be loaded during initialization: {}", e.getMessage());
                logger.warn("This is OK - will try lazy loading when actually needed.");
            }
            
            // Log status
            if (!openblasLoaded || !opencvLoaded) {
                logger.warn("Some native libraries could not be loaded during initialization. " +
                    "This may be OK if they are loaded lazily when needed.");
            }
            
        } catch (Exception e) {
            // Log warning but don't fail - let actual usage fail if libraries are truly missing
            logger.warn("Warning during native library initialization: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Initialize UDTF - load models and resources
     */
    @Override
    public void setup(com.aliyun.odps.udf.ExecutionContext ctx) throws com.aliyun.odps.udf.UDFException {
        try {
            logger.info("Initializing LIQE UDTF...");
            
            // Initialize native libraries first (OpenCV, OpenBLAS)
            // This must happen before any OpenCV operations
            // Use lazy loading - don't fail if libraries can't be loaded now
            try {
                initializeNativeLibraries();
            } catch (com.aliyun.odps.udf.UDFException e) {
                // Re-throw UDFException
                throw e;
            } catch (Exception e) {
                // Log warning but continue - libraries will be loaded lazily when needed
                logger.warn("Native library initialization had issues, but continuing: {}", e.getMessage());
            }
            
            // Set collector (available via getCollector() method)
            // Collector will be set by ODPS framework
            
            // Initialize model manager
            modelManager = new ONNXModelManager();
            
            // Load models from ODPS resources or file paths
            // Priority:
            // 1. System properties (for testing)
            // 2. ODPS resource files (via ExecutionContext)
            // 3. File system paths
            
            String clipModelPath = null;
            String liqeModelPath = null;
            String textFeaturesPath = null;
            
            // Try to get from system properties first (for local testing)
            clipModelPath = System.getProperty("liqe.clip_model.path");
            liqeModelPath = System.getProperty("liqe.liqe_model.path");
            textFeaturesPath = System.getProperty("liqe.text_features.path");
            
            // If not set via system properties, try to get from ODPS resources
            if (clipModelPath == null || liqeModelPath == null || textFeaturesPath == null) {
                try {
                    // Try to read from ODPS resource files
                    // In ODPS, resources are available via ExecutionContext
                    // Resource files should be added via ADD FILE command
                    clipModelPath = getResourcePath("clip_model.onnx", ctx);
                    liqeModelPath = getResourcePath("liqe_model.onnx", ctx);
                    textFeaturesPath = getResourcePath("text_features.json", ctx);
                } catch (Exception e) {
                    logger.debug("Could not get resource paths from ExecutionContext: {}", e.getMessage());
                }
            }
            
            // If still not found, try file system paths (for local testing)
            if (clipModelPath == null) {
                clipModelPath = getResourcePath("clip_model.onnx", null);
            }
            if (liqeModelPath == null) {
                liqeModelPath = getResourcePath("liqe_model.onnx", null);
            }
            if (textFeaturesPath == null) {
                textFeaturesPath = getResourcePath("text_features.json", null);
            }
            
            // Validate and load models
            if (clipModelPath == null || liqeModelPath == null || textFeaturesPath == null) {
                throw new RuntimeException("Cannot find model files. Please ensure they are uploaded as ODPS resources " +
                    "or available in file system. Expected: clip_model.onnx, liqe_model.onnx, text_features.json");
            }
            
            // Resolve model paths - handle both file paths and ODPS resource names
            String resolvedClipPath = resolveModelPath(clipModelPath, "clip_model.onnx", ctx);
            String resolvedLiqePath = resolveModelPath(liqeModelPath, "liqe_model.onnx", ctx);
            String resolvedTextPath = resolveModelPath(textFeaturesPath, "text_features.json", ctx);
            
            // Verify files exist
            java.io.File clipFile = new java.io.File(resolvedClipPath);
            java.io.File liqeFile = new java.io.File(resolvedLiqePath);
            java.io.File textFile = new java.io.File(resolvedTextPath);
            
            if (!clipFile.exists() || !liqeFile.exists() || !textFile.exists()) {
                throw new RuntimeException("Model files not found. " +
                    "clip_model.onnx exists: " + clipFile.exists() + " (" + resolvedClipPath + "), " +
                    "liqe_model.onnx exists: " + liqeFile.exists() + " (" + resolvedLiqePath + "), " +
                    "text_features.json exists: " + textFile.exists() + " (" + resolvedTextPath + ")");
            }
            
            // Load models from file paths
            modelManager.loadClipModel(resolvedClipPath);
            modelManager.loadLiqeModel(resolvedLiqePath);
            logger.info("Models loaded from paths: clip={}, liqe={}, text={}", 
                resolvedClipPath, resolvedLiqePath, resolvedTextPath);
            
            // Initialize image downloader
            imageDownloader = new ImageDownloader();
            
            initialized = true;
            logger.info("LIQE UDTF initialized successfully");
            
        } catch (Exception e) {
            logger.error("Error initializing LIQE UDTF: {}", e.getMessage(), e);
            initialized = false;
            throw new com.aliyun.odps.udf.UDFException("Failed to initialize LIQE UDTF: " + e.getMessage());
        }
    }
    
    /**
     * Convert input argument to String array
     * Supports both String[] and List<String> (ArrayList from COLLECT_LIST)
     */
    private String[] convertToStringArray(Object input) {
        if (input == null) {
            return new String[0];
        }
        
        // Case 1: Already a String array
        if (input instanceof String[]) {
            return (String[]) input;
        }
        
        // Case 2: List (ArrayList from COLLECT_LIST)
        if (input instanceof List) {
            @SuppressWarnings("unchecked")
            List<String> list = (List<String>) input;
            return list.toArray(new String[list.size()]);
        }
        
        // Case 3: Try to convert Object array to String array
        if (input.getClass().isArray()) {
            Object[] objArray = (Object[]) input;
            String[] strArray = new String[objArray.length];
            for (int i = 0; i < objArray.length; i++) {
                strArray[i] = objArray[i] != null ? objArray[i].toString() : null;
            }
            return strArray;
        }
        
        // Fallback: convert to string array with single element
        logger.warn("Unexpected input type: {}, converting to string array", input.getClass().getName());
        return new String[]{input.toString()};
    }
    
    /**
     * Process batch of image URLs
     * IMPORTANT: Must ensure output count equals input count
     * Supports both String[] and List<String> (from COLLECT_LIST)
     */
    @Override
    public void process(Object[] args) throws com.aliyun.odps.udf.UDFException, java.io.IOException {
        // Convert input to String array (supports both String[] and ArrayList)
        String[] imageUrls = convertToStringArray(args[0]);
        
        if (!initialized) {
            logger.error("UDTF not initialized, outputting error scores for all URLs");
            for (String url : imageUrls) {
                try {
                    forward(url, ERROR_SCORE);
                } catch (com.aliyun.odps.udf.UDFException e) {
                    logger.error("Error forwarding error result for {}: {}", url, e.getMessage());
                }
            }
            return;
        }
        
        Set<String> processedUrls = new HashSet<>();
        
        try {
            counter++;
            logger.info("Processing batch #{}: {} URLs", counter, imageUrls.length);
            
            // Process in batches
            for (int batchStart = 0; batchStart < imageUrls.length; batchStart += BATCH_SIZE) {
                int batchEnd = Math.min(batchStart + BATCH_SIZE, imageUrls.length);
                String[] batchUrls = Arrays.copyOfRange(imageUrls, batchStart, batchEnd);
                
                processBatch(batchUrls, processedUrls);
            }
            
            // Final check: ensure all URLs have been processed
            for (String url : imageUrls) {
                if (!processedUrls.contains(url)) {
                    logger.warn("URL {} was not processed, outputting error score", url);
                    try {
                        forward(url, ERROR_SCORE);
                    } catch (com.aliyun.odps.udf.UDFException e) {
                        logger.error("Error forwarding error result for {}: {}", url, e.getMessage());
                    }
                    processedUrls.add(url);
                }
            }
            
        } catch (Exception e) {
            logger.error("Error in process method: {}", e.getMessage(), e);
            // Output error scores for all unprocessed URLs
            for (String url : imageUrls) {
                if (!processedUrls.contains(url)) {
                    try {
                        forward(url, ERROR_SCORE);
                    } catch (com.aliyun.odps.udf.UDFException ex) {
                        logger.error("Error forwarding error result for {}: {}", url, ex.getMessage());
                    }
                    processedUrls.add(url);
                }
            }
        }
    }
    
    /**
     * Process a single batch of URLs
     */
    private void processBatch(String[] batchUrls, Set<String> processedUrls) {
        List<String> validUrls = new ArrayList<>();
        List<Mat> images = new ArrayList<>();
        
        // Download and preprocess images
        for (String url : batchUrls) {
            try {
                Mat image = imageDownloader.downloadImage(url);
                if (image != null && !image.empty()) {
                    // Resize image
                    Mat resized = ImagePreprocessor.resizeImage(image);
                    images.add(resized);
                    validUrls.add(url);
                    image.close();
                } else {
                    logger.warn("Failed to download or decode image: {}", url);
                    validUrls.add(url); // Still add to list to maintain order
                    images.add(null); // Mark as invalid
                }
            } catch (Exception e) {
                logger.error("Error downloading image {}: {}", url, e.getMessage());
                validUrls.add(url);
                images.add(null);
            }
        }
        
        // Process valid images
        if (validUrls.isEmpty()) {
            // All failed, output error scores
            for (String url : batchUrls) {
                try {
                    forward(url, ERROR_SCORE);
                } catch (com.aliyun.odps.udf.UDFException e) {
                    logger.error("Error forwarding error result for {}: {}", url, e.getMessage());
                }
                processedUrls.add(url);
            }
            return;
        }
        
        try {
            // Extract patches from images and run inference
            List<Float> scores = new ArrayList<>();
            
            for (int i = 0; i < validUrls.size(); i++) {
                String url = validUrls.get(i);
                Mat image = images.get(i);
                
                if (image == null || image.empty()) {
                    scores.add((float) ERROR_SCORE);
                    continue;
                }
                
                try {
                    // Extract patches
                    List<float[][][][]> allPatches = ImagePreprocessor.extractPatches(image);
                    List<float[][][][]> selectedPatches = ImagePreprocessor.selectPatches(allPatches);
                    
                    if (selectedPatches.isEmpty()) {
                        logger.warn("No patches extracted from image: {}", url);
                        scores.add((float) ERROR_SCORE);
                        continue;
                    }
                    
                    // Normalize patches
                    // selectedPatches is List<float[][][][]> where each element is [3, 224, 224]
                    // We need to convert to array format for normalizePatches
                    float[][][][] patchesArray = new float[selectedPatches.size()][][][];
                    for (int j = 0; j < selectedPatches.size(); j++) {
                        patchesArray[j] = selectedPatches.get(j)[0]; // Get first (and only) element [3, 224, 224]
                    }
                    float[][][][] normalizedPatches = ImagePreprocessor.normalizePatches(patchesArray);
                    
                    // Convert to ONNX input format
                    // normalizedPatches is [num_patches, 3, 224, 224]
                    // We need to wrap each patch back into [1, 3, 224, 224] for patchesToOnnxInput
                    List<float[][][][]> normalizedList = new ArrayList<>();
                    for (int p = 0; p < normalizedPatches.length; p++) {
                        float[][][][] wrappedPatch = new float[1][][][];
                        wrappedPatch[0] = normalizedPatches[p];
                        normalizedList.add(wrappedPatch);
                    }
                    float[] onnxInput = ImagePreprocessor.patchesToOnnxInput(normalizedList);
                    
                    // Run LIQE model inference
                    int numPatches = selectedPatches.size();
                    float[] qualityScores = modelManager.runLiqeModel(
                        onnxInput, 1, numPatches
                    );
                    
                    if (qualityScores.length > 0) {
                        float score = qualityScores[0];
                        // Validate score range
                        if (score >= 1.0f && score <= 5.0f) {
                            scores.add(score);
                        } else {
                            logger.warn("Score {} out of range for URL {}", score, url);
                            scores.add((float) ERROR_SCORE);
                        }
                    } else {
                        scores.add((float) ERROR_SCORE);
                    }
                    
                    // Clean up
                    image.close();
                    
                } catch (Exception e) {
                    logger.error("Error processing image {}: {}", url, e.getMessage(), e);
                    scores.add((float) ERROR_SCORE);
                    if (image != null && !image.empty()) {
                        image.close();
                    }
                }
            }
            
            // Forward results
            for (int i = 0; i < batchUrls.length; i++) {
                String url = batchUrls[i];
                double score;
                
                // Find corresponding score
                int validIdx = validUrls.indexOf(url);
                if (validIdx >= 0 && validIdx < scores.size()) {
                    score = scores.get(validIdx);
                } else {
                    score = ERROR_SCORE;
                }
                
                try {
                    forward(url, score);
                } catch (com.aliyun.odps.udf.UDFException e) {
                    logger.error("Error forwarding result for {}: {}", url, e.getMessage());
                }
                processedUrls.add(url);
            }
            
        } catch (Exception e) {
            logger.error("Error in batch processing: {}", e.getMessage(), e);
            // Output error scores for all URLs in batch
            for (String url : batchUrls) {
                if (!processedUrls.contains(url)) {
                    try {
                        forward(url, ERROR_SCORE);
                    } catch (com.aliyun.odps.udf.UDFException ex) {
                        logger.error("Error forwarding error result for {}: {}", url, ex.getMessage());
                    }
                    processedUrls.add(url);
                }
            }
        } finally {
            // Clean up images
            for (Mat image : images) {
                if (image != null) {
                    try {
                        if (!image.empty()) {
                            image.close();
                        }
                    } catch (Exception e) {
                        // Ignore cleanup errors (image may already be closed)
                        logger.debug("Error closing image: {}", e.getMessage());
                    }
                }
            }
        }
    }
    
    /**
     * Forward result to UDTF collector
     */
    private void forward(String url, double score) throws com.aliyun.odps.udf.UDFException {
        UDTFCollector collector = getCollector();
        if (collector != null) {
            collector.collect(new Object[]{url, score});
        }
    }
    
    /**
     * Get resource path - tries to get from ODPS resource or file system
     * @param resourceName Name of the resource file
     * @param ctx ExecutionContext (can be null for local testing)
     * @return Path to the resource file, or null if not found
     */
    private String getResourcePath(String resourceName, com.aliyun.odps.udf.ExecutionContext ctx) {
        // Priority 1: Try to read from ODPS ExecutionContext (if available)
        if (ctx != null) {
            try {
                // In ODPS, resources added via ADD FILE are available via ExecutionContext
                // Try to read as stream to check if it exists
                java.io.BufferedInputStream stream = ctx.readResourceFileAsStream(resourceName);
                if (stream != null) {
                    stream.close();
                    // In ODPS, when using ExecutionContext.readResourceFileAsStream(),
                    // the resource file is available, but ONNX Runtime needs a file path.
                    // We need to copy the stream to a temporary file and return that path.
                    // For now, return the resource name - ODPS will make it available
                    // at a location that can be accessed via the resource name
                    // The actual path resolution will be handled when loading the model
                    return resourceName;
                }
            } catch (Exception e) {
                logger.debug("Could not read resource {} from ExecutionContext: {}", resourceName, e.getMessage());
            }
        }
        
        // Priority 2: Try file system paths (for local testing)
        try {
            // Try current directory first
            java.io.File file = new java.io.File(resourceName);
            if (file.exists()) {
                return file.getAbsolutePath();
            }
            
            // Try resources directory
            file = new java.io.File("resources/" + resourceName);
            if (file.exists()) {
                return file.getAbsolutePath();
            }
            
            // Try weights/onnx directory (common location - relative to java_codes)
            file = new java.io.File("weights/onnx/" + resourceName);
            if (file.exists()) {
                return file.getAbsolutePath();
            }
            
            // Try ../weights/onnx (relative to JAR location - for backward compatibility)
            file = new java.io.File("../weights/onnx/" + resourceName);
            if (file.exists()) {
                return file.getAbsolutePath();
            }
            
        } catch (Exception e) {
            logger.debug("Could not find resource {} in file system: {}", resourceName, e.getMessage());
        }
        
        return null;
    }
    
    /**
     * Load model from ODPS resource or file path
     * If resourcePath is a resource name (not a file path), copy from ExecutionContext to temp file
     */
    private String resolveModelPath(String resourcePath, String resourceName, com.aliyun.odps.udf.ExecutionContext ctx) 
            throws java.io.IOException {
        // If it's already a valid file path, return it
        java.io.File file = new java.io.File(resourcePath);
        if (file.exists() && file.isFile()) {
            return file.getAbsolutePath();
        }
        
        // If it's a resource name (from ODPS), try to read from ExecutionContext and save to temp file
        if (ctx != null) {
            try {
                java.io.BufferedInputStream stream = ctx.readResourceFileAsStream(resourcePath);
                if (stream != null) {
                    // Copy to temporary file
                    java.io.File tempFile = java.io.File.createTempFile("liqe_" + resourceName.replace(".", "_"), 
                        "." + resourceName.substring(resourceName.lastIndexOf(".") + 1));
                    tempFile.deleteOnExit();
                    
                    try (java.io.FileOutputStream fos = new java.io.FileOutputStream(tempFile)) {
                        byte[] buffer = new byte[8192];
                        int bytesRead;
                        while ((bytesRead = stream.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                    }
                    
                    stream.close();
                    logger.info("Copied ODPS resource {} to temporary file: {}", resourcePath, tempFile.getAbsolutePath());
                    return tempFile.getAbsolutePath();
                }
            } catch (Exception e) {
                logger.warn("Could not load resource {} from ExecutionContext: {}", resourcePath, e.getMessage());
            }
        }
        
        // If still not found, return the original path (might fail later with better error message)
        return resourcePath;
    }
    
    /**
     * Cleanup resources
     */
    @Override
    public void close() {
        try {
            if (modelManager != null) {
                modelManager.close();
            }
            if (imageDownloader != null) {
                imageDownloader.close();
            }
            logger.info("LIQE UDTF closed");
        } catch (Exception e) {
            logger.error("Error closing UDTF: {}", e.getMessage());
        }
    }
}

