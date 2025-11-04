package com.autonavi.liqe;

import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Rect;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Image preprocessing utilities for LIQE model
 * Handles image resizing, patch extraction, and normalization
 */
public class ImagePreprocessor {
    
    // Constants from Python code
    private static final float[] OPENAI_CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    private static final float[] OPENAI_CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};
    
    private static final int TARGET_WIDTH = 640;
    private static final int TARGET_HEIGHT = 360;
    private static final int PATCH_SIZE = 224;
    private static final int STEP = 32;
    private static final int NUM_PATCHES = 15;
    
    // Static initializer to ensure OpenCV native libraries are loaded
    static {
        try {
            // Force loading of OpenCV native libraries
            // This ensures native libraries are available before any OpenCV operations
            Class.forName("org.bytedeco.opencv.global.opencv_core");
            Class.forName("org.bytedeco.opencv.global.opencv_imgproc");
            Class.forName("org.bytedeco.opencv.global.opencv_imgcodecs");
        } catch (ClassNotFoundException e) {
            // Log warning but don't fail - let actual usage fail if libraries are missing
            System.err.println("Warning: OpenCV classes not found: " + e.getMessage());
        } catch (UnsatisfiedLinkError e) {
            // Log error but don't fail here - let actual usage fail
            System.err.println("Warning: OpenCV native libraries could not be loaded: " + e.getMessage());
            System.err.println("This may be due to missing native libraries in ODPS environment.");
        }
    }
    
    /**
     * Resize image maintaining aspect ratio and padding to target size
     * Equivalent to Python resize_img function
     */
    public static Mat resizeImage(Mat image) {
        int originalWidth = image.cols();
        int originalHeight = image.rows();
        
        // Calculate aspect ratios
        double originalAspect = (double) originalWidth / originalHeight;
        double targetAspect = (double) TARGET_WIDTH / TARGET_HEIGHT;
        
        int newWidth, newHeight;
        if (originalAspect > targetAspect) {
            // Wider than target, resize based on width
            newWidth = TARGET_WIDTH;
            newHeight = (int) Math.round(newWidth / originalAspect);
        } else {
            // Higher than target, resize based on height
            newHeight = TARGET_HEIGHT;
            newWidth = (int) Math.round(newHeight * originalAspect);
        }
        
        // Resize image
        Mat resized = new Mat();
        resize(image, resized, new Size(newWidth, newHeight));
        
        // Create black background
        Mat result = new Mat(TARGET_HEIGHT, TARGET_WIDTH, CV_8UC3, new Scalar(0, 0, 0, 0));
        
        // Calculate padding
        int padLeft = (TARGET_WIDTH - newWidth) / 2;
        int padTop = (TARGET_HEIGHT - newHeight) / 2;
        
        // Copy resized image to center of black background
        Rect roiRect = new Rect(padLeft, padTop, newWidth, newHeight);
        Mat roi = new Mat(result, roiRect);
        resized.copyTo(roi);
        
        resized.close();
        return result;
    }
    
    /**
     * Extract patches from image
     * Equivalent to Python: x.unfold(2, 224, 32).unfold(3, 224, 32)
     */
    public static List<float[][][][]> extractPatches(Mat image) {
        List<float[][][][]> patches = new ArrayList<>();
        
        int h = image.rows();
        int w = image.cols();
        
        // Extract patches with stride STEP
        for (int y = 0; y <= h - PATCH_SIZE; y += STEP) {
            for (int x = 0; x <= w - PATCH_SIZE; x += STEP) {
                // Extract patch
                Rect rect = new Rect(x, y, PATCH_SIZE, PATCH_SIZE);
                Mat patch = new Mat(image, rect);
                
                // Convert to float array [3, 224, 224] and normalize to [0, 1]
                float[][][] patchArray = matToFloatArray(patch);
                patches.add(new float[][][][]{patchArray});
                
                patch.close();
            }
        }
        
        return patches;
    }
    
    /**
     * Select patches for inference (equivalent to Python patch selection)
     */
    public static List<float[][][][]> selectPatches(List<float[][][][]> allPatches) {
        int numPatches = Math.min(allPatches.size(), NUM_PATCHES);
        
        if (numPatches == 0) {
            return new ArrayList<>();
        }
        
        List<float[][][][]> selected = new ArrayList<>();
        
        if (numPatches >= allPatches.size()) {
            // Use all patches
            selected.addAll(allPatches);
        } else {
            // Select evenly spaced patches
            int step = Math.max(1, allPatches.size() / numPatches);
            for (int i = 0; i < numPatches; i++) {
                int idx = i * step;
                if (idx < allPatches.size()) {
                    selected.add(allPatches.get(idx));
                }
            }
        }
        
        return selected;
    }
    
    /**
     * Convert Mat to float array [3, height, width] with values in [0, 1]
     */
    private static float[][][] matToFloatArray(Mat mat) {
        int channels = mat.channels();
        int height = mat.rows();
        int width = mat.cols();
        
        float[][][] result = new float[channels][height][width];
        
        // Convert BGR to RGB and normalize
        byte[] buffer = new byte[channels * height * width];
        mat.data().get(buffer);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = (h * width + w) * channels + c;
                    int pixel = buffer[idx] & 0xFF;
                    // Convert BGR to RGB: channel 0->2, 1->1, 2->0
                    int rgbChannel = (channels == 3) ? (2 - c) : c;
                    result[rgbChannel][h][w] = pixel / 255.0f;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Normalize image patches using CLIP mean and std
     * Input: [num_patches, 3, 224, 224] in range [0, 1]
     * Output: [num_patches, 3, 224, 224] normalized
     */
    public static float[][][][] normalizePatches(float[][][][] patches) {
        int numPatches = patches.length;
        float[][][][] normalized = new float[numPatches][3][224][224];
        
        for (int p = 0; p < numPatches; p++) {
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < 224; h++) {
                    for (int w = 0; w < 224; w++) {
                        normalized[p][c][h][w] = 
                            (patches[p][c][h][w] - OPENAI_CLIP_MEAN[c]) / OPENAI_CLIP_STD[c];
                    }
                }
            }
        }
        
        return normalized;
    }
    
    /**
     * Convert patches to ONNX input format: [batch_size, num_patches, 3, 224, 224]
     */
    public static float[] patchesToOnnxInput(List<float[][][][]> patches) {
        int numPatches = patches.size();
        int totalSize = numPatches * 3 * 224 * 224;
        float[] result = new float[totalSize];
        
        int idx = 0;
        for (float[][][][] patch : patches) {
            float[][][] patchData = patch[0]; // [3, 224, 224]
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < 224; h++) {
                    for (int w = 0; w < 224; w++) {
                        result[idx++] = patchData[c][h][w];
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * Convert Mat to BufferedImage for display/debugging
     */
    public static BufferedImage matToBufferedImage(Mat mat) {
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter converterToJava2D = new Java2DFrameConverter();
        return converterToJava2D.convert(converterToMat.convert(mat));
    }
}

