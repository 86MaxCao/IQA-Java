package com.autonavi.iqa.utils;

import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Rect;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImagePreprocessor {

    private static final Logger logger = LoggerFactory.getLogger(ImagePreprocessor.class);

    public static final float[] OPENAI_CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    public static final float[] OPENAI_CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};

    public static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    public static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    public static final float[] INCEPTION_MEAN = {0.5f, 0.5f, 0.5f};
    public static final float[] INCEPTION_STD = {0.5f, 0.5f, 0.5f};
    
    public static final int TARGET_WIDTH = 640;
    public static final int TARGET_HEIGHT = 360;
    public static final int PATCH_SIZE = 224;
    public static final int STEP = 32;
    public static final int NUM_PATCHES = 15;
    
    // Static initializer to ensure OpenCV native libraries are loaded
    static {
        try {
            Class.forName("org.bytedeco.opencv.global.opencv_core");
            Class.forName("org.bytedeco.opencv.global.opencv_imgproc");
            Class.forName("org.bytedeco.opencv.global.opencv_imgcodecs");
        } catch (ClassNotFoundException e) {
            System.err.println("Warning: OpenCV classes not found: " + e.getMessage());
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Warning: OpenCV native libraries could not be loaded: " + e.getMessage());
        }
    }
    
    /**
     * Resize image maintaining aspect ratio and padding to target size
     */
    public static Mat resizeImage(Mat image) {
        int originalWidth = image.cols();
        int originalHeight = image.rows();
        
        double originalAspect = (double) originalWidth / originalHeight;
        double targetAspect = (double) TARGET_WIDTH / TARGET_HEIGHT;
        
        int newWidth, newHeight;
        if (originalAspect > targetAspect) {
            newWidth = TARGET_WIDTH;
            newHeight = (int) Math.round(newWidth / originalAspect);
        } else {
            newHeight = TARGET_HEIGHT;
            newWidth = (int) Math.round(newHeight * originalAspect);
        }
        
        Mat resized = new Mat();
        resize(image, resized, new Size(newWidth, newHeight));
        
        Mat result = new Mat(TARGET_HEIGHT, TARGET_WIDTH, CV_8UC3, new Scalar(0, 0, 0, 0));
        
        int padLeft = (TARGET_WIDTH - newWidth) / 2;
        int padTop = (TARGET_HEIGHT - newHeight) / 2;
        
        Rect roiRect = new Rect(padLeft, padTop, newWidth, newHeight);
        Mat roi = new Mat(result, roiRect);
        resized.copyTo(roi);
        
        resized.close();
        return result;
    }
    
    /**
     * Extract patches from image (LIQE-specific)
     */
    public static List<float[][][][]> extractPatches(Mat image) {
        List<float[][][][]> patches = new ArrayList<>();
        
        int h = image.rows();
        int w = image.cols();
        
        for (int y = 0; y <= h - PATCH_SIZE; y += STEP) {
            for (int x = 0; x <= w - PATCH_SIZE; x += STEP) {
                Rect rect = new Rect(x, y, PATCH_SIZE, PATCH_SIZE);
                Mat patch = new Mat(image, rect);
                
                float[][][] patchArray = matToFloatArray(patch);
                patches.add(new float[][][][]{patchArray});
                
                patch.close();
            }
        }
        
        return patches;
    }
    
    /**
     * Select patches for inference
     */
    public static List<float[][][][]> selectPatches(List<float[][][][]> allPatches) {
        int numPatches = Math.min(allPatches.size(), NUM_PATCHES);
        
        if (numPatches == 0) {
            return new ArrayList<>();
        }
        
        List<float[][][][]> selected = new ArrayList<>();
        
        if (numPatches >= allPatches.size()) {
            selected.addAll(allPatches);
        } else {
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
        
        byte[] buffer = new byte[channels * height * width];
        mat.data().get(buffer);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = (h * width + w) * channels + c;
                    int pixel = buffer[idx] & 0xFF;
                    int rgbChannel = (channels == 3) ? (2 - c) : c;
                    result[rgbChannel][h][w] = pixel / 255.0f;
                }
            }
        }
        
        return result;
    }
    
    /**
     * Normalize image patches using CLIP mean and std
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
     * Convert patches to ONNX input format
     */
    public static float[] patchesToOnnxInput(List<float[][][][]> patches) {
        int numPatches = patches.size();
        int totalSize = numPatches * 3 * 224 * 224;
        float[] result = new float[totalSize];
        
        int idx = 0;
        for (float[][][][] patch : patches) {
            float[][][] patchData = patch[0];
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
    
    public static BufferedImage matToBufferedImage(Mat mat) {
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter converterToJava2D = new Java2DFrameConverter();
        return converterToJava2D.convert(converterToMat.convert(mat));
    }

    /**
     * Extract uniform crops from an image for multi-crop inference.
     * Produces a grid of overlapping crops that cover the image as evenly as possible.
     *
     * @param image    source image (BGR, CV_8UC3)
     * @param cropSize crop width and height in pixels
     * @param numCrops desired number of crops
     * @return list of crop regions as float[1][3][cropSize][cropSize] in RGB [0,1]
     */
    public static List<float[][][][]> uniformCrop(Mat image, int cropSize, int numCrops) {
        int h = image.rows();
        int w = image.cols();

        Mat src = image;
        if (h < cropSize || w < cropSize) {
            int newH = Math.max(h, cropSize);
            int newW = Math.max(w, cropSize);
            src = new Mat();
            resize(image, src, new Size(newW, newH));
            h = newH;
            w = newW;
        }

        int gridSide = (int) Math.ceil(Math.sqrt(numCrops));
        int actualCrops = gridSide * gridSide;

        float stepH = (h - cropSize) / (float) Math.max(gridSide - 1, 1);
        float stepW = (w - cropSize) / (float) Math.max(gridSide - 1, 1);

        List<float[][][][]> crops = new ArrayList<>(actualCrops);
        for (int i = 0; i < gridSide; i++) {
            for (int j = 0; j < gridSide; j++) {
                int y = Math.min(Math.round(i * stepH), h - cropSize);
                int x = Math.min(Math.round(j * stepW), w - cropSize);
                Rect rect = new Rect(x, y, cropSize, cropSize);
                Mat patch = new Mat(src, rect);
                float[][][] arr = matToFloatArray(patch);
                crops.add(new float[][][][]{arr});
                patch.close();
            }
        }

        if (src != image) {
            src.close();
        }
        return crops;
    }

    /**
     * Resize an image to exactly {@code width x height} (no aspect-ratio preservation).
     *
     * @param image  source image
     * @param width  target width
     * @param height target height
     * @return resized image; caller must close
     */
    public static Mat resizeExact(Mat image, int width, int height) {
        Mat result = new Mat();
        resize(image, result, new Size(width, height));
        return result;
    }

    /**
     * Center-crop an image to {@code cropSize x cropSize}.
     *
     * @param image    source image
     * @param cropSize target size
     * @return cropped image; caller must close
     */
    public static Mat centerCrop(Mat image, int cropSize) {
        int h = image.rows();
        int w = image.cols();
        int y = (h - cropSize) / 2;
        int x = (w - cropSize) / 2;
        Rect rect = new Rect(Math.max(x, 0), Math.max(y, 0),
                Math.min(cropSize, w), Math.min(cropSize, h));
        return new Mat(image, rect);
    }

    /**
     * Normalize a single image tensor in-place.
     *
     * @param data float[3][H][W] in [0,1] range
     * @param mean per-channel mean (RGB order)
     * @param std  per-channel standard deviation (RGB order)
     */
    public static void normalize(float[][][] data, float[] mean, float[] std) {
        int channels = data.length;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < data[c].length; h++) {
                for (int w = 0; w < data[c][h].length; w++) {
                    data[c][h][w] = (data[c][h][w] - mean[c]) / std[c];
                }
            }
        }
    }

    /**
     * Flatten a list of crops into a single ONNX-ready float array.
     * Each crop is float[1][3][cropSize][cropSize]; output is float[N*3*H*W].
     */
    public static float[] cropsToOnnxInput(List<float[][][][]> crops, int cropSize) {
        int n = crops.size();
        int total = n * 3 * cropSize * cropSize;
        float[] result = new float[total];
        int idx = 0;
        for (float[][][][] crop : crops) {
            float[][][] data = crop[0];
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < cropSize; h++) {
                    for (int w = 0; w < cropSize; w++) {
                        result[idx++] = data[c][h][w];
                    }
                }
            }
        }
        return result;
    }
}

