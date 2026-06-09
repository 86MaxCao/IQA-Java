package com.autonavi.iqa.models.clipiqa;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseIQAModel;
import com.autonavi.iqa.utils.ImageDownloader;
import com.autonavi.iqa.utils.ImagePreprocessor;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * CLIPIQA model for blind image quality assessment.
 *
 * <p>Uses CLIP image encoder with quality-aware text prompts. Preprocessing
 * uses CLIP normalization (same as LIQE). A single center crop of 224x224 is
 * used for inference, outputting a quality score in [0, 1].
 */
public class CLIPIQAModel extends BaseIQAModel {

    private static final int CROP_SIZE = 224;

    private CLIPIQAModelManager modelManager;
    private ImageDownloader imageDownloader;

    @Override
    protected void initializeModel(ModelConfig config) throws ModelException {
        modelManager = new CLIPIQAModelManager();
        String modelPath = config.getModelPath("clipiqa_model");
        if (modelPath == null) {
            modelPath = config.getModelPath("model");
        }
        if (modelPath == null) {
            throw new ModelException("Missing model path: 'clipiqa_model' or 'model'");
        }
        modelManager.loadModel(modelPath, config.getExecutionContext());
        imageDownloader = new ImageDownloader();
    }

    @Override
    protected double assessImageQuality(String imageUrl) throws ModelException {
        Mat image = null;
        Mat resized = null;
        Mat cropped = null;
        try {
            image = imageDownloader.downloadImage(imageUrl);
            if (image == null || image.empty()) {
                throw new ModelException("Failed to download image: " + imageUrl);
            }

            int shortSide = Math.min(image.rows(), image.cols());
            float scale = (float) CROP_SIZE / shortSide;
            int newH = Math.round(image.rows() * scale);
            int newW = Math.round(image.cols() * scale);
            resized = ImagePreprocessor.resizeExact(image, newW, newH);
            cropped = ImagePreprocessor.centerCrop(resized, CROP_SIZE);

            List<float[][][][]> cropList = ImagePreprocessor.uniformCrop(cropped, CROP_SIZE, 1);
            if (cropList.isEmpty()) {
                throw new ModelException("No crop extracted");
            }

            float[][][] data = cropList.get(0)[0];
            ImagePreprocessor.normalize(data, ImagePreprocessor.OPENAI_CLIP_MEAN, ImagePreprocessor.OPENAI_CLIP_STD);
            float[] flat = ImagePreprocessor.cropsToOnnxInput(
                    java.util.Collections.singletonList(cropList.get(0)), CROP_SIZE);

            return modelManager.runSingleCrop(flat);
        } catch (ModelException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelException("CLIPIQA assessment failed: " + e.getMessage(), e);
        } finally {
            if (cropped != null) cropped.close();
            if (resized != null) resized.close();
            if (image != null) image.close();
        }
    }

    @Override
    public String getModelName() {
        return "clipiqa";
    }

    @Override
    public String getModelVersion() {
        return "1.0";
    }

    @Override
    protected void cleanup() {
        if (modelManager != null) { modelManager.close(); modelManager = null; }
        if (imageDownloader != null) { imageDownloader.close(); imageDownloader = null; }
    }
}
