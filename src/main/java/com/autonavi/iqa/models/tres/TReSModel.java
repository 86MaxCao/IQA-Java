package com.autonavi.iqa.models.tres;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseIQAModel;
import com.autonavi.iqa.utils.ImageDownloader;
import com.autonavi.iqa.utils.ImagePreprocessor;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * TReS (Transformer for Relative Score) model for blind IQA.
 *
 * <p>Uses 50 uniform 224x224 crops with ImageNet normalization.
 * The final score is the average across all crops.
 *
 * <p>The ONNX model is exported in eval mode only (no flipped-image branch
 * or consistency loss used during training).
 */
public class TReSModel extends BaseIQAModel {

    private static final int CROP_SIZE = 224;
    private static final int NUM_CROPS = 50;

    private TReSModelManager modelManager;
    private ImageDownloader imageDownloader;

    @Override
    protected void initializeModel(ModelConfig config) throws ModelException {
        modelManager = new TReSModelManager();
        String modelPath = config.getModelPath("tres_model");
        if (modelPath == null) {
            modelPath = config.getModelPath("model");
        }
        if (modelPath == null) {
            throw new ModelException("Missing model path: 'tres_model' or 'model'");
        }
        modelManager.loadModel(modelPath, config.getExecutionContext());
        imageDownloader = new ImageDownloader();
    }

    @Override
    protected double assessImageQuality(String imageUrl) throws ModelException {
        Mat image = null;
        try {
            image = imageDownloader.downloadImage(imageUrl);
            if (image == null || image.empty()) {
                throw new ModelException("Failed to download image: " + imageUrl);
            }

            List<float[][][][]> crops = ImagePreprocessor.uniformCrop(image, CROP_SIZE, NUM_CROPS);
            if (crops.isEmpty()) {
                throw new ModelException("No crops extracted");
            }

            double sum = 0;
            for (float[][][][] crop : crops) {
                float[][][] data = crop[0];
                ImagePreprocessor.normalize(data, ImagePreprocessor.IMAGENET_MEAN, ImagePreprocessor.IMAGENET_STD);
                float[] flat = ImagePreprocessor.cropsToOnnxInput(
                        java.util.Collections.singletonList(crop), CROP_SIZE);
                sum += modelManager.runSingleCrop(flat);
            }
            return sum / crops.size();
        } catch (ModelException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelException("TReS assessment failed: " + e.getMessage(), e);
        } finally {
            if (image != null) image.close();
        }
    }

    @Override
    public String getModelName() {
        return "tres";
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
