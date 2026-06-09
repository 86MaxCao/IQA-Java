package com.autonavi.iqa.models.musiq;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseIQAModel;
import com.autonavi.iqa.utils.ImageDownloader;
import com.autonavi.iqa.utils.ImagePreprocessor;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * MUSIQ (Multi-Scale Image Quality Transformer) model for blind IQA.
 *
 * <p>Simplified single-scale deployment: 15 uniform 224x224 crops, normalized to
 * [-1, 1] range. The final score is the average across all crops.
 *
 * <p>Note: the original MUSIQ uses multi-scale patches with positional encoding.
 * This simplified version uses uniform crops at a single scale for ONNX compatibility.
 */
public class MUSIQModel extends BaseIQAModel {

    private static final int CROP_SIZE = 224;
    private static final int NUM_CROPS = 15;
    private static final float[] MUSIQ_MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] MUSIQ_STD = {0.5f, 0.5f, 0.5f};

    private MUSIQModelManager modelManager;
    private ImageDownloader imageDownloader;

    @Override
    protected void initializeModel(ModelConfig config) throws ModelException {
        modelManager = new MUSIQModelManager();
        String modelPath = config.getModelPath("musiq_model");
        if (modelPath == null) {
            modelPath = config.getModelPath("model");
        }
        if (modelPath == null) {
            throw new ModelException("Missing model path: 'musiq_model' or 'model'");
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
                ImagePreprocessor.normalize(data, MUSIQ_MEAN, MUSIQ_STD);
                float[] flat = ImagePreprocessor.cropsToOnnxInput(
                        java.util.Collections.singletonList(crop), CROP_SIZE);
                sum += modelManager.runSingleCrop(flat);
            }
            return sum / crops.size();
        } catch (ModelException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelException("MUSIQ assessment failed: " + e.getMessage(), e);
        } finally {
            if (image != null) image.close();
        }
    }

    @Override
    public String getModelName() {
        return "musiq";
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
