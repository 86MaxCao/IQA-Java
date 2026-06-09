package com.autonavi.iqa.models.maniqa;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseIQAModel;
import com.autonavi.iqa.utils.ImageDownloader;
import com.autonavi.iqa.utils.ImagePreprocessor;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * MANIQA model for blind image quality assessment.
 *
 * <p>Uses 20 uniform 224x224 crops with Inception normalization (mean=0.5, std=0.5).
 * The final score is the average across all crops.
 */
public class MANIQAModel extends BaseIQAModel {

    private static final int CROP_SIZE = 224;
    private static final int NUM_CROPS = 20;

    private MANIQAModelManager modelManager;
    private ImageDownloader imageDownloader;

    @Override
    protected void initializeModel(ModelConfig config) throws ModelException {
        modelManager = new MANIQAModelManager();
        String modelPath = config.getModelPath("maniqa_model");
        if (modelPath == null) {
            modelPath = config.getModelPath("model");
        }
        if (modelPath == null) {
            throw new ModelException("Missing model path: 'maniqa_model' or 'model'");
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
                ImagePreprocessor.normalize(data, ImagePreprocessor.INCEPTION_MEAN, ImagePreprocessor.INCEPTION_STD);
                float[] flat = ImagePreprocessor.cropsToOnnxInput(
                        java.util.Collections.singletonList(crop), CROP_SIZE);
                sum += modelManager.runSingleCrop(flat);
            }
            return sum / crops.size();
        } catch (ModelException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelException("MANIQA assessment failed: " + e.getMessage(), e);
        } finally {
            if (image != null) image.close();
        }
    }

    @Override
    public String getModelName() {
        return "maniqa";
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
