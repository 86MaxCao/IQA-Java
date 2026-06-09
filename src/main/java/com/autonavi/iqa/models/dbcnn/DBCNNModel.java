package com.autonavi.iqa.models.dbcnn;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseIQAModel;
import com.autonavi.iqa.utils.ImageDownloader;
import com.autonavi.iqa.utils.ImagePreprocessor;
import org.bytedeco.opencv.opencv_core.Mat;

/**
 * DBCNN (Deep Bilinear CNN) model for blind image quality assessment.
 *
 * <p>DBCNN accepts full-resolution images (resized to a fixed size for consistency).
 * No multi-crop is needed; the model processes the entire image at once.
 *
 * <p>Preprocessing: resize to 512x384, ImageNet normalization.
 */
public class DBCNNModel extends BaseIQAModel {

    private static final int RESIZE_WIDTH = 512;
    private static final int RESIZE_HEIGHT = 384;

    private DBCNNModelManager modelManager;
    private ImageDownloader imageDownloader;

    @Override
    protected void initializeModel(ModelConfig config) throws ModelException {
        modelManager = new DBCNNModelManager();
        String modelPath = config.getModelPath("dbcnn_model");
        if (modelPath == null) {
            modelPath = config.getModelPath("model");
        }
        if (modelPath == null) {
            throw new ModelException("Missing model path: 'dbcnn_model' or 'model'");
        }
        modelManager.loadModel(modelPath, config.getExecutionContext());
        imageDownloader = new ImageDownloader();
    }

    @Override
    protected double assessImageQuality(String imageUrl) throws ModelException {
        Mat image = null;
        Mat resized = null;
        try {
            image = imageDownloader.downloadImage(imageUrl);
            if (image == null || image.empty()) {
                throw new ModelException("Failed to download image: " + imageUrl);
            }

            resized = ImagePreprocessor.resizeExact(image, RESIZE_WIDTH, RESIZE_HEIGHT);
            float[][][] rgb = matToNormalized(resized);
            float[] flat = flatten(rgb, RESIZE_HEIGHT, RESIZE_WIDTH);

            return modelManager.runInferenceSingle(flat, RESIZE_HEIGHT, RESIZE_WIDTH);
        } catch (ModelException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelException("DBCNN assessment failed: " + e.getMessage(), e);
        } finally {
            if (resized != null) resized.close();
            if (image != null) image.close();
        }
    }

    @Override
    public String getModelName() {
        return "dbcnn";
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

    private static float[][][] matToNormalized(Mat mat) {
        int h = mat.rows(), w = mat.cols(), ch = mat.channels();
        byte[] buf = new byte[ch * h * w];
        mat.data().get(buf);
        float[][][] result = new float[3][h][w];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int pixel = buf[(y * w + x) * ch + c] & 0xFF;
                    int rgb = 2 - c; // BGR → RGB
                    result[rgb][y][x] = (pixel / 255.0f - ImagePreprocessor.IMAGENET_MEAN[rgb])
                            / ImagePreprocessor.IMAGENET_STD[rgb];
                }
            }
        }
        return result;
    }

    private static float[] flatten(float[][][] data, int h, int w) {
        float[] out = new float[3 * h * w];
        int idx = 0;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    out[idx++] = data[c][y][x];
        return out;
    }
}
