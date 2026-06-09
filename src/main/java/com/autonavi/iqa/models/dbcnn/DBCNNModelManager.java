package com.autonavi.iqa.models.dbcnn;

import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseONNXModelManager;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * ONNX model manager for DBCNN (Deep Bilinear CNN for Blind Image Quality Assessment).
 *
 * <p>DBCNN accepts variable-resolution input, so no fixed crop is required.
 * The caller resizes the image to a reasonable size (e.g. 512x384) and passes
 * the full image through the model.
 */
public class DBCNNModelManager extends BaseONNXModelManager {

    public float runInferenceSingle(float[] input, int height, int width) throws ModelException {
        if (!isLoaded()) {
            throw new ModelException("DBCNN model not loaded");
        }

        long[] shape = {1, 3, height, width};
        try {
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape);
            try (OrtSession.Result result = session.run(Collections.singletonMap("input", tensor))) {
                float[][] output = (float[][]) result.get(0).getValue();
                return output[0][0];
            } finally {
                tensor.close();
            }
        } catch (OrtException e) {
            throw new ModelException("DBCNN inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public float[] runInference(float[] input) throws ModelException {
        throw new UnsupportedOperationException("Use runInferenceSingle(input, height, width)");
    }

    @Override
    public float[][] runBatchInference(float[][] inputs) throws ModelException {
        throw new UnsupportedOperationException("Use runInferenceSingle for DBCNN");
    }
}
