package com.autonavi.iqa.models.hyperiqa;

import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseONNXModelManager;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * ONNX model manager for HyperIQA.
 *
 * <p>HyperIQA uses a hyper-network that dynamically generates quality-prediction
 * weights from content-aware features. The ONNX model takes a single
 * 224x224 crop and outputs a scalar quality score.
 */
public class HyperIQAModelManager extends BaseONNXModelManager {

    private static final int CROP_SIZE = 224;

    public float runSingleCrop(float[] input) throws ModelException {
        if (!isLoaded()) {
            throw new ModelException("HyperIQA model not loaded");
        }

        long[] shape = {1, 3, CROP_SIZE, CROP_SIZE};
        try {
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape);
            try (OrtSession.Result result = session.run(Collections.singletonMap("input", tensor))) {
                float[][] output = (float[][]) result.get(0).getValue();
                return output[0][0];
            } finally {
                tensor.close();
            }
        } catch (OrtException e) {
            throw new ModelException("HyperIQA inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public float[] runInference(float[] input) throws ModelException {
        return new float[]{runSingleCrop(input)};
    }

    @Override
    public float[][] runBatchInference(float[][] inputs) throws ModelException {
        float[][] results = new float[inputs.length][1];
        for (int i = 0; i < inputs.length; i++) {
            results[i][0] = runSingleCrop(inputs[i]);
        }
        return results;
    }
}
