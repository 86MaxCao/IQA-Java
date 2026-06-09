package com.autonavi.iqa.models.clipiqa;

import com.autonavi.iqa.common.ModelException;
import com.autonavi.iqa.models.base.BaseONNXModelManager;
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * ONNX model manager for CLIPIQA (CLIP-based Image Quality Assessment).
 *
 * <p>CLIPIQA uses a CLIP image encoder combined with learned quality-aware
 * text prompts. The ONNX model has text features baked in and takes a single
 * 224x224 crop as input, outputting a score in [0, 1].
 */
public class CLIPIQAModelManager extends BaseONNXModelManager {

    private static final int CROP_SIZE = 224;

    public float runSingleCrop(float[] input) throws ModelException {
        if (!isLoaded()) {
            throw new ModelException("CLIPIQA model not loaded");
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
            throw new ModelException("CLIPIQA inference failed: " + e.getMessage(), e);
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
