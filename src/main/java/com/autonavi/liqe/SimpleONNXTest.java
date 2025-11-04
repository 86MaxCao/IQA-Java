package com.autonavi.liqe;

import ai.onnxruntime.*;

/**
 * Simple test to verify ONNX Runtime can load and run models
 * This is a minimal test before full implementation
 */
public class SimpleONNXTest {
    
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: SimpleONNXTest <clip_model.onnx> <liqe_model.onnx>");
            System.out.println("Example: SimpleONNXTest weights/onnx/clip_model.onnx weights/onnx/liqe_model.onnx");
            System.exit(1);
        }
        
        String clipModelPath = args[0];
        String liqeModelPath = args[1];
        
        System.out.println(repeatString("=", 60));
        System.out.println("Simple ONNX Runtime Test");
        System.out.println(repeatString("=", 60));
        
        OrtEnvironment env = null;
        OrtSession clipSession = null;
        OrtSession liqeSession = null;
        
        try {
            // 1. Initialize ONNX Runtime environment
            System.out.println("\n[1/4] Initializing ONNX Runtime environment...");
            env = OrtEnvironment.getEnvironment();
            System.out.println("✓ ONNX Runtime environment initialized");
            // Note: getVersion() is not available in all ONNX Runtime versions
            // System.out.println("  ONNX Runtime version: " + OrtEnvironment.getVersion());
            
            // 2. Load CLIP model
            System.out.println("\n[2/4] Loading CLIP model from: " + clipModelPath);
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            // Note: Thread configuration methods may vary by ONNX Runtime version
            try {
                opts.setIntraOpNumThreads(1);
            } catch (Exception e) {
                // Ignore if method doesn't exist
            }
            try {
                opts.setInterOpNumThreads(1);
            } catch (Exception e) {
                // Ignore if method doesn't exist
            }
            
            clipSession = env.createSession(clipModelPath, opts);
            System.out.println("✓ CLIP model loaded successfully");
            
            // Print model inputs/outputs
            System.out.println("  Model inputs:");
            for (NodeInfo input : clipSession.getInputInfo().values()) {
                System.out.println("    - " + input.getName() + ": " + input.getInfo());
            }
            System.out.println("  Model outputs:");
            for (NodeInfo output : clipSession.getOutputInfo().values()) {
                System.out.println("    - " + output.getName() + ": " + output.getInfo());
            }
            
            // 3. Test CLIP model with dummy input
            System.out.println("\n[3/4] Testing CLIP model with dummy input...");
            float[] dummyImage = new float[1 * 3 * 224 * 224]; // [1, 3, 224, 224]
            // Fill with random values
            for (int i = 0; i < dummyImage.length; i++) {
                dummyImage[i] = (float) (Math.random() * 2.0 - 1.0); // Range [-1, 1]
            }
            
            long[] shape = {1, 3, 224, 224};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, 
                java.nio.FloatBuffer.wrap(dummyImage), shape);
            
            try (OrtSession.Result result = clipSession.run(
                    java.util.Collections.singletonMap("image", inputTensor)
            )) {
                OnnxValue output = result.get(0);
                Object outputValue = output.getValue();
                
                if (outputValue instanceof float[][]) {
                    float[][] outputArray = (float[][]) outputValue;
                    System.out.println("✓ CLIP model inference successful");
                    System.out.println("  Output shape: [" + outputArray.length + ", " + 
                        (outputArray.length > 0 ? outputArray[0].length : 0) + "]");
                } else {
                    System.out.println("✓ CLIP model inference successful");
                    System.out.println("  Output type: " + outputValue.getClass().getName());
                }
            } finally {
                inputTensor.close();
            }
            
            // 4. Load LIQE model
            System.out.println("\n[4/4] Loading LIQE model from: " + liqeModelPath);
            liqeSession = env.createSession(liqeModelPath, opts);
            System.out.println("✓ LIQE model loaded successfully");
            
            // Print model inputs/outputs
            System.out.println("  Model inputs:");
            for (NodeInfo input : liqeSession.getInputInfo().values()) {
                System.out.println("    - " + input.getName() + ": " + input.getInfo());
            }
            System.out.println("  Model outputs:");
            for (NodeInfo output : liqeSession.getOutputInfo().values()) {
                System.out.println("    - " + output.getName() + ": " + output.getInfo());
            }
            
            // Test LIQE model with dummy input
            System.out.println("\n[5/5] Testing LIQE model with dummy input...");
            int numPatches = 15;
            float[] dummyPatches = new float[1 * numPatches * 3 * 224 * 224]; // [1, 15, 3, 224, 224]
            for (int i = 0; i < dummyPatches.length; i++) {
                dummyPatches[i] = (float) (Math.random() * 2.0 - 1.0);
            }
            
            float[] dummyLogitScale = new float[]{4.6052f}; // Default CLIP logit scale
            
            long[] patchesShape = {1, numPatches, 3, 224, 224};
            long[] logitShape = {1};
            
            OnnxTensor patchesTensor = OnnxTensor.createTensor(env,
                java.nio.FloatBuffer.wrap(dummyPatches), patchesShape);
            OnnxTensor logitTensor = OnnxTensor.createTensor(env,
                java.nio.FloatBuffer.wrap(dummyLogitScale), logitShape);
            
            java.util.Map<String, OnnxTensor> inputs = new java.util.HashMap<>();
            inputs.put("image_patches", patchesTensor);
            inputs.put("logit_scale", logitTensor);
            
            try (OrtSession.Result result = liqeSession.run(inputs)) {
                OnnxValue output = result.get(0);
                Object outputValue = output.getValue();
                
                if (outputValue instanceof float[]) {
                    float[] outputArray = (float[]) outputValue;
                    System.out.println("✓ LIQE model inference successful");
                    System.out.println("  Output shape: [" + outputArray.length + "]");
                    if (outputArray.length > 0) {
                        System.out.println("  Quality score: " + outputArray[0]);
                        // Find min and max manually (no stream support for float[])
                        float min = outputArray[0];
                        float max = outputArray[0];
                        for (float val : outputArray) {
                            if (val < min) min = val;
                            if (val > max) max = val;
                        }
                        System.out.println("  Score range: [" + min + ", " + max + "]");
                    }
                } else {
                    System.out.println("✓ LIQE model inference successful");
                    System.out.println("  Output type: " + outputValue.getClass().getName());
                }
            } finally {
                patchesTensor.close();
                logitTensor.close();
            }
            
            System.out.println("\n" + repeatString("=", 60));
            System.out.println("All tests passed! ONNX Runtime is working correctly.");
            System.out.println(repeatString("=", 60));
            
        } catch (Exception e) {
            System.err.println("\n✗ Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                if (clipSession != null) clipSession.close();
                if (liqeSession != null) liqeSession.close();
            } catch (OrtException e) {
                System.err.println("Error closing sessions: " + e.getMessage());
            }
        }
    }
    
    /**
     * Repeat a string n times (Java 8 compatible replacement for String.repeat())
     */
    private static String repeatString(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
}

