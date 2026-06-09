package com.autonavi.iqa.utils;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ImagePreprocessorTest {

    // --- Normalization constants ---

    @Test
    void imagenetMeanHasThreeChannels() {
        assertEquals(3, ImagePreprocessor.IMAGENET_MEAN.length);
        assertEquals(3, ImagePreprocessor.IMAGENET_STD.length);
    }

    @Test
    void clipMeanHasThreeChannels() {
        assertEquals(3, ImagePreprocessor.OPENAI_CLIP_MEAN.length);
        assertEquals(3, ImagePreprocessor.OPENAI_CLIP_STD.length);
    }

    @Test
    void inceptionMeanIsPointFive() {
        for (float v : ImagePreprocessor.INCEPTION_MEAN) {
            assertEquals(0.5f, v, 1e-6f);
        }
        for (float v : ImagePreprocessor.INCEPTION_STD) {
            assertEquals(0.5f, v, 1e-6f);
        }
    }

    @Test
    void stdValuesArePositive() {
        for (float v : ImagePreprocessor.IMAGENET_STD) assertTrue(v > 0);
        for (float v : ImagePreprocessor.OPENAI_CLIP_STD) assertTrue(v > 0);
        for (float v : ImagePreprocessor.INCEPTION_STD) assertTrue(v > 0);
    }

    // --- normalize() ---

    @Test
    void normalizeAppliesMeanAndStd() {
        float[][][] data = new float[3][2][2];
        // Fill channel 0 with 0.5
        for (int h = 0; h < 2; h++)
            for (int w = 0; w < 2; w++)
                data[0][h][w] = 0.5f;

        float[] mean = {0.5f, 0.0f, 0.0f};
        float[] std = {0.25f, 1.0f, 1.0f};

        ImagePreprocessor.normalize(data, mean, std);

        // (0.5 - 0.5) / 0.25 = 0.0
        assertEquals(0.0f, data[0][0][0], 1e-6f);
        // channel 1 was 0.0: (0.0 - 0.0) / 1.0 = 0.0
        assertEquals(0.0f, data[1][0][0], 1e-6f);
    }

    @Test
    void normalizeModifiesInPlace() {
        float[][][] data = {{{1.0f}}};
        float[] mean = {0.5f};
        float[] std = {0.5f};

        ImagePreprocessor.normalize(data, mean, std);
        // (1.0 - 0.5) / 0.5 = 1.0
        assertEquals(1.0f, data[0][0][0], 1e-6f);
    }

    // --- selectPatches() ---

    @Test
    void selectPatchesReturnsRequestedCount() {
        List<float[][][][]> patches = makeDummyPatches(100);
        List<float[][][][]> selected = ImagePreprocessor.selectPatches(patches);
        assertEquals(ImagePreprocessor.NUM_PATCHES, selected.size());
    }

    @Test
    void selectPatchesReturnsAllWhenFewerThanLimit() {
        List<float[][][][]> patches = makeDummyPatches(5);
        List<float[][][][]> selected = ImagePreprocessor.selectPatches(patches);
        assertEquals(5, selected.size());
    }

    @Test
    void selectPatchesReturnsEmptyForEmptyInput() {
        List<float[][][][]> patches = new ArrayList<>();
        List<float[][][][]> selected = ImagePreprocessor.selectPatches(patches);
        assertTrue(selected.isEmpty());
    }

    // --- patchesToOnnxInput() ---

    @Test
    void patchesToOnnxInputProducesCorrectSize() {
        int patchSize = 224;
        int numPatches = 3;
        List<float[][][][]> patches = new ArrayList<>();
        for (int i = 0; i < numPatches; i++) {
            patches.add(new float[][][][]{new float[3][patchSize][patchSize]});
        }

        float[] result = ImagePreprocessor.patchesToOnnxInput(patches);
        assertEquals(numPatches * 3 * patchSize * patchSize, result.length);
    }

    @Test
    void patchesToOnnxInputPreservesValues() {
        float[][][] data = new float[3][224][224];
        data[0][0][0] = 0.42f;
        data[1][100][100] = 0.99f;
        List<float[][][][]> patches = new ArrayList<>();
        patches.add(new float[][][][]{data});

        float[] result = ImagePreprocessor.patchesToOnnxInput(patches);
        assertEquals(0.42f, result[0], 1e-6f);
        // index for channel 1, row 100, col 100: 1*224*224 + 100*224 + 100
        int idx = 224 * 224 + 100 * 224 + 100;
        assertEquals(0.99f, result[idx], 1e-6f);
    }

    // --- cropsToOnnxInput() ---

    @Test
    void cropsToOnnxInputMatchesPatchSize() {
        int cropSize = 32;
        List<float[][][][]> crops = new ArrayList<>();
        crops.add(new float[][][][]{new float[3][cropSize][cropSize]});
        crops.add(new float[][][][]{new float[3][cropSize][cropSize]});

        float[] result = ImagePreprocessor.cropsToOnnxInput(crops, cropSize);
        assertEquals(2 * 3 * cropSize * cropSize, result.length);
    }

    // --- normalizePatches() ---

    @Test
    void normalizePatchesUsesCLIPConstants() {
        float[][][][] patches = new float[1][3][224][224];
        // Fill with CLIP mean so result should be ~0
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 224; h++)
                for (int w = 0; w < 224; w++)
                    patches[0][c][h][w] = ImagePreprocessor.OPENAI_CLIP_MEAN[c];

        float[][][][] normalized = ImagePreprocessor.normalizePatches(patches);
        assertEquals(0.0f, normalized[0][0][0][0], 1e-4f);
        assertEquals(0.0f, normalized[0][1][0][0], 1e-4f);
        assertEquals(0.0f, normalized[0][2][0][0], 1e-4f);
    }

    @Test
    void normalizePatchesPreservesDimensions() {
        float[][][][] patches = new float[5][3][224][224];
        float[][][][] normalized = ImagePreprocessor.normalizePatches(patches);
        assertEquals(5, normalized.length);
        assertEquals(3, normalized[0].length);
        assertEquals(224, normalized[0][0].length);
        assertEquals(224, normalized[0][0][0].length);
    }

    // --- Constants ---

    @Test
    void defaultPatchSizeIs224() {
        assertEquals(224, ImagePreprocessor.PATCH_SIZE);
    }

    // --- Helper ---

    private static List<float[][][][]> makeDummyPatches(int count) {
        List<float[][][][]> patches = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            patches.add(new float[][][][]{new float[3][224][224]});
        }
        return patches;
    }
}
