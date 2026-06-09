package com.autonavi.iqa.factory;

import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.ModelException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ModelFactoryTest {

    @Test
    void nullConfigThrowsModelException() {
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel((ModelConfig) null));
        assertTrue(ex.getMessage().contains("null"));
    }

    @Test
    void emptyModelNameThrowsModelException() {
        ModelConfig config = new ModelConfig();
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel(config));
        assertTrue(ex.getMessage().contains("not specified"));
    }

    @Test
    void blankModelNameThrowsModelException() {
        ModelConfig config = new ModelConfig("  ");
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel(config));
        assertTrue(ex.getMessage().contains("not specified"));
    }

    @Test
    void unknownModelNameThrowsModelException() {
        ModelConfig config = new ModelConfig("nonexistent_model");
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel(config));
        assertTrue(ex.getMessage().contains("Unknown model"));
        assertTrue(ex.getMessage().contains("nonexistent_model"));
    }

    @Test
    void twoArgOverloadSetsModelName() {
        ModelConfig config = new ModelConfig();
        // Will fail at initialize() because no ONNX model files are available,
        // but it should at least find the class and attempt instantiation.
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel("liqe", config));
        // The error should be about initialization, not about unknown model
        assertFalse(ex.getMessage().contains("Unknown model"));
    }

    @Test
    void twoArgOverloadWithNullConfigCreatesDefault() {
        // Should not throw NPE; should create a default config and set the model name
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel("liqe", null));
        assertFalse(ex.getMessage().contains("Unknown model"));
    }

    @Test
    void unknownModelMessageListsAvailable() {
        ModelConfig config = new ModelConfig("fake");
        ModelException ex = assertThrows(ModelException.class,
                () -> ModelFactory.createModel(config));
        // Error message should include available models
        assertTrue(ex.getMessage().contains("liqe") || ex.getMessage().contains("Available"));
    }
}
