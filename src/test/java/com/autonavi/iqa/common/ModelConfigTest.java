package com.autonavi.iqa.common;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelConfigTest {

    @Test
    void defaultConstructorInitializesEmptyMaps() {
        ModelConfig config = new ModelConfig();

        assertNull(config.getModelName());
        assertNull(config.getModelVersion());
        assertNotNull(config.getModelPaths());
        assertTrue(config.getModelPaths().isEmpty());
        assertNotNull(config.getParameters());
        assertTrue(config.getParameters().isEmpty());
    }

    @Test
    void nameConstructorSetsName() {
        ModelConfig config = new ModelConfig("liqe");

        assertEquals("liqe", config.getModelName());
        assertNotNull(config.getModelPaths());
        assertNotNull(config.getParameters());
    }

    @Test
    void setAndGetModelPath() {
        ModelConfig config = new ModelConfig();
        config.setModelPath("onnx", "/models/model.onnx");

        assertEquals("/models/model.onnx", config.getModelPath("onnx"));
        assertNull(config.getModelPath("nonexistent"));
    }

    @Test
    void setModelPathsReplacesMap() {
        ModelConfig config = new ModelConfig();
        config.setModelPath("old", "v1");

        Map<String, String> newPaths = new HashMap<>();
        newPaths.put("new", "v2");
        config.setModelPaths(newPaths);

        assertNull(config.getModelPath("old"));
        assertEquals("v2", config.getModelPath("new"));
    }

    @Test
    void typedParameterRetrieval() {
        ModelConfig config = new ModelConfig();
        config.setParameter("numCrops", 25);
        config.setParameter("threshold", 0.5);
        config.setParameter("name", "test");

        assertEquals(25, config.getParameter("numCrops", Integer.class));
        assertEquals(0.5, config.getParameter("threshold", Double.class));
        assertEquals("test", config.getParameter("name", String.class));
    }

    @Test
    void getParameterReturnsNullForMissing() {
        ModelConfig config = new ModelConfig();
        assertNull(config.getParameter("missing", String.class));
    }

    @Test
    void getParameterReturnsNullForTypeMismatch() {
        ModelConfig config = new ModelConfig();
        config.setParameter("count", "not-an-int");

        assertNull(config.getParameter("count", Integer.class));
    }

    @Test
    void modelVersionRoundTrip() {
        ModelConfig config = new ModelConfig();
        config.setModelVersion("1.2.3");
        assertEquals("1.2.3", config.getModelVersion());
    }

    @Test
    void setParametersReplacesMap() {
        ModelConfig config = new ModelConfig();
        config.setParameter("a", 1);

        Map<String, Object> newParams = new HashMap<>();
        newParams.put("b", 2);
        config.setParameters(newParams);

        assertNull(config.getParameter("a", Integer.class));
        assertEquals(2, config.getParameter("b", Integer.class));
    }
}
