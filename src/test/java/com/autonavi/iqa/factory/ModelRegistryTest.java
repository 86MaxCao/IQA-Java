package com.autonavi.iqa.factory;

import com.autonavi.iqa.common.IImageQualityModel;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ModelRegistryTest {

    @Test
    void allSevenModelsAreRegistered() {
        assertEquals(7, ModelRegistry.getModelCount());
    }

    @Test
    void knownModelsAreRegistered() {
        String[] expected = {"liqe", "dbcnn", "hyperiqa", "maniqa", "musiq", "tres", "clipiqa"};
        for (String name : expected) {
            assertTrue(ModelRegistry.isRegistered(name),
                    "Model should be registered: " + name);
        }
    }

    @Test
    void lookupIsCaseInsensitive() {
        assertNotNull(ModelRegistry.getModelClass("LIQE"));
        assertNotNull(ModelRegistry.getModelClass("Liqe"));
        assertNotNull(ModelRegistry.getModelClass("liqe"));
    }

    @Test
    void unknownModelReturnsNull() {
        assertNull(ModelRegistry.getModelClass("nonexistent"));
    }

    @Test
    void nullModelReturnsNull() {
        assertNull(ModelRegistry.getModelClass(null));
        assertFalse(ModelRegistry.isRegistered(null));
    }

    @Test
    void getAvailableModelsReturnsAllNames() {
        Set<String> models = ModelRegistry.getAvailableModels();
        assertTrue(models.contains("liqe"));
        assertTrue(models.contains("dbcnn"));
        assertTrue(models.contains("clipiqa"));
    }

    @Test
    void getModelClassReturnsIImageQualityModelSubtype() {
        Class<? extends IImageQualityModel> clazz = ModelRegistry.getModelClass("liqe");
        assertNotNull(clazz);
        assertTrue(IImageQualityModel.class.isAssignableFrom(clazz));
    }

    @Test
    void registerModelRejectsNullName() {
        assertThrows(IllegalArgumentException.class,
                () -> ModelRegistry.registerModel(null, com.autonavi.iqa.models.liqe.LIQEModel.class));
    }

    @Test
    void registerModelRejectsEmptyName() {
        assertThrows(IllegalArgumentException.class,
                () -> ModelRegistry.registerModel("  ", com.autonavi.iqa.models.liqe.LIQEModel.class));
    }

    @Test
    void registerModelRejectsNullClass() {
        assertThrows(IllegalArgumentException.class,
                () -> ModelRegistry.registerModel("test", null));
    }
}
