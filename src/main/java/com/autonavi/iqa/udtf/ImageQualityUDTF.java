package com.autonavi.iqa.udtf;

import com.aliyun.odps.udf.ExecutionContext;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;
import com.autonavi.iqa.common.IImageQualityModel;
import com.autonavi.iqa.common.ModelConfig;
import com.autonavi.iqa.common.QualityScore;
import com.autonavi.iqa.factory.ModelFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Unified Image Quality Assessment UDTF for ODPS/MaxCompute.
 *
 * <p>Supports multiple IQA models (LIQE, DBCNN, HyperIQA, MANIQA, MUSIQ, TReS, CLIPIQA)
 * selected at runtime via the first element of the input array.
 *
 * <p>SQL usage:
 * <pre>
 * SELECT T.url, T.score
 * FROM (
 *   SELECT COLLECT_LIST(image_url) AS urls FROM image_table
 * ) t1
 * LATERAL VIEW image_quality_udtf(ARRAY['liqe'], urls) T AS url, score;
 * </pre>
 *
 * <p>Input: {@code string, array<string>} — model name followed by image URL array.
 * <br>Output: {@code string, double} — image URL and quality score.
 * A score of {@code -1.0} indicates an error.
 */
@Resolve("string, array<string> -> string, double")
public class ImageQualityUDTF extends UDTF {

    private static final Logger logger = LoggerFactory.getLogger(ImageQualityUDTF.class);

    private IImageQualityModel model;
    private boolean initialized = false;
    private String currentModelName;

    @Override
    public void setup(ExecutionContext ctx) throws UDFException {
        try {
            initializeNativeLibraries();
        } catch (Exception e) {
            logger.warn("Native library pre-loading had issues, continuing: {}", e.getMessage());
        }
    }

    @Override
    public void process(Object[] args) throws UDFException {
        String modelName = (String) args[0];
        String[] imageUrls = toStringArray(args[1]);

        if (imageUrls.length == 0) {
            return;
        }

        try {
            ensureModelLoaded(modelName);
        } catch (Exception e) {
            logger.error("Failed to load model '{}': {}", modelName, e.getMessage(), e);
            for (String url : imageUrls) {
                forward(url, QualityScore.ERROR_SCORE);
            }
            return;
        }

        for (String url : imageUrls) {
            double score = QualityScore.ERROR_SCORE;
            try {
                QualityScore result = model.assessQuality(url);
                score = result.getScore();
            } catch (Exception e) {
                logger.error("Error assessing '{}': {}", url, e.getMessage());
            }
            forward(url, score);
        }
    }

    @Override
    public void close() throws UDFException {
        if (model != null) {
            try {
                model.close();
                logger.info("Model '{}' closed", currentModelName);
            } catch (Exception e) {
                logger.warn("Error closing model: {}", e.getMessage());
            }
            model = null;
            initialized = false;
        }
    }

    private void ensureModelLoaded(String modelName) throws Exception {
        if (initialized && modelName.equalsIgnoreCase(currentModelName)) {
            return;
        }

        if (model != null) {
            model.close();
            model = null;
            initialized = false;
        }

        ModelConfig config = new ModelConfig();
        config.setModelName(modelName);
        model = ModelFactory.createModel(config);
        currentModelName = modelName;
        initialized = true;
        logger.info("Model '{}' loaded successfully", modelName);
    }

    @SuppressWarnings("unchecked")
    private static String[] toStringArray(Object input) {
        if (input instanceof List) {
            List<String> list = (List<String>) input;
            return list.toArray(new String[0]);
        }
        if (input instanceof String[]) {
            return (String[]) input;
        }
        if (input != null && input.getClass().isArray()) {
            Object[] arr = (Object[]) input;
            String[] result = new String[arr.length];
            for (int i = 0; i < arr.length; i++) {
                result[i] = arr[i] != null ? arr[i].toString() : null;
            }
            return result;
        }
        return new String[0];
    }

    private static void initializeNativeLibraries() {
        String tempDir = System.getProperty("java.io.tmpdir");
        if (tempDir != null) {
            System.setProperty("org.bytedeco.javacpp.cachedir", tempDir);
        }

        tryLoadClass("org.bytedeco.openblas.global.openblas_nolapack", "OpenBLAS");
        tryLoadClass("org.bytedeco.opencv.global.opencv_core", "OpenCV");
    }

    private static void tryLoadClass(String className, String label) {
        try {
            Class.forName(className);
            logger.info("{} loaded successfully", label);
        } catch (ClassNotFoundException e) {
            logger.warn("{} classes not found: {}", label, e.getMessage());
        } catch (UnsatisfiedLinkError e) {
            logger.warn("{} native library deferred to lazy loading: {}", label, e.getMessage());
        }
    }
}
