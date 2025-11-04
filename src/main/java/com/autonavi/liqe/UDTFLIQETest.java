package com.autonavi.liqe;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Test class that directly tests UDTF process method
 * Simulates ODPS UDTF environment for testing
 */
public class UDTFLIQETest {
    
    private static final Logger logger = LoggerFactory.getLogger(UDTFLIQETest.class);
    
    // Default test URL
    private static final String DEFAULT_TEST_URL = 
        "http://dzlk-cloud.oss-cn-zhangjiakou-gd.aliyuncs.com/zlkoss/socol/20250507/fc673e41c5384cdeb23b3cfd21ff2657/52196787/2025050714/1746598676496_143014e2-2b80-4124-b1c3-e57da2c43373.jpg";
    
    /**
     * Mock UDTFCollector to capture output
     */
    static class MockUDTFCollector implements com.aliyun.odps.udf.UDTFCollector {
        private List<Object[]> results = new ArrayList<>();
        
        @Override
        public void collect(Object[] row) {
            results.add(row);
            // Print for debugging
            System.out.println("  Output: url=" + row[0] + ", score=" + row[1]);
        }
        
        public List<Object[]> getResults() {
            return results;
        }
        
        public void clear() {
            results.clear();
        }
    }
    
    /**
     * Mock ExecutionContext for testing
     * Note: We cannot extend ExecutionContext directly because its constructor
     * requires ODPS internal classes that are not available in test environment.
     * Since UDTFLIQE.setup() doesn't actually use any ExecutionContext methods,
     * we can pass null or create a minimal wrapper.
     * 
     * However, since setup() expects ExecutionContext, we'll use reflection to
     * bypass the constructor issue, or modify the test to not use ExecutionContext.
     */
    static class MockExecutionContext {
        public MockExecutionContext() {
            // Empty constructor - no ODPS dependencies
        }
    }
    
    /**
     * Test UDTF directly
     */
    public static void main(String[] args) {
        String modelDir = null;
        String[] testUrls = null;
        
        if (args.length >= 1) {
            modelDir = args[0];
            if (args.length > 1) {
                // Use provided URLs
                testUrls = new String[args.length - 1];
                System.arraycopy(args, 1, testUrls, 0, args.length - 1);
            } else {
                // Use default test URL
                testUrls = new String[]{DEFAULT_TEST_URL};
            }
        } else {
            System.out.println("Usage:");
            System.out.println("  UDTFLIQETest <model_directory> [url1] [url2] ...");
            System.out.println("");
            System.out.println("Example:");
            System.out.println("  UDTFLIQETest weights/onnx");
            System.out.println("  UDTFLIQETest weights/onnx " + DEFAULT_TEST_URL);
            System.exit(1);
        }
        
        String clipModelPath = modelDir + "/clip_model.onnx";
        String liqeModelPath = modelDir + "/liqe_model.onnx";
        String textFeaturesPath = modelDir + "/text_features.json";
        
        System.out.println(repeatString("=", 60));
        System.out.println("UDTF LIQE Test - Testing process() method directly");
        System.out.println(repeatString("=", 60));
        System.out.println();
        
        // Create mock collector
        MockUDTFCollector collector = new MockUDTFCollector();
        
        // Create UDTF instance
        UDTFLIQE udtf = new UDTFLIQE();
        
        try {
            // Setup UDTF (this will try to load models)
            // Set system properties to tell UDTF where to find models
            System.out.println("Setting up UDTF...");
            System.out.println("  CLIP model: " + clipModelPath);
            System.out.println("  LIQE model: " + liqeModelPath);
            System.out.println("  Text features: " + textFeaturesPath);
            System.out.println();
            
            // Set system properties so UDTF can find the models
            System.setProperty("liqe.clip_model.path", clipModelPath);
            System.setProperty("liqe.liqe_model.path", liqeModelPath);
            System.setProperty("liqe.text_features.path", textFeaturesPath);
            
            // Setup UDTF - use reflection to avoid ExecutionContext constructor dependency
            // Since UDTFLIQE.setup() doesn't actually use the ExecutionContext parameter,
            // we can pass null or create a proxy using reflection
            try {
                // Try to create ExecutionContext using reflection (if possible)
                // Otherwise, we'll need to modify the approach
                java.lang.reflect.Method setupMethod = UDTFLIQE.class.getMethod("setup", com.aliyun.odps.udf.ExecutionContext.class);
                
                // Try to create a minimal ExecutionContext proxy
                // Since ExecutionContext constructor requires ODPS internals, we'll use null
                // and catch the exception if setup actually needs it
                try {
                    // Try calling with null - if setup doesn't use ctx, this will work
                    setupMethod.invoke(udtf, (com.aliyun.odps.udf.ExecutionContext) null);
                } catch (Exception e) {
                    // If null doesn't work, try creating a minimal proxy
                    // Create ExecutionContext using reflection to bypass constructor
                    java.lang.reflect.Constructor<?> ctxConstructor = com.aliyun.odps.udf.ExecutionContext.class.getDeclaredConstructor();
                    ctxConstructor.setAccessible(true);
                    com.aliyun.odps.udf.ExecutionContext ctx = (com.aliyun.odps.udf.ExecutionContext) ctxConstructor.newInstance();
                    setupMethod.invoke(udtf, ctx);
                }
            } catch (Exception e) {
                // If reflection fails, try direct call with null (may work if ctx is not used)
                System.out.println("Warning: Could not use reflection, trying direct call...");
                try {
                    udtf.setup(null);
                } catch (Exception e2) {
                    throw new RuntimeException("Failed to setup UDTF. ExecutionContext cannot be mocked in test environment. " +
                        "Consider testing UDTF logic through LIQETest instead.", e2);
                }
            }
            
            // Set collector manually for testing
            udtf.setCollector(collector);
            
            // Test process method
            System.out.println("Testing process() method with " + testUrls.length + " URL(s)...");
            System.out.println();
            
            for (String url : testUrls) {
                System.out.println("Input URL: " + url);
            }
            System.out.println();
            
            // Call process method (simulating ODPS UDTF call)
            // process() expects: Object[] args where args[0] is String[] (array of URLs)
            Object[] processArgs = new Object[]{testUrls};
            udtf.process(processArgs);
            
            // Check results
            List<Object[]> results = collector.getResults();
            
            System.out.println();
            System.out.println(repeatString("=", 60));
            System.out.println("Test Results");
            System.out.println(repeatString("=", 60));
            System.out.println();
            
            if (results.size() != testUrls.length) {
                System.out.println("⚠ WARNING: Output count (" + results.size() + 
                    ") does not match input count (" + testUrls.length + ")");
            } else {
                System.out.println("✓ Output count matches input count: " + results.size());
            }
            System.out.println();
            
            // Print results
            for (int i = 0; i < results.size(); i++) {
                Object[] row = results.get(i);
                String url = (String) row[0];
                Double score = (Double) row[1];
                
                System.out.println("[" + (i + 1) + "] " + url);
                System.out.println("    Quality Score: " + String.format("%.4f", score));
                
                if (score < 0) {
                    System.out.println("    Status: ERROR (processing failed)");
                } else if (score >= 1.0 && score <= 5.0) {
                    System.out.println("    Status: SUCCESS");
                } else {
                    System.out.println("    Status: WARNING (score out of range)");
                }
                System.out.println();
            }
            
            // Summary
            long successCount = results.stream()
                .mapToLong(row -> {
                    Double score = (Double) row[1];
                    return (score >= 1.0 && score <= 5.0) ? 1 : 0;
                })
                .sum();
            
            long errorCount = results.stream()
                .mapToLong(row -> {
                    Double score = (Double) row[1];
                    return (score < 0) ? 1 : 0;
                })
                .sum();
            
            System.out.println("Summary:");
            System.out.println("  Total: " + results.size());
            System.out.println("  Success: " + successCount);
            System.out.println("  Errors: " + errorCount);
            System.out.println();
            
        } catch (Exception e) {
            System.err.println("\n✗ Error in test: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } finally {
            // Cleanup
            udtf.close();
        }
        
        System.out.println(repeatString("=", 60));
        System.out.println("Test completed!");
        System.out.println(repeatString("=", 60));
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
