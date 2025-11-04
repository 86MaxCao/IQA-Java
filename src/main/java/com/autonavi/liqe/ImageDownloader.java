package com.autonavi.liqe;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imdecode;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR;

/**
 * Utility class for downloading images from URLs
 */
public class ImageDownloader {
    
    private static final Logger logger = LoggerFactory.getLogger(ImageDownloader.class);
    private static final int MAX_RETRIES = 3;
    private static final int TIMEOUT_MS = 10000;
    
    private final CloseableHttpClient httpClient;
    
    // Static initializer to ensure OpenCV native libraries are loaded
    static {
        try {
            // Force loading of OpenCV native libraries
            // This ensures native libraries are available before any OpenCV operations
            Class.forName("org.bytedeco.opencv.global.opencv_imgcodecs");
            logger.debug("OpenCV imgcodecs classes loaded successfully");
        } catch (ClassNotFoundException e) {
            logger.warn("OpenCV imgcodecs classes not found: {}", e.getMessage());
        } catch (UnsatisfiedLinkError e) {
            logger.error("Failed to load OpenCV native libraries: {}", e.getMessage());
            throw new RuntimeException("OpenCV native libraries could not be loaded. " +
                "This may be due to missing native libraries in ODPS environment.", e);
        }
    }
    
    public ImageDownloader() {
        this.httpClient = HttpClients.createDefault();
    }
    
    /**
     * Download image from URL and convert to OpenCV Mat
     * Retries up to MAX_RETRIES times on failure
     */
    public Mat downloadImage(String imageUrl) {
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                HttpGet request = new HttpGet(imageUrl);
                request.setHeader("User-Agent", "Mozilla/5.0");
                
                try (CloseableHttpResponse response = httpClient.execute(request)) {
                    int statusCode = response.getStatusLine().getStatusCode();
                    if (statusCode == 200) {
                        byte[] imageBytes = EntityUtils.toByteArray(response.getEntity());
                        return decodeImage(imageBytes);
                    } else {
                        logger.warn("Failed to download image from {}: HTTP {}", imageUrl, statusCode);
                    }
                }
            } catch (IOException e) {
                logger.warn("Error downloading image (attempt {}/{}): {}", attempt, MAX_RETRIES, e.getMessage());
                if (attempt == MAX_RETRIES) {
                    logger.error("Failed to download image after {} attempts: {}", MAX_RETRIES, imageUrl);
                }
            } catch (Exception e) {
                logger.error("Unexpected error downloading image: {}", imageUrl, e);
                break;
            }
        }
        
        // Return null on failure - caller should handle this
        return null;
    }
    
    /**
     * Decode image bytes to OpenCV Mat
     */
    private Mat decodeImage(byte[] imageBytes) {
        try {
            // Create Mat from byte array
            Mat encoded = new Mat(imageBytes);
            Mat decoded = imdecode(encoded, IMREAD_COLOR);
            encoded.close();
            
            if (decoded.empty()) {
                logger.warn("Failed to decode image");
                decoded.close();
                return null;
            }
            
            return decoded;
        } catch (Exception e) {
            logger.error("Error decoding image: {}", e.getMessage());
            return null;
        }
    }
    
    /**
     * Close HTTP client resources
     */
    public void close() {
        try {
            httpClient.close();
        } catch (IOException e) {
            logger.error("Error closing HTTP client: {}", e.getMessage());
        }
    }
}

