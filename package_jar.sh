#!/bin/bash

# Package LIQE UDTF into JAR file
# This script supports both Maven and manual packaging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "LIQE UDTF JAR Packaging"
echo "=========================================="
echo ""

# Check if Maven is available
if command -v mvn >/dev/null 2>&1; then
    echo "✓ Maven found, using Maven to package..."
    echo ""
    
    # Method 1: Use Maven (recommended)
    echo "Step 1: Cleaning previous build..."
    mvn clean
    
    echo ""
    echo "Step 2: Compiling Java source..."
    mvn compile
    
    echo ""
    echo "Step 3: Packaging JAR with dependencies..."
    mvn package
    
    echo ""
    echo "✓ JAR files created:"
    echo "  - target/iqa-udtf-1.0.0.jar (main JAR)"
    echo "  - target/iqa-udtf-1.0.0-jar-with-dependencies.jar (fat JAR with all dependencies)"
    echo ""
    echo "📦 For ODPS deployment, use: target/iqa-udtf-1.0.0-jar-with-dependencies.jar"
    
else
    echo "⚠️  Maven not found, using manual packaging..."
    echo ""
    
    # Method 2: Manual packaging (without Maven)
    echo "Step 1: Checking compiled classes..."
    if [ ! -d "target/classes" ]; then
        echo "Error: Classes not compiled. Please run ./compile_and_test.sh first"
        exit 1
    fi
    
    echo "Step 2: Creating JAR structure..."
    JAR_DIR="target/jar-temp"
    rm -rf "$JAR_DIR"
    mkdir -p "$JAR_DIR"
    
    # Copy compiled classes
    echo "  - Copying compiled classes..."
    cp -r target/classes/* "$JAR_DIR/"
    
    # Copy dependencies
    echo "  - Extracting dependency JARs..."
    if [ ! -d "target/lib" ]; then
        echo "Error: Dependencies not found in target/lib. Please run ./compile_and_test.sh first"
        exit 1
    fi
    
    # Extract all JARs into the same directory
    # IMPORTANT: Extract in order to preserve native libraries (especially Linux .so files)
    for jar in target/lib/*.jar; do
        if [ -f "$jar" ]; then
            echo "    Extracting: $(basename $jar)"
            # Use unzip if available (more reliable), otherwise use jar
            JAR_PATH=$(cd "$(dirname "$jar")" && pwd)/$(basename "$jar")
            if command -v unzip >/dev/null 2>&1; then
                # Use -o to overwrite (important for native libraries)
                # Preserve native library files (.so, .dylib, .dll)
                (cd "$JAR_DIR" && unzip -q -o "$JAR_PATH" 2>&1 | grep -v "warning\|Archive\|inflating:" || true)
            else
                (cd "$JAR_DIR" && jar xf "$JAR_PATH" 2>&1 | grep -v "warning" || true)
            fi
        fi
    done
    
    # Verify Linux native libraries are present (for ODPS)
    echo "  - Verifying native libraries..."
    LINUX_LIBS=$(find "$JAR_DIR" -name "*.so" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$LINUX_LIBS" -gt 0 ]; then
        echo "    ✓ Found $LINUX_LIBS Linux native libraries (.so files)"
    else
        echo "    ⚠ WARNING: No Linux native libraries (.so files) found!"
        echo "    This may cause issues in ODPS (Linux environment)."
    fi
    
    # Remove duplicate META-INF files (keep only MANIFEST.MF)
    echo "  - Cleaning META-INF..."
    rm -rf "$JAR_DIR/META-INF"/*.SF "$JAR_DIR/META-INF"/*.RSA "$JAR_DIR/META-INF"/*.DSA 2>/dev/null || true
    
    # Create manifest
    echo "  - Creating MANIFEST.MF..."
    mkdir -p "$JAR_DIR/META-INF"
    cat > "$JAR_DIR/META-INF/MANIFEST.MF" << 'EOF'
Manifest-Version: 1.0
Created-By: LIQE UDTF Package Script
Main-Class: com.autonavi.liqe.UDTFLIQE

EOF
    
    echo ""
    echo "Step 3: Creating JAR file..."
    JAR_NAME="iqa-udtf-1.0.0-jar-with-dependencies.jar"
    cd "$JAR_DIR"
    jar cf "../$JAR_NAME" . 2>/dev/null || {
        # If jar command fails, try using zip
        cd "$SCRIPT_DIR"
        cd "$JAR_DIR"
        zip -r "../$JAR_NAME" . > /dev/null
        cd "$SCRIPT_DIR"
    }
    cd "$SCRIPT_DIR"
    
    # Cleanup temp directory
    rm -rf "$JAR_DIR"
    
    echo ""
    echo "✓ JAR file created: target/$JAR_NAME"
    echo ""
    echo "📦 For ODPS deployment, use: target/$JAR_NAME"
fi

echo ""
echo "=========================================="
echo "Packaging completed!"
echo "=========================================="
echo ""
echo "Next steps for ODPS deployment:"
echo "1. Upload the fat JAR (jar-with-dependencies.jar) to ODPS"
echo "2. Upload model files (clip_model.onnx, liqe_model.onnx, text_features.json)"
echo "3. Create UDTF function pointing to: com.autonavi.liqe.UDTFLIQE"
echo ""

