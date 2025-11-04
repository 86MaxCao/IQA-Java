#!/bin/bash

# Script to compile and test Java code without Maven
# This script manually downloads dependencies and compiles

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "LIQE Java UDTF - Manual Build Script"
echo "=========================================="
echo ""

# Check Java
if ! command -v java &> /dev/null; then
    echo "Error: Java not found. Please install Java 11+"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -n 1)
echo "Java version: $JAVA_VERSION"
echo ""

# Create directories
mkdir -p target/classes
mkdir -p target/lib
mkdir -p target/test-classes

echo "Step 1: Downloading dependencies..."
echo "-----------------------------------"
echo ""
echo "NOTE: You need to manually download the following JAR files to target/lib/:"
echo ""
echo "1. ONNX Runtime:"
echo "   wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.16.3/onnxruntime-1.16.3.jar -O target/lib/onnxruntime.jar"
echo ""
echo "2. Jackson:"
echo "   wget https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.15.2/jackson-databind-2.15.2.jar -O target/lib/jackson-databind.jar"
echo "   wget https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.15.2/jackson-core-2.15.2.jar -O target/lib/jackson-core.jar"
echo "   wget https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.15.2/jackson-annotations-2.15.2.jar -O target/lib/jackson-annotations.jar"
echo ""
echo "3. JavaCV (OpenCV):"
echo "   wget https://repo1.maven.org/maven2/org/bytedeco/javacv-platform/1.5.9/javacv-platform-1.5.9.jar -O target/lib/javacv-platform.jar"
echo ""
echo "4. Apache HttpClient:"
echo "   wget https://repo1.maven.org/maven2/org/apache/httpcomponents/httpclient/4.5.14/httpclient-4.5.14.jar -O target/lib/httpclient.jar"
echo "   wget https://repo1.maven.org/maven2/org/apache/httpcomponents/httpcore/4.4.16/httpcore-4.4.16.jar -O target/lib/httpcore.jar"
echo ""
echo "5. SLF4J:"
echo "   wget https://repo1.maven.org/maven2/org/slf4j/slf4j-api/2.0.7/slf4j-api-2.0.7.jar -O target/lib/slf4j-api.jar"
echo "   wget https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/2.0.7/slf4j-simple-2.0.7.jar -O target/lib/slf4j-simple.jar"
echo ""
echo "After downloading dependencies, run this script again to compile."
echo ""

# Check if dependencies exist
if [ ! -f "target/lib/onnxruntime.jar" ]; then
    echo "Dependencies not found. Please download them first."
    echo ""
    echo "Quick download command (if you have wget):"
    echo "  mkdir -p target/lib"
    echo "  cd target/lib"
    echo "  wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.16.3/onnxruntime-1.16.3.jar -O onnxruntime.jar"
    echo "  # ... (download other dependencies)"
    echo "  cd ../.."
    exit 1
fi

echo "Step 2: Compiling Java source..."
echo "-----------------------------------"

# Build classpath
CLASSPATH="target/classes"
for jar in target/lib/*.jar; do
    if [ -f "$jar" ]; then
        CLASSPATH="$CLASSPATH:$jar"
    fi
done

# Compile with Java 8 compatibility (ODPS requires Java 8)
# First compile mock ODPS classes if they exist
if [ -f "src/main/java/com/aliyun/odps/volume/FileSystem.java" ]; then
    javac -d target/classes -cp "$CLASSPATH" -source 8 -target 8 \
        src/main/java/com/aliyun/odps/volume/FileSystem.java \
        src/main/java/com/aliyun/odps/data/VolumeInfo.java 2>/dev/null || true
fi

# Then compile main classes
javac -d target/classes \
    -cp "$CLASSPATH" \
    -source 8 -target 8 \
    src/main/java/com/autonavi/liqe/*.java

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
else
    echo "✗ Compilation failed!"
    exit 1
fi

echo ""
echo "Step 3: Testing ONNX Runtime..."
echo "-----------------------------------"
echo ""
echo "To test ONNX Runtime, run:"
echo ""
echo "  java -cp \"target/classes:$CLASSPATH\" \\"
echo "    com.autonavi.liqe.SimpleONNXTest \\"
echo "    weights/onnx/clip_model.onnx \\"
echo "    weights/onnx/liqe_model.onnx"
echo ""
echo "Make sure ONNX model files exist in weights/onnx/ directory"
echo ""

