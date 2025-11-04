#!/bin/bash

# Test script for UDTF process method with URL
# This script directly tests the UDTF.process() method

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # Stay in java_codes directory

# Default test URLs (if no arguments provided)
DEFAULT_URL1="http://dzlk-cloud.oss-cn-zhangjiakou-gd.aliyuncs.com/zlkoss/socol/20250507/fc673e41c5384cdeb23b3cfd21ff2657/52196787/2025050714/1746598676496_143014e2-2b80-4124-b1c3-e57da2c43373.jpg"
DEFAULT_URL2="http://dzlk-cloud.oss-cn-zhangjiakou-gd.aliyuncs.com/zlkoss/socol/20250508/d18a491d15304b599df90ed10a04763c/3581110/2025050809/1746666136635_c7a7b064-7d4e-40dd-9dae-3be191012e90.webp"

# Collect test URLs from command line arguments (supports n images)
# If no arguments provided, use default 2 URLs for testing
TEST_URLS=()
if [ $# -eq 0 ]; then
    # No arguments: use default URLs
    TEST_URLS=("$DEFAULT_URL1" "$DEFAULT_URL2")
else
    # Arguments provided: use all arguments as URLs
    TEST_URLS=("$@")
fi

# Model paths (relative to java_codes directory)
MODEL_DIR="weights/onnx"
CLIP_MODEL="${MODEL_DIR}/clip_model.onnx"
LIQE_MODEL="${MODEL_DIR}/liqe_model.onnx"
TEXT_FEATURES="${MODEL_DIR}/text_features.json"

echo "=========================================="
echo "UDTF LIQE Test - Testing process() method (Batch)"
echo "=========================================="
echo ""
echo "Test URLs (${#TEST_URLS[@]} image(s)):"
for i in "${!TEST_URLS[@]}"; do
    echo "  [$((i+1))] ${TEST_URLS[$i]}"
done
echo ""

# Check if model files exist
if [ ! -f "$CLIP_MODEL" ]; then
    echo "Error: CLIP model not found at $CLIP_MODEL"
    echo "Please run the Python conversion script first:"
    echo "  python scripts/liqe_torch2onnx.py --output_dir weights/onnx"
    exit 1
fi

if [ ! -f "$LIQE_MODEL" ]; then
    echo "Error: LIQE model not found at $LIQE_MODEL"
    exit 1
fi

if [ ! -f "$TEXT_FEATURES" ]; then
    echo "Error: Text features not found at $TEXT_FEATURES"
    exit 1
fi

echo "✓ Model files found"
echo ""

# Check if compiled classes exist
if [ ! -d "target/classes" ]; then
    echo "Error: Java classes not compiled"
    echo "Please compile first:"
    echo "  ./compile_and_test.sh"
    exit 1
fi

# Build classpath
CLASSPATH="target/classes"
if [ -d "target/lib" ]; then
    for jar in target/lib/*.jar; do
        if [ -f "$jar" ]; then
            CLASSPATH="$CLASSPATH:$jar"
        fi
    done
fi

# Auto-detect and use Java 11+ if available
# First, try to use JAVA_HOME if set
if [ -n "$JAVA_HOME" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    JAVA_CMD="$JAVA_HOME/bin/java"
    echo "Using JAVA_HOME: $JAVA_HOME"
elif command -v /usr/libexec/java_home >/dev/null 2>&1; then
    # Try Java 25 first, then 17, then 11
    for version in 25 17 11; do
        JAVA_HOME_CANDIDATE=$(/usr/libexec/java_home -v "$version" 2>/dev/null)
        if [ -n "$JAVA_HOME_CANDIDATE" ] && [ -x "$JAVA_HOME_CANDIDATE/bin/java" ]; then
            export JAVA_HOME="$JAVA_HOME_CANDIDATE"
            JAVA_CMD="$JAVA_HOME/bin/java"
            echo "Auto-detected Java $version: $JAVA_HOME"
            break
        fi
    done
fi

# If still no Java found, try system java
if [ -z "$JAVA_CMD" ] || [ ! -x "$JAVA_CMD" ]; then
    JAVA_CMD="java"
fi

# Check Java version - need Java 11+
JAVA_VERSION=$($JAVA_CMD -version 2>&1 | head -1)
echo "Java version: $JAVA_VERSION"

# Try to detect Java version more accurately
JAVA_MAJOR_VERSION=""
# Check for Java 9+ format (version "11", "17", etc.)
if echo "$JAVA_VERSION" | grep -qE 'version "([0-9]+)'; then
    JAVA_MAJOR_VERSION=$(echo "$JAVA_VERSION" | grep -oE 'version "([0-9]+)' | grep -oE '[0-9]+')
fi
# Check for Java 8 format (version "1.8.0_xxx")
if [ -z "$JAVA_MAJOR_VERSION" ] && echo "$JAVA_VERSION" | grep -qE 'version "1\.8'; then
    JAVA_MAJOR_VERSION="8"
fi

if [ -z "$JAVA_MAJOR_VERSION" ]; then
    echo "⚠️  Could not detect Java version, but continuing..."
elif [ "$JAVA_MAJOR_VERSION" -lt 11 ]; then
    echo ""
    echo "⚠️  WARNING: Java version $JAVA_MAJOR_VERSION detected. Java 11+ is required."
    echo ""
    echo "Available Java versions on your system:"
    /usr/libexec/java_home -V 2>&1 | grep -E "^\s+\d+\.|^\s+Java" | head -5
    echo ""
    echo "Please set JAVA_HOME to point to Java 11 or higher:"
    echo "  export JAVA_HOME=\$(/usr/libexec/java_home -v 25)"
    echo "  # OR for Java 17:"
    echo "  export JAVA_HOME=\$(/usr/libexec/java_home -v 17)"
    echo "  # OR for Java 11:"
    echo "  export JAVA_HOME=\$(/usr/libexec/java_home -v 11)"
    echo ""
    echo "Then run this script again."
    echo ""
    exit 1
fi
echo ""

echo "Running UDTF test (testing process method)..."
echo ""

# Run test with necessary JVM arguments
$JAVA_CMD -cp "$CLASSPATH" \
    -Djna.nosys=true \
    -Dorg.bytedeco.javacpp.platform.macosx-arm64=true \
    com.autonavi.liqe.UDTFLIQETest \
    "$MODEL_DIR" \
    "${TEST_URLS[@]}"

