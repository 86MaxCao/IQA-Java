# GitHub Releases Guide

## Why Use GitHub Releases?

**Best Practice**: Do NOT commit `.jar` files directly to the repository. Instead, use **GitHub Releases** to distribute binary files.

### Benefits:
1. ✅ **Repository Size**: Keeps repository clean and fast to clone
2. ✅ **Version Control**: Clear versioning of releases
3. ✅ **Download Tracking**: GitHub tracks download statistics
4. ✅ **Standard Practice**: Follows industry best practices (used by Maven, Gradle, etc.)
5. ✅ **Easy Access**: Users can download pre-built JARs without building from source

## How to Create a Release

### Step 1: Build the JAR

```bash
cd java_codes
./package_jar.sh
# Or using Maven:
mvn clean package
```

This creates: `target/iqa-udtf-1.0.0-jar-with-dependencies.jar`

### Step 2: Create GitHub Release

1. Go to your repository on GitHub
2. Click **"Releases"** → **"Create a new release"**
3. Fill in:
   - **Tag**: `v1.0.0` (follow semantic versioning)
   - **Title**: `v1.0.0 - Initial Release`
   - **Description**: 
     ```markdown
     ## Features
     - LIQE model support
     - ODPS UDTF integration
     - Batch processing
     
     ## Installation
     Download `iqa-udtf-1.0.0-jar-with-dependencies.jar` and follow the [deployment guide](README.md#odps-deployment).
     ```
4. **Upload JAR**: Drag and drop `target/iqa-udtf-1.0.0-jar-with-dependencies.jar`
5. Click **"Publish release"**

### Step 3: Update README

The README will automatically reference the release URL. Users can download via:

```bash
# Download latest release
wget https://github.com/86MaxCao/iqa-odps-udtf/releases/latest/download/iqa-udtf-1.0.0-jar-with-dependencies.jar

# Or download specific version
wget https://github.com/86MaxCao/iqa-odps-udtf/releases/download/v1.0.0/iqa-udtf-1.0.0-jar-with-dependencies.jar
```

## Release Checklist

- [ ] Build JAR with `./package_jar.sh`
- [ ] Test JAR locally
- [ ] Verify JAR size (~200MB is expected)
- [ ] Create release tag
- [ ] Write release notes
- [ ] Upload JAR file
- [ ] Update README with download links
- [ ] Test download link works

## Versioning

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes (e.g., 2.0.0)
- **MINOR**: New features, backward compatible (e.g., 1.1.0)
- **PATCH**: Bug fixes (e.g., 1.0.1)

## Alternative: GitHub Actions

For automated releases, you can set up GitHub Actions to automatically build and release on tag push.

