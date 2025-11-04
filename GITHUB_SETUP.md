# GitHub Repository Setup Guide

## Project Name

**Recommended:** `liqe-odps-udtf`

Alternative names:
- `liqe-java-udtf`
- `odps-liqe-udtf`

## Quick Setup

### Option 1: Using GitHub CLI (Recommended)

If you have GitHub CLI (`gh`) installed:

```bash
# Install GitHub CLI if not installed
# macOS: brew install gh
# Linux: see https://cli.github.com/manual/installation

# Login to GitHub
gh auth login

# Create repository and push
cd java_codes
./setup_git_repo.sh
gh repo create liqe-odps-udtf --public --source=. --remote=origin --push
```

### Option 2: Manual Setup

1. **Create repository on GitHub:**
   - Go to: https://github.com/new
   - Repository name: `liqe-odps-udtf`
   - Description: `LIQE Image Quality Assessment UDTF for ODPS - Java implementation with ONNX Runtime`
   - Visibility: Public (recommended) or Private
   - **Important:** Do NOT initialize with README, .gitignore, or license (we already have them)

2. **Initialize and push:**

```bash
cd java_codes

# Run setup script
./setup_git_repo.sh

# Set your GitHub username (replace YOUR_USERNAME)
export GITHUB_USERNAME=86MaxCao
git remote add origin https://github.com/86MaxCao/iqa-odps-udtf.git

# Make initial commit
git commit -m "Initial commit: LIQE ODPS UDTF implementation

- Java UDTF implementation for ODPS
- ONNX Runtime integration
- Batch image processing support
- Complete model conversion pipeline
- Comprehensive testing scripts"

# Push to GitHub
git branch -M main
git push -u origin main
```

## Repository Settings

After creating the repository, consider:

1. **Add topics/tags:**
   - `odps`
   - `maxcompute`
   - `image-quality-assessment`
   - `liqe`
   - `udtf`
   - `java`
   - `onnx-runtime`
   - `computer-vision`

2. **Add description:**
   ```
   LIQE Image Quality Assessment UDTF for Alibaba Cloud MaxCompute (ODPS). 
   Java implementation with ONNX Runtime for efficient batch processing.
   ```

3. **Enable GitHub Pages** (optional, for documentation)

4. **Add branch protection** (for main branch, if working in a team)

## Verification

After pushing, verify:
- ✅ README.md is displayed correctly
- ✅ All files are committed
- ✅ .gitignore is working (target/, *.jar, etc. are excluded)
- ✅ License is shown

## Next Steps

1. Add badges to README (if using GitHub Actions)
2. Set up GitHub Actions for CI/CD (optional)
3. Add CONTRIBUTING.md (if open source)
4. Add issue templates (if open source)

