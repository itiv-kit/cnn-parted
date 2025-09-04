#!/bin/bash

# Cleanup script for CNNParted installation
# Removes unnecessary files and build artifacts after successful installation

echo "🧹 Cleaning up CNNParted installation..."

# Remove temporary documentation files that were created during development
echo "📝 Removing temporary documentation files..."
rm -f ACCELERGY_FINAL_FIX.md
rm -f ACCELERGY_FIX.md
rm -f CLEANUP_SUMMARY.md
rm -f FIX_SUMMARY.md
rm -f IMPROVEMENTS.md
rm -f MIGRATION.md
rm -f README_old.md

# Remove build artifacts from submodules
echo "🔨 Removing build artifacts..."
find tools/ -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name "obj_dbg" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name "*.o" -delete 2>/dev/null || true
find tools/ -name "*.so" -delete 2>/dev/null || true
find tools/ -name "*.a" -delete 2>/dev/null || true
find tools/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name "*.pyc" -delete 2>/dev/null || true

# Remove git-related files from submodules (keep them as source directories only)
echo "🗂️  Cleaning submodule git files..."
find tools/ -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name ".gitignore" -delete 2>/dev/null || true
find tools/ -name ".gitmodules" -delete 2>/dev/null || true

# Remove CI/test files that aren't needed for runtime
echo "🧪 Removing CI/test files..."
find tools/ -name ".github" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name ".gitlab-ci.yml" -delete 2>/dev/null || true
find tools/ -name "*.yml" -path "*/test/*" -delete 2>/dev/null || true

# Remove documentation files from submodules (keep only essential READMEs)
echo "📚 Removing non-essential documentation..."
find tools/ -name "doc" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name "docs" -type d -exec rm -rf {} + 2>/dev/null || true
find tools/ -name "*.md" -not -name "README.md" -delete 2>/dev/null || true

echo "✅ Cleanup complete!"
echo "📊 Essential files preserved, unnecessary build artifacts removed"
