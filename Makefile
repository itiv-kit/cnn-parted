# CNNParted Simplified Build System
.PHONY: help install dev clean test check-deps

# Default target
help:
	@echo "CNNParted Build System"
	@echo "===================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Full installation with all tools"
	@echo "  dev         - Development setup (Python tools only)"
	@echo "  clean       - Clean up installation artifacts"
	@echo "  test        - Run basic functionality test"
	@echo "  check-deps  - Check system dependencies"
	@echo ""
	@echo "Usage examples:"
	@echo "  make install   # Full installation"
	@echo "  make dev       # Quick development setup"
	@echo "  make clean     # Cleanup after installation"

# Full installation using simplified script
install:
	@echo "🚀 Running full installation..."
	./install.sh

# Development setup using simplified script
dev:
	@echo "🛠️  Running development setup..."
	./dev_setup.sh

# Clean up artifacts
clean:
	@echo "🧹 Cleaning up..."
	@if [ -f "cleanup_installation.sh" ]; then \
		./cleanup_installation.sh; \
	else \
		echo "⚠️  cleanup_installation.sh not found"; \
	fi

# Basic functionality test
test:
	@echo "🧪 Testing installation..."
	@if [ -f "activate_env.sh" ]; then \
		echo "Testing full installation..."; \
		bash -c "source activate_env.sh && which cnnparted && echo '✅ Full installation working'"; \
	elif [ -f "activate_dev.sh" ]; then \
		echo "Testing development installation..."; \
		bash -c "source activate_dev.sh && which cnnparted && echo '✅ Dev installation working'"; \
	else \
		echo "❌ No installation found. Run 'make install' or 'make dev' first"; \
		exit 1; \
	fi

# Check for required system dependencies
check-deps:
	@echo "🔍 Checking system dependencies..."
	@echo "Basic build tools:"
	@for cmd in git make gcc g++ python3; do \
		if ! command -v $$cmd >/dev/null 2>&1; then \
			echo "❌ Missing: $$cmd"; \
			echo "   Install with: sudo apt-get install build-essential git python3 python3-dev"; \
			exit 1; \
		else \
			echo "✅ Found: $$cmd"; \
		fi \
	done
	@echo ""
	@echo "Additional dependencies:"
	@for cmd in scons clang; do \
		if ! command -v $$cmd >/dev/null 2>&1; then \
			echo "❌ Missing: $$cmd"; \
			echo "   Install with: sudo apt-get install $$cmd"; \
			exit 1; \
		else \
			echo "✅ Found: $$cmd"; \
		fi \
	done
	@echo ""
	@echo "Development libraries:"
	@for lib in libconfig++-dev libboost-iostreams-dev; do \
		if ! dpkg -l | grep -q $$lib 2>/dev/null; then \
			echo "❌ Missing: $$lib"; \
			echo "   Install with: sudo apt-get install $$lib libboost-serialization-dev libyaml-cpp-dev libboost-thread-dev"; \
			exit 1; \
		else \
			echo "✅ Found: $$lib"; \
		fi \
	done
	@echo "✅ All required dependencies found"
