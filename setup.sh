#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything
#   ./setup.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "╔════════════════════════════════╗"
echo "║       Memex Bootstrap         ║"
echo "╚════════════════════════════════╝"
echo ""

# Check Docker
echo "[1/5] Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker first."
    exit 1
fi
echo "  ✓ Docker running"

# Start services
echo "[2/5] Starting services..."
docker compose up -d --build
echo "  ✓ Services started"

# Wait for healthy
echo "[3/5] Waiting for services to be healthy..."
for svc in qdrant ollama docling redis ml-services; do
    echo -n "  Waiting for $svc..."
    until docker compose ps "$svc" 2>/dev/null | grep -q "healthy"; do
        sleep 3
    done
    echo " ✓"
done

# Pull Ollama model
echo "[4/5] Pulling embedding model (bge-m3)..."
docker compose exec -T ollama ollama pull bge-m3
echo "  ✓ bge-m3 ready"

# Warm up ML services
echo "[5/5] Warming up ML services..."
curl -s http://localhost:5002/health > /dev/null && echo "  ✓ ML services healthy" || echo "  ⚠ ML services still loading"

echo "╔══════════════════════════════════════╗"
echo "║     Bootstrap Complete              ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Services running:"
docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || docker compose ps
echo ""
echo "  Run MCP:  uv run memex"
echo ""
