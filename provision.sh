#!/bin/bash
# =============================================================================
# ComfyUI Provisioning Script für vast.ai
# Base Image: nvidia/cuda:12.8.0-devel-ubuntu22.04
# Includes: PyTorch + CUDA 12.8, SageAttention, ComfyUI-Manager, KJNodes
# =============================================================================
set -e

LOG="/var/log/provision.log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo " ComfyUI Provisioning Start: $(date)"
echo "========================================"


# =============================================================================
# ██████╗  ██████╗ ██╗    ██╗███╗   ██╗██╗      ██████╗  █████╗ ██████╗ 
# ██╔══██╗██╔═══██╗██║    ██║████╗  ██║██║     ██╔═══██╗██╔══██╗██╔══██╗
# ██║  ██║██║   ██║██║ █╗ ██║██╔██╗ ██║██║     ██║   ██║███████║██║  ██║
# ██║  ██║██║   ██║██║███╗██║██║╚██╗██║██║     ██║   ██║██╔══██║██║  ██║
# ██████╔╝╚██████╔╝╚███╔███╔╝██║ ╚████║███████╗╚██████╔╝██║  ██║██████╔╝
# ╚═════╝  ╚═════╝  ╚══╝╚══╝ ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ 
# 
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  DOWNLOAD LISTS – URLs hier eintragen                                   │
# │  Format pro Zeile: URL [optionaler_dateiname]                           │
# │  Leerzeilen und Zeilen mit # werden ignoriert                           │
# └─────────────────────────────────────────────────────────────────────────┘

# Checkpoints  →  ComfyUI/models/checkpoints/
CHECKPOINTS=(
    "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/blob/main/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"
)

# LoRAs  →  ComfyUI/models/loras/
LORAS=(
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/blob/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors"
    "https://huggingface.co/lightx2v/Wan2.2-Lightning/blob/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors"
    
)

# Text Encoder / CLIP  →  ComfyUI/models/clip/
TEXT_ENCODERS=(
    "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/blob/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors"
)

# VAE  →  ComfyUI/models/vae/
VAE=(
    "https://huggingface.co/wangkanai/wan22-vae/blob/main/vae/wan/wan22-vae.safetensors"
)

# Upscale Models  →  ComfyUI/models/upscale_models/
UPSCALE_MODELS=(
    # "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
)

# ControlNet  →  ComfyUI/models/controlnet/
CONTROLNET=(
    # "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.safetensors controlnet_canny.safetensors"
)

# UNET / Diffusion Models  →  ComfyUI/models/unet/
UNET=(
    "https://huggingface.co/BigDannyPt/Wan-2.2-Remix-GGUF/blob/main/I2V/v3.0/High/wan22RemixT2VI2V_i2vHighV30-Q8_0.gguf"
    "https://huggingface.co/BigDannyPt/Wan-2.2-Remix-GGUF/blob/main/I2V/v3.0/Low/wan22RemixT2VI2V_i2vLowV30-Q8_0.gguf"
)

# Embeddings / Textual Inversions  →  ComfyUI/models/embeddings/
EMBEDDINGS=(
    # "https://civitai.com/api/download/models/XXXXX embedding.pt"
)

# CivitAI API Token (benötigt für manche Downloads)
# Einfach hier eintragen oder als Umgebungsvariable CIVITAI_TOKEN setzen:
CIVITAI_TOKEN="${CIVITAI_TOKEN:-}"


# =============================================================================
# SYSTEM PAKETE
# =============================================================================
echo ""
echo ">>> [1/7] System-Pakete installieren..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    aria2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libgomp1 \
    ninja-build \
    build-essential \
    cmake \
    ca-certificates \
    > /dev/null

# Python 3.11 als Standard
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1
python3 -m pip install --upgrade pip setuptools wheel -q

echo "    ✓ System-Pakete fertig"


# =============================================================================
# PYTORCH MIT CUDA 12.8
# =============================================================================
echo ""
echo ">>> [2/7] PyTorch mit CUDA 12.8 installieren..."

pip install --upgrade \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    -q

# Schnelltest
python3 -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    CUDA verfügbar: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo "    ✓ PyTorch fertig"


# =============================================================================
# SAGE ATTENTION
# =============================================================================
echo ""
echo ">>> [3/7] SageAttention installieren..."

# Triton (Voraussetzung)
pip install triton -q

# SageAttention aus Source (für CUDA 12.8 Kompatibilität)
pip install sageattention -q || {
    echo "    Pip-Install fehlgeschlagen, kompiliere aus Source..."
    cd /tmp
    git clone https://github.com/thu-ml/SageAttention.git --depth=1
    cd SageAttention
    pip install -e . -q
    cd /
}

echo "    ✓ SageAttention fertig"


# =============================================================================
# COMFYUI
# =============================================================================
echo ""
echo ">>> [4/7] ComfyUI installieren..."

COMFY_DIR="/workspace/ComfyUI"

if [ ! -d "$COMFY_DIR" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" --depth=1
else
    echo "    ComfyUI-Verzeichnis existiert bereits, update..."
    cd "$COMFY_DIR" && git pull
fi

cd "$COMFY_DIR"
pip install -r requirements.txt -q

# Modell-Verzeichnisse anlegen
for dir in checkpoints loras clip vae upscale_models controlnet unet embeddings clip_vision; do
    mkdir -p "$COMFY_DIR/models/$dir"
done

echo "    ✓ ComfyUI fertig"


# =============================================================================
# CUSTOM NODES
# =============================================================================
echo ""
echo ">>> [5/7] Custom Nodes installieren..."

CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

install_node() {
    local name="$1"
    local url="$2"
    local dir="$CUSTOM_NODES_DIR/$name"
    if [ ! -d "$dir" ]; then
        echo "    → $name installieren..."
        git clone "$url" "$dir" --depth=1 -q
        if [ -f "$dir/requirements.txt" ]; then
            pip install -r "$dir/requirements.txt" -q
        fi
    else
        echo "    → $name bereits vorhanden, update..."
        cd "$dir" && git pull -q
    fi
}

install_node "ComfyUI-Manager" \
    "https://github.com/ltdrdata/ComfyUI-Manager.git"

install_node "ComfyUI-KJNodes" \
    "https://github.com/kijai/ComfyUI-KJNodes.git"

echo "    ✓ Custom Nodes fertig"


# =============================================================================
# MODELL-DOWNLOADER
# =============================================================================
echo ""
echo ">>> [6/7] Modelle herunterladen..."

# Download-Funktion mit aria2c (schnell, multi-connection)
download_model() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"

    # CivitAI Token anhängen falls vorhanden
    local dl_url="$url"
    if [[ "$url" == *"civitai.com"* ]] && [ -n "$CIVITAI_TOKEN" ]; then
        if [[ "$url" == *"?"* ]]; then
            dl_url="${url}&token=${CIVITAI_TOKEN}"
        else
            dl_url="${url}?token=${CIVITAI_TOKEN}"
        fi
    fi

    # Dateinamen ermitteln
    if [ -z "$filename" ]; then
        # Aus URL ableiten (letztes Segment, ohne Query-Parameter)
        filename=$(basename "${url%%\?*}")
    fi

    local dest_path="$dest_dir/$filename"

    if [ -f "$dest_path" ]; then
        echo "    → Überspringe (existiert): $filename"
        return 0
    fi

    echo "    → Lade: $filename"
    mkdir -p "$dest_dir"

    aria2c \
        --console-log-level=error \
        --summary-interval=0 \
        -x 8 -s 8 \
        --file-allocation=none \
        -d "$dest_dir" \
        -o "$filename" \
        "$dl_url" \
    || wget -q --show-progress \
        -O "$dest_path" \
        "$dl_url"
}

# Liste abarbeiten: "URL [optionaler_dateiname]"
process_list() {
    local dest_dir="$1"
    shift
    local entries=("$@")
    for entry in "${entries[@]}"; do
        [[ -z "$entry" || "$entry" == \#* ]] && continue
        read -r url filename <<< "$entry"
        download_model "$url" "$dest_dir" "$filename"
    done
}

process_list "$COMFY_DIR/models/checkpoints"    "${CHECKPOINTS[@]+"${CHECKPOINTS[@]}"}"
process_list "$COMFY_DIR/models/loras"          "${LORAS[@]+"${LORAS[@]}"}"
process_list "$COMFY_DIR/models/clip"           "${TEXT_ENCODERS[@]+"${TEXT_ENCODERS[@]}"}"
process_list "$COMFY_DIR/models/vae"            "${VAE[@]+"${VAE[@]}"}"
process_list "$COMFY_DIR/models/upscale_models" "${UPSCALE_MODELS[@]+"${UPSCALE_MODELS[@]}"}"
process_list "$COMFY_DIR/models/controlnet"     "${CONTROLNET[@]+"${CONTROLNET[@]}"}"
process_list "$COMFY_DIR/models/unet"           "${UNET[@]+"${UNET[@]}"}"
process_list "$COMFY_DIR/models/embeddings"     "${EMBEDDINGS[@]+"${EMBEDDINGS[@]}"}"

echo "    ✓ Downloads fertig"

pip install jupyterlab -q

nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    > /var/log/jupyter.log 2>&1 &

echo "✓ JupyterLab läuft auf Port 8888"

# =============================================================================
# COMFYUI STARTEN
# =============================================================================
echo ""
echo ">>> [7/7] ComfyUI starten..."

cd "$COMFY_DIR"

# SageAttention aktivieren (--use-sage-attention ab ComfyUI 0.3.x)
nohup python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --use-sage-attention \
    --preview-method auto \
    > /var/log/comfyui.log 2>&1 &

COMFY_PID=$!
echo "    ComfyUI PID: $COMFY_PID"

# Warten bis ComfyUI erreichbar ist
echo "    Warte auf ComfyUI..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8188 > /dev/null 2>&1; then
        echo "    ✓ ComfyUI läuft auf Port 8188"
        break
    fi
    sleep 2
done

echo ""
echo "========================================"
echo " Provisioning abgeschlossen: $(date)"
echo " Log: /var/log/provision.log"
echo " ComfyUI Log: /var/log/comfyui.log"
echo " ComfyUI URL: http://0.0.0.0:8188"
echo "========================================"
