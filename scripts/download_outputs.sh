#!/bin/bash
# Download outputs from GCloud VM to local machine
# Usage: ./scripts/download_outputs.sh [local_destination]

set -e

PROJECT="user-turn-lora"
ZONE="us-central1-a"
VM_NAME="userturn-lora-a100"
REMOTE_PATH="/home/sebastianboehler/output"
LOCAL_DEST="${1:-.}"  # Default to current directory

echo "Downloading outputs from $VM_NAME..."
echo "Remote: $REMOTE_PATH"
echo "Local: $LOCAL_DEST"
echo ""

# Create local destination if it doesn't exist
mkdir -p "$LOCAL_DEST"

# Use gcloud scp to download
gcloud compute scp --recurse \
    --project="$PROJECT" \
    --zone="$ZONE" \
    "$VM_NAME:$REMOTE_PATH/*" \
    "$LOCAL_DEST/"

echo ""
echo "Download complete!"
echo "Files saved to: $LOCAL_DEST"
ls -la "$LOCAL_DEST"
