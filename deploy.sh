#!/bin/bash
# Cloud Run Deployment Script for Image Translation API
# Usage: ./deploy.sh

set -e  # Exit on error

# Configuration
PROJECT_ID="markazqa-36bbe"  # Update with your GCP project ID
REGION="us-central1"
SERVICE_NAME="image-translation"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Image Translation API to Cloud Run"
echo "================================================"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Please install it first."
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker not found. Please install it first."
    exit 1
fi

# Set active project
echo "📝 Setting active GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build Docker image
echo ""
echo "🏗️  Building Docker image..."
echo "⚠️  Note: This will take 5-10 minutes due to model downloads"
docker build -t ${IMAGE_NAME}:latest .

# Push to Container Registry
echo ""
echo "📤 Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --memory 4Gi \
  --cpu 2 \
  --timeout 540 \
  --max-instances 10 \
  --min-instances 0 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY="${GEMINI_API_KEY}" \
  --set-env-vars GCP_PROJECT_ID="${PROJECT_ID}" \
  --set-env-vars GCP_BUCKET_NAME="${GCP_BUCKET_NAME}" \
  --set-env-vars GCP_CDN_URL="${GCP_CDN_URL}"

# Get the service URL
echo ""
echo "✅ Deployment complete!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo ""
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "Test endpoints:"
echo "  Health check: ${SERVICE_URL}/health"
echo "  API docs: ${SERVICE_URL}/docs"
echo ""
echo "📋 To view logs:"
echo "  gcloud run services logs tail ${SERVICE_NAME} --region ${REGION}"
