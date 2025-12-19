# Deployment Guide - Azure

## Overview
Deploy the Tunisian Real Estate Price Prediction Platform to Microsoft Azure using Virtual Machines and Docker.

## Prerequisites
- Azure account with $100 credit
- Azure CLI installed
- Docker knowledge
- SSH key pair

## Architecture on Azure

```
┌─────────────────────────────────────────┐
│         Azure Virtual Machine           │
│  ┌────────────────────────────────────┐ │
│  │  Docker Containers:                │ │
│  │  - Frontend (Nginx)                │ │
│  │  - Backend (FastAPI)               │ │
│  │  - MLflow Server                   │ │
│  │  - PostgreSQL                      │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
              │
              │ (Public IP)
              │
         ┌────▼────┐
         │ Internet │
         └─────────┘
```

## Step-by-Step Deployment

### 1. Install Azure CLI

**Windows:**
```powershell
winget install Microsoft.AzureCLI
```

**macOS:**
```bash
brew install azure-cli
```

**Linux:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### 2. Login to Azure

```bash
az login
```

### 3. Create Resource Group

```bash
az group create \
  --name real-estate-rg \
  --location eastus
```

### 4. Create Virtual Machine

```bash
az vm create \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard
```

**Recommended VM Sizes:**
- **Standard_B2s**: 2 vCPUs, 4 GB RAM (~$30/month) - Development
- **Standard_B2ms**: 2 vCPUs, 8 GB RAM (~$60/month) - Production
- **Standard_D2s_v3**: 2 vCPUs, 8 GB RAM (~$70/month) - High performance

### 5. Configure Network Security Group

```bash
# Allow HTTP
az vm open-port \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --port 80 \
  --priority 1001

# Allow Backend API
az vm open-port \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --port 8000 \
  --priority 1002

# Allow MLflow
az vm open-port \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --port 5000 \
  --priority 1003

# Allow HTTPS (optional)
az vm open-port \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --port 443 \
  --priority 1004
```

### 6. Get VM Public IP

```bash
az vm show \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --show-details \
  --query publicIps \
  --output tsv
```

Save this IP address!

### 7. SSH into VM

```bash
ssh azureuser@<VM_PUBLIC_IP>
```

### 8. Install Docker on VM

```bash
# Update packages
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version

# Logout and login again for group changes
exit
ssh azureuser@<VM_PUBLIC_IP>
```

### 9. Clone Your Repository

```bash
# Install git if needed
sudo apt-get install git -y

# Clone repo
git clone https://github.com/your-username/ML-project.git
cd ML-project
```

### 10. Configure Environment

```bash
# Backend
cd back
cp .env.example .env
nano .env  # Edit configurations

# Frontend
cd ../front
cp .env.example .env
nano .env
# Set: VITE_API_URL=http://<VM_PUBLIC_IP>:8000

# ML
cd ../ML
cp .env.example .env
nano .env

cd ..
```

### 11. Build and Run Containers

```bash
# Build and start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### 12. Verify Deployment

- **Frontend**: http://\<VM_PUBLIC_IP\>
- **Backend API**: http://\<VM_PUBLIC_IP\>:8000/docs
- **MLflow**: http://\<VM_PUBLIC_IP\>:5000

## Data Setup on Azure

### Upload Kaggle Dataset

```bash
# From your local machine
scp tunisia_real_estate.csv azureuser@<VM_PUBLIC_IP>:~/ML-project/ML/data/raw/

# Or download directly on VM
ssh azureuser@<VM_PUBLIC_IP>
cd ~/ML-project/ML/data/raw/
wget <dataset-url>
```

### Run Initial Training

```bash
ssh azureuser@<VM_PUBLIC_IP>
cd ~/ML-project

# Run ML training container
docker compose --profile training up ml_training
```

## Automated Backups

### Backup Script

```bash
# Create backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/azureuser/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker exec real_estate_db pg_dump -U postgres real_estate > \
  $BACKUP_DIR/db_backup_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz \
  ~/ML-project/back/models/ \
  ~/ML-project/ML/models/

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

```bash
chmod +x ~/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add line:
0 2 * * * /home/azureuser/backup.sh
```

## Monitoring

### System Monitoring

```bash
# Check Docker containers
docker compose ps

# Check logs
docker compose logs backend
docker compose logs frontend
docker compose logs mlflow

# Check system resources
docker stats

# Check disk usage
df -h
```

### Application Monitoring

```bash
# Backend health
curl http://localhost:8000/health

# MLflow
curl http://localhost:5000
```

## Scaling Considerations

### Vertical Scaling (Resize VM)

```bash
az vm resize \
  --resource-group real-estate-rg \
  --name real-estate-vm \
  --size Standard_B2ms
```

### Horizontal Scaling

For production with high traffic:
1. Use **Azure Container Instances** or **Azure Kubernetes Service**
2. Set up **Azure Load Balancer**
3. Use **Azure Database for PostgreSQL**
4. Store models in **Azure Blob Storage**

## SSL/HTTPS Setup

### Using Let's Encrypt

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

Update `docker-compose.yml` to expose port 443.

## Cost Optimization

### Azure Cost Calculator
Use the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) to estimate costs.

### Tips
1. **Stop VM when not in use**: `az vm deallocate`
2. **Use B-series burstable VMs** for development
3. **Delete unused resources**
4. **Monitor spending** in Azure Portal
5. **Set up budget alerts**

### Shutdown Schedule

```bash
# Stop VM at night (11 PM)
az vm deallocate \
  --resource-group real-estate-rg \
  --name real-estate-vm

# Start VM in morning (7 AM)
az vm start \
  --resource-group real-estate-rg \
  --name real-estate-vm
```

## Maintenance

### Update Application

```bash
ssh azureuser@<VM_PUBLIC_IP>
cd ~/ML-project

# Pull latest code
git pull origin main

# Rebuild containers
docker compose down
docker compose build
docker compose up -d
```

### Database Maintenance

```bash
# Enter PostgreSQL container
docker exec -it real_estate_db psql -U postgres

# Vacuum database
VACUUM ANALYZE;

# Check database size
\l+
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs <service_name>

# Restart service
docker compose restart <service_name>

# Rebuild
docker compose up -d --build <service_name>
```

### Out of Disk Space

```bash
# Check usage
df -h

# Clean Docker
docker system prune -a --volumes

# Clean old logs
sudo journalctl --vacuum-time=7d
```

### High Memory Usage

```bash
# Check memory
free -h

# Restart containers
docker compose restart
```

## Security Best Practices

1. **Change default passwords** in `.env`
2. **Use Azure Key Vault** for secrets
3. **Enable Azure Security Center**
4. **Regular security updates**: `sudo apt-get update && sudo apt-get upgrade`
5. **Use private networks** for internal communication
6. **Enable Azure Monitor** for logs

## Cleanup

When you're done:

```bash
# Delete resource group (deletes everything)
az group delete --name real-estate-rg --yes --no-wait

# Or stop VM to save costs
az vm deallocate \
  --resource-group real-estate-rg \
  --name real-estate-vm
```

## Next Steps

- Set up CI/CD with GitHub Actions
- Implement monitoring with Azure Monitor
- Add logging with Azure Log Analytics
- Set up auto-scaling rules
- Configure Azure CDN for frontend

## Support

For Azure support:
- [Azure Documentation](https://docs.microsoft.com/azure/)
- [Azure Support](https://azure.microsoft.com/support/)
- [Azure Community](https://techcommunity.microsoft.com/t5/azure/ct-p/Azure)
