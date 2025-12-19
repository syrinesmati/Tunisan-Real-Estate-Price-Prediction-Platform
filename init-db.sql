-- Initialize databases for MLflow and application

-- Create MLflow database
CREATE DATABASE mlflow;

-- Create application database
CREATE DATABASE real_estate;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;
GRANT ALL PRIVILEGES ON DATABASE real_estate TO postgres;
