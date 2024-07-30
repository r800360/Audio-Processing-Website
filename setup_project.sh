#!/bin/bash

# Create frontend directory structure
mkdir -p frontend/public
mkdir -p frontend/src/components
mkdir -p frontend/src/hooks
mkdir -p frontend/src/pages
mkdir -p frontend/src/styles

# Create backend directory structure
mkdir -p backend/src/controllers
mkdir -p backend/src/routes
mkdir -p backend/src/services
mkdir -p backend/uploads
mkdir -p backend/models

# Create machine learning directory structure
mkdir -p ml/models
mkdir -p ml/scripts
mkdir -p ml/data

# Create documentation directory
mkdir -p docs

# Create root files
touch .gitignore
touch docker-compose.yml
touch Dockerfile
touch README.md
touch frontend/package.json
touch backend/package.json
touch ml/requirements.txt
touch docs/API.md
touch docs/USER_GUIDE.md
touch docs/ARCHITECTURE.md
touch frontend/src/App.js
touch frontend/src/index.js
touch backend/src/app.js

# Print success message
echo "Project structure created successfully!"

