# React Dockerfile
FROM node:23.10.0-alpine

WORKDIR /app

COPY apps/web/package.json apps/web/package-lock.json ./

RUN npm install --legacy-peer-deps

COPY apps/web ./
COPY data /data

# Set environment variables for development server
ENV WDS_SOCKET_HOST=0.0.0.0
ENV WDS_SOCKET_PORT=8080
ENV DANGEROUSLY_DISABLE_HOST_CHECK=true

# Expose port
EXPOSE 8080

# Start the development server with special flags
CMD ["npm", "start", "--", "--host", "0.0.0.0"]
