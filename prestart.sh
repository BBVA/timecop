#! /usr/bin/env bash


echo "Initializing"
if ! whoami &> /dev/null; then
  echo "User not configured"
  if [ -w /etc/passwd ]; then
    echo "Updating /etc/passwd"
    echo "${USER_NAME:-default}:x:$(id -u):0:${USER_NAME:-default} user:${HOME}:/sbin/nologin" >> /etc/passwd
  fi
fi
echo "Running timecop"


service redis-server start
sleep 10;


cd /app; celery worker -A app.celery --concurrency=1  --loglevel=info &
