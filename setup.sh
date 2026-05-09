#!/bin/bash

SERVICE_NAME="voice-server"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
WORKDIR="/opt/code/voice-server"
EXEC_PATH="${WORKDIR}/voice-server > /dev/null 2>&1"

# 创建 systemd service 文件
echo "创建 systemd 服务: $SERVICE_PATH"
cat <<EOF | sudo tee "$SERVICE_PATH" > /dev/null
[Unit]
Description=${SERVICE_NAME}
After=syslog.target
After=network.target
After=mariadb.service
After=redis.service
Requires=mariadb.service redis.service

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${WORKDIR}
ExecStart=${EXEC_PATH} > /dev/null 2>&1
StandardOutput=null
StandardError=null
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 设置执行权限（确保程序存在）
if [ ! -x "$EXEC_PATH" ]; then
    echo "警告：可执行文件不存在或无权限: $EXEC_PATH"
else
    echo "已找到可执行文件: $EXEC_PATH"
fi

# 重新加载 systemd 并启用服务
echo "重新加载 systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo "启动服务 ${SERVICE_NAME}..."
sudo systemctl enable --now "$SERVICE_NAME"

# 查看服务状态
echo "服务状态："
sudo systemctl status "$SERVICE_NAME" --no-pager
