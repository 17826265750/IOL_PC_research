# IOL PC Research 部署指南

## 快速开始

### 前置要求

- Docker 20.10+
- Docker Compose 2.0+

### 一键部署

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 访问应用

- **前端界面**: http://localhost
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

---

## 详细配置

### 环境变量

复制环境变量模板并根据需要修改：

```bash
cd backend
cp .env.example .env
```

主要配置项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接字符串 | SQLite |
| `CORS_ORIGINS` | 允许的跨域来源 | `["http://localhost"]` |
| `SECRET_KEY` | 安全密钥 | 需要修改 |

### 数据持久化

数据存储在 Docker volume `iol-data` 中。查看数据位置：

```bash
docker volume inspect iol-data
```

备份数据：

```bash
# 创建备份
docker run --rm -v iol-data:/data -v $(pwd):/backup alpine tar czf /backup/iol-backup-$(date +%Y%m%d).tar.gz /data

# 恢复备份
docker run --rm -v iol-data:/data -v $(pwd):/backup alpine tar xzf /backup/iol-backup-YYYYMMDD.tar.gz -C /
```

---

## 常用命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 重新构建
docker-compose up -d --build

# 查看后端日志
docker-compose logs -f backend

# 查看前端日志
docker-compose logs -f frontend

# 进入后端容器
docker exec -it iol-backend /bin/bash

# 清理所有容器和镜像
docker-compose down --rmi all -v
```

---

## 生产环境部署

### 1. 修改安全配置

```bash
# 生成随机密钥
openssl rand -hex 32
```

在 `.env` 中设置：
```
SECRET_KEY=<生成的密钥>
```

### 2. 配置 HTTPS

使用 Let's Encrypt 获取免费证书：

```bash
# 安装 certbot
apt-get install certbot

# 获取证书
certbot certonly --standalone -d your-domain.com
```

修改 `docker-compose.yml` 添加 HTTPS 支持：

```yaml
frontend:
  # ...
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - /etc/letsencrypt:/etc/letsencrypt:ro
```

### 3. 防火墙配置

```bash
# 仅开放必要端口
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

---

## 云平台部署

### 阿里云 ECS

1. 购买 ECS 实例（推荐 2核4G 起步）
2. 安装 Docker：
   ```bash
   curl -fsSL https://get.docker.com | sh
   systemctl enable docker
   systemctl start docker
   ```
3. 上传代码并运行 `docker-compose up -d`

### 腾讯云 CVM

步骤类似，可使用腾讯云容器镜像服务 (CCR) 托管镜像。

---

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查看端口占用
   netstat -tlnp | grep :80
   # 修改 docker-compose.yml 中的端口映射
   ```

2. **容器启动失败**
   ```bash
   # 查看详细日志
   docker-compose logs backend
   docker-compose logs frontend
   ```

3. **数据库连接失败**
   - 检查 DATABASE_URL 配置
   - 确保 data 目录权限正确

4. **前端无法访问后端 API**
   - 检查 nginx.conf 中的 proxy_pass 配置
   - 确认 backend 容器正常运行

---

## 架构说明

```
┌─────────────────────────────────────────────┐
│                 用户浏览器                    │
└─────────────────┬───────────────────────────┘
                  │ :80
┌─────────────────▼───────────────────────────┐
│           Frontend (Nginx)                   │
│  - 静态文件服务                               │
│  - /api 反向代理到 backend                   │
└─────────────────┬───────────────────────────┘
                  │ :8000
┌─────────────────▼───────────────────────────┐
│           Backend (FastAPI)                  │
│  - RESTful API                               │
│  - 雨流计数、寿命预测等计算                    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              SQLite / PostgreSQL             │
│              (Docker Volume)                 │
└─────────────────────────────────────────────┘
```
