import React from 'react'
import {
  Box,
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Chip,
  Stack,
} from '@mui/material'
import {
  Timeline as PredictionIcon,
  Waves as RainflowIcon,
  ShowChart as DamageIcon,
  ArrowForward as ArrowForwardIcon,
  Science as FittingIcon,
} from '@mui/icons-material'
import { useNavigate } from 'react-router-dom'

interface FeatureCardProps {
  title: string
  description: string
  icon: React.ElementType
  path: string
  color: 'primary' | 'secondary' | 'success' | 'warning' | 'info' | 'error'
}

const FeatureCard: React.FC<FeatureCardProps> = ({
  title,
  description,
  icon: Icon,
  path,
  color,
}) => {
  const navigate = useNavigate()

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        transition: 'transform 0.2s, box-shadow 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 4,
        },
      }}
    >
      <CardContent sx={{ flexGrow: 1 }}>
        <Box
          sx={{
            width: 48,
            height: 48,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
            bgcolor: `${color}.main`,
            color: `${color}.contrastText`,
          }}
        >
          <Icon sx={{ fontSize: 28 }} />
        </Box>
        <Typography variant="h6" component="h2" gutterBottom fontWeight={600}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </CardContent>
      <CardActions>
        <Button
          size="small"
          endIcon={<ArrowForwardIcon />}
          onClick={() => navigate(path)}
          color={color}
        >
          打开
        </Button>
      </CardActions>
    </Card>
  )
}

export const Home: React.FC = () => {
  const features: FeatureCardProps[] = [
    {
      title: '参数拟合',
      description:
        '基于功率循环试验数据，自动拟合各寿命模型的参数，支持数据导入和结果保存。',
      icon: FittingIcon,
      path: '/fitting',
      color: 'secondary',
    },
    {
      title: '寿命预测',
      description:
        '使用5种常用寿命模型（Coffin-Manson、Coffin-Manson-Arrhenius、Norris-Landzberg、CIPS 2008、LESIT）进行功率器件寿命预测。',
      icon: PredictionIcon,
      path: '/prediction',
      color: 'primary',
    },
    {
      title: '雨流计数',
      description:
        '对温度或应力时间序列数据进行雨流循环计数，提取循环幅值、均值和循环次数。',
      icon: RainflowIcon,
      path: '/rainflow',
      color: 'success',
    },
    {
      title: '累计损伤与剩余寿命评估',
      description:
        '基于Miner线性累计损伤理论，计算多工况条件下的累计损伤度、剩余寿命和健康状态评估。',
      icon: DamageIcon,
      path: '/damage',
      color: 'warning',
    },
  ]

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" component="h1" gutterBottom fontWeight={600}>
          CIPS 2008 寿命预测软件
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          专业功率器件寿命预测分析平台，支持多种主流寿命模型和全面的可靠性评估。
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap">
          <Chip label="Coffin-Manson" size="small" variant="outlined" />
          <Chip label="Coffin-Manson-Arrhenius" size="small" variant="outlined" />
          <Chip label="Norris-Landzberg" size="small" variant="outlined" />
          <Chip label="CIPS 2008" size="small" variant="outlined" color="primary" />
          <Chip label="LESIT" size="small" variant="outlined" />
        </Stack>
      </Box>

      <Grid container spacing={3}>
        {features.map((feature) => (
          <Grid item xs={12} sm={6} md={4} key={feature.title}>
            <FeatureCard {...feature} />
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 6 }}>
        <Card sx={{ bgcolor: 'primary.main' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom color="primary.contrastText">
              快速开始
            </Typography>
            <Typography variant="body2" color="primary.contrastText" sx={{ opacity: 0.9 }}>
              1. 在"参数拟合"页面导入实验数据并拟合模型参数
              <br />
              2. 在"寿命预测"页面选择合适的寿命模型进行单条件寿命预测
              <br />
              3. 在"雨流计数"页面对功耗时间序列进行热仿真与雨流循环分析
              <br />
              4. 在“累计损伤与剩余寿命评估”页面定义任务剖面，计算Miner累计损伤与剩余寿命
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Container>
  )
}

export default Home
