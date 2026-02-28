/**
 * 功率模块寿命分析软件 - 威布尔拟合结果展示组件
 * @author GSH
 */
import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  Paper,
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Remove as RemoveIcon,
} from '@mui/icons-material'
import type { WeibullFitResult } from '@/types'

interface WeibullFitResultProps {
  result: WeibullFitResult
}

/**
 * Interpret shape parameter (β) for failure pattern
 */
function interpretShape(shape: number): {
  label: string
  description: string
  icon: React.ReactNode
  color: 'success' | 'warning' | 'error' | 'info'
} {
  if (shape < 1) {
    return {
      label: '早期失效（婴儿期）',
      description: '失效率随时间递减，通常由制造缺陷或质量控制问题导致',
      icon: <TrendingDownIcon />,
      color: 'warning',
    }
  } else if (shape < 1.5) {
    return {
      label: '近似随机失效',
      description: '失效率基本恒定，失效主要由随机因素引起',
      icon: <RemoveIcon />,
      color: 'info',
    }
  } else if (shape >= 1.5 && shape < 3) {
    return {
      label: '早期磨损',
      description: '失效率缓慢增加，开始出现磨损迹象',
      icon: <TrendingUpIcon />,
      color: 'success',
    }
  } else {
    return {
      label: '磨损失效',
      description: '失效率快速增加，主要由磨损、疲劳或老化导致',
      icon: <TrendingUpIcon />,
      color: 'error',
    }
  }
}

/**
 * Format number with appropriate precision
 */
function formatNumber(value: number, decimals: number = 2): string {
  if (value >= 10000 || value < 0.01) {
    return value.toExponential(decimals)
  }
  return value.toFixed(decimals)
}

export const WeibullFitResultDisplay: React.FC<WeibullFitResultProps> = ({
  result,
}) => {
  const shapeInterpretation = interpretShape(result.shape)

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {/* Shape Parameter (β) */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                形状参数 (β)
              </Typography>
              <Typography variant="h4" fontWeight={700} color="primary">
                {formatNumber(result.shape, 3)}
              </Typography>
              {result.shape_ci && (
                <Typography variant="caption" color="text.secondary">
                  CI: [{formatNumber(result.shape_ci.lower, 3)},{' '}
                  {formatNumber(result.shape_ci.upper, 3)}]
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Scale Parameter (η) */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                尺度参数 (η)
              </Typography>
              <Typography variant="h4" fontWeight={700} color="primary">
                {formatNumber(result.scale, 1)}
              </Typography>
              {result.scale_ci && (
                <Typography variant="caption" color="text.secondary">
                  CI: [{formatNumber(result.scale_ci.lower, 1)},{' '}
                  {formatNumber(result.scale_ci.upper, 1)}]
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* MTTF */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                平均失效时间 (MTTF)
              </Typography>
              <Typography variant="h4" fontWeight={700} color="primary">
                {formatNumber(result.mttf, 1)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                小时/循环
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* R² */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography
                variant="subtitle2"
                color="text.secondary"
                gutterBottom
              >
                拟合优度 (R²)
              </Typography>
              <Typography
                variant="h4"
                fontWeight={700}
                color={result.r_squared >= 0.95 ? 'success.main' : 'warning.main'}
              >
                {formatNumber(result.r_squared, 4)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {result.r_squared >= 0.95 ? '拟合优秀' : result.r_squared >= 0.9 ? '拟合良好' : '拟合一般'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Shape Interpretation */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {shapeInterpretation.icon}
          <Typography variant="subtitle1" fontWeight={600}>
            失效模式分析
          </Typography>
          <Chip
            label={shapeInterpretation.label}
            color={shapeInterpretation.color}
            size="small"
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          {shapeInterpretation.description}
        </Typography>
        {result.shape >= 1 && result.shape < 4 && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            β = {result.shape.toFixed(2)} 表示威布尔分布接近正态分布（β ≈ 3.5），
            表明失效主要由材料老化或疲劳累积导致。
          </Typography>
        )}
      </Paper>

      {/* Additional Parameters */}
      {result.confidence_level && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            置信水平: {(result.confidence_level * 100).toFixed(0)}%
            {result.shape_std_error && (
              <> | 形状参数标准误: {formatNumber(result.shape_std_error, 4)}</>
            )}
            {result.scale_std_error && (
              <> | 尺度参数标准误: {formatNumber(result.scale_std_error, 2)}</>
            )}
          </Typography>
        </Box>
      )}
    </Box>
  )
}

export default WeibullFitResultDisplay
