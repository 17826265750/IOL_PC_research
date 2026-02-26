import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  Paper,
  LinearProgress,
} from '@mui/material'
import ReactECharts from 'echarts-for-react'
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material'

interface HealthData {
  healthIndex: number // 0-100
  remainingLifeHours?: number
  trend?: 'improving' | 'stable' | 'degrading'
  confidenceInterval?: [number, number]
}

interface Props {
  data: HealthData
  historicalData?: Array<{ timestamp: string; healthIndex: number }>
}

const getHealthColor = (index: number): string => {
  if (index >= 80) return '#4caf50' // green
  if (index >= 60) return '#8bc34a' // light green
  if (index >= 40) return '#ff9800' // orange
  if (index >= 20) return '#ff5722' // deep orange
  return '#f44336' // red
}

const getHealthLabel = (index: number): string => {
  if (index >= 90) return '优秀'
  if (index >= 75) return '良好'
  if (index >= 50) return '一般'
  if (index >= 25) return '较差'
  return '危险'
}

export const HealthIndicator: React.FC<Props> = ({ data, historicalData = [] }) => {
  const { healthIndex, remainingLifeHours, trend = 'stable', confidenceInterval } = data

  // Gauge option
  const getGaugeOption = () => ({
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        min: 0,
        max: 100,
        splitNumber: 10,
        axisLine: {
          lineStyle: {
            width: 20,
            color: [
              [0.2, '#f44336'],
              [0.4, '#ff5722'],
              [0.6, '#ff9800'],
              [0.8, '#8bc34a'],
              [1, '#4caf50'],
            ],
          },
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,40.1H25.3L37.7,0.7H12.8z',
          length: '12%',
          width: 20,
          offsetCenter: [0, '-60%'],
          itemStyle: {
            color: 'auto',
          },
        },
        axisTick: {
          length: 12,
          lineStyle: {
            color: 'auto',
            width: 2,
          },
        },
        splitLine: {
          length: 20,
          lineStyle: {
            color: 'auto',
            width: 5,
          },
        },
        axisLabel: {
          color: '#464646',
          fontSize: 14,
          distance: -50,
          formatter: (value: number) => {
            if (value === 0) return '危险'
            if (value === 25) return '较差'
            if (value === 50) return '一般'
            if (value === 75) return '良好'
            if (value === 100) return '优秀'
            return ''
          },
        },
        title: {
          offsetCenter: [0, '-20%'],
          fontSize: 20,
        },
        detail: {
          fontSize: 40,
          offsetCenter: [0, '0%'],
          valueAnimation: true,
          formatter: (value: number) => `${value.toFixed(0)}%`,
          color: 'auto',
        },
        data: [
          {
            value: healthIndex,
            name: '健康指数',
          },
        ],
      },
    ],
  })

  // Trend chart option
  const getTrendOption = () => {
    if (historicalData.length === 0) return null

    const dates = historicalData.map((d) => new Date(d.timestamp).toLocaleDateString())
    const values = historicalData.map((d) => d.healthIndex)
    const upperBound = confidenceInterval
      ? values.map((v) => v + (confidenceInterval[1] - confidenceInterval[0]) / 2)
      : []
    const lowerBound = confidenceInterval
      ? values.map((v) => v - (confidenceInterval[1] - confidenceInterval[0]) / 2)
      : []

    return {
      title: {
        text: '健康指数趋势',
        left: 'center',
        textStyle: {
          fontSize: 14,
        },
      },
      tooltip: {
        trigger: 'axis',
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '20%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          fontSize: 10,
        },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        name: '健康指数 (%)',
      },
      series: [
        lowerBound.length > 0 && {
          name: '置信区间',
          type: 'line',
          data: lowerBound,
          lineStyle: {
            opacity: 0,
          },
          stack: 'confidence',
          symbol: 'none',
        },
        upperBound.length > 0 && {
          name: '置信区间',
          type: 'line',
          data: upperBound,
          lineStyle: {
            opacity: 0,
          },
          areaStyle: {
            color: 'rgba(66, 135, 245, 0.1)',
          },
          stack: 'confidence',
          symbol: 'none',
        },
        {
          name: '健康指数',
          type: 'line',
          data: values,
          smooth: true,
          lineStyle: {
            color: getHealthColor(healthIndex),
            width: 2,
          },
          itemStyle: {
            color: getHealthColor(healthIndex),
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: `${getHealthColor(healthIndex)}40` },
                { offset: 1, color: `${getHealthColor(healthIndex)}05` },
              ],
            },
          },
        },
      ].filter(Boolean),
    }
  }

  const getTrendIcon = () => {
    switch (trend) {
      case 'improving':
        return <TrendingUp sx={{ color: 'success.main' }} />
      case 'degrading':
        return <TrendingDown sx={{ color: 'error.main' }} />
      default:
        return <Remove sx={{ color: 'text.secondary' }} />
    }
  }

  const getTrendLabel = () => {
    switch (trend) {
      case 'improving':
        return '改善'
      case 'degrading':
        return '退化'
      default:
        return '稳定'
    }
  }

  const trendOption = getTrendOption()

  return (
    <Stack spacing={2}>
      {/* Main Health Gauge */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom textAlign="center">
            器件健康状态
          </Typography>

          <Box sx={{ height: 300, display: 'flex', justifyContent: 'center' }}>
            <ReactECharts option={getGaugeOption()} style={{ height: '100%', width: '100%' }} />
          </Box>

          <Box sx={{ textAlign: 'center', mt: 2 }}>
            <Stack direction="row" spacing={1} justifyContent="center" alignItems="center">
              {getTrendIcon()}
              <Typography
                variant="h4"
                sx={{
                  color: getHealthColor(healthIndex),
                  fontWeight: 600,
                }}
              >
                {getHealthLabel(healthIndex)}
              </Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary">
              当前状态: {getTrendLabel()}
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Statistics Cards */}
      <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
        <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
          <Typography variant="caption" color="text.secondary">
            健康指数
          </Typography>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <LinearProgress
              variant="determinate"
              value={healthIndex}
              sx={{
                flex: 1,
                height: 8,
                borderRadius: 4,
                backgroundColor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: getHealthColor(healthIndex),
                  borderRadius: 4,
                },
              }}
            />
            <Typography
              variant="h6"
              sx={{
                color: getHealthColor(healthIndex),
                fontWeight: 600,
                minWidth: 50,
                textAlign: 'right',
              }}
            >
              {healthIndex.toFixed(0)}%
            </Typography>
          </Box>
        </Paper>

        {remainingLifeHours !== undefined && (
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              预计剩余寿命
            </Typography>
            <Typography variant="h6" sx={{ color: 'info.main', fontWeight: 600 }}>
              {remainingLifeHours > 0
                ? `${remainingLifeHours.toLocaleString(undefined, { maximumFractionDigits: 0 })} h`
                : 'N/A'}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {remainingLifeHours > 0
                ? `约 ${(remainingLifeHours / 24 / 365).toFixed(1)} 年`
                : ''}
            </Typography>
          </Paper>
        )}

        <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
          <Typography variant="caption" color="text.secondary">
            状态趋势
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            {getTrendIcon()}
            <Typography variant="h6">{getTrendLabel()}</Typography>
          </Stack>
        </Paper>

        {confidenceInterval && (
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              预测区间 (95%)
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {confidenceInterval[0].toFixed(0)}% - {confidenceInterval[1].toFixed(0)}%
            </Typography>
          </Paper>
        )}
      </Stack>

      {/* Trend Chart */}
      {trendOption && (
        <Card>
          <CardContent>
            <ReactECharts option={trendOption} style={{ height: '250px' }} />
          </CardContent>
        </Card>
      )}

      {/* Recommendations based on health */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            维护建议
          </Typography>
          <Stack spacing={1}>
            {healthIndex >= 80 && (
              <Typography variant="body2" color="success.main">
                • 器件状态良好，按计划进行常规维护
              </Typography>
            )}
            {healthIndex >= 60 && healthIndex < 80 && (
              <>
                <Typography variant="body2" color="info.main">
                  • 器件状态正常，建议定期监测关键参数
                </Typography>
                <Typography variant="body2" color="info.main">
                  • 关注运行温度变化，避免过载运行
                </Typography>
              </>
            )}
            {healthIndex >= 40 && healthIndex < 60 && (
              <>
                <Typography variant="body2" color="warning.main">
                  • 器件状态一般，建议增加监测频率
                </Typography>
                <Typography variant="body2" color="warning.main">
                  • 考虑优化工作条件，减少温度循环
                </Typography>
                <Typography variant="body2" color="warning.main">
                  • 准备预防性维护计划
                </Typography>
              </>
            )}
            {healthIndex >= 20 && healthIndex < 40 && (
              <>
                <Typography variant="body2" color="error.main">
                  • 器件状态较差，应尽快安排检修
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 准备备件，计划更换
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 避免极端工况运行
                </Typography>
              </>
            )}
            {healthIndex < 20 && (
              <>
                <Typography variant="body2" color="error.main">
                  • 器件处于危险状态，建议立即更换
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 分析失效原因，防止再次发生
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 考虑系统级检修
                </Typography>
              </>
            )}
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  )
}

export default HealthIndicator
