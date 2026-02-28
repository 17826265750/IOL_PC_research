/**
 * 功率模块寿命分析软件 - 威布尔概率图组件
 * @author GSH
 */
import React, { useMemo } from 'react'
import { Box, Typography, Paper } from '@mui/material'
import ReactECharts from 'echarts-for-react'
import type { WeibullProbabilityPlotResult } from '@/types'

interface WeibullProbabilityPlotProps {
  data: WeibullProbabilityPlotResult
  shape: number
  scale: number
}

/**
 * Format number with limited decimal places
 */
function formatAxisLabel(value: number): string {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0)
  } else if (Math.abs(value) >= 10) {
    return value.toFixed(1)
  } else if (Math.abs(value) >= 1) {
    return value.toFixed(2)
  } else {
    return value.toFixed(3)
  }
}

export const WeibullProbabilityPlot: React.FC<WeibullProbabilityPlotProps> = ({
  data,
  shape,
  scale,
}) => {
  const option = useMemo(() => {
    // Scatter points - use actual time on X axis, Weibull Y on Y axis
    const scatterData = data.points.map((p) => [p.time, p.weibull_y])

    // Fitted line - convert from ln_time to actual time
    const lineData = data.fitted_line.x.map((lnT, i) => [
      Math.exp(lnT),  // Convert ln(time) back to time
      data.fitted_line.y[i]
    ])

    // Calculate axis ranges
    const times = data.points.map((p) => p.time)
    const yValues = data.points.map((p) => p.weibull_y)
    const xMin = Math.min(...times) * 0.9
    const xMax = Math.max(...times) * 1.1
    const yMin = Math.min(...yValues, ...data.fitted_line.y) - 0.3
    const yMax = Math.max(...yValues, ...data.fitted_line.y) + 0.3

    return {
      title: {
        text: '威布尔概率图',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 600,
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.seriesName === '数据点') {
            const point = data.points[params.dataIndex]
            const time = point.time.toFixed(1)
            const rank = (point.median_rank * 100).toFixed(1)
            return `循环/时间: ${time}<br/>中位秩: ${rank}%`
          } else {
            return params.seriesName
          }
        },
      },
      legend: {
        data: ['数据点', '拟合直线'],
        top: 30,
      },
      grid: {
        left: 80,
        right: 40,
        top: 80,
        bottom: 60,
      },
      xAxis: {
        type: 'log',
        name: '循环/时间',
        nameLocation: 'middle',
        nameGap: 30,
        min: xMin,
        max: xMax,
        axisLabel: {
          formatter: (value: number) => formatAxisLabel(value)
        },
        splitLine: {
          show: true,
          lineStyle: {
            type: 'dashed',
            opacity: 0.3,
          },
        },
      },
      yAxis: {
        type: 'value',
        name: 'ln(-ln(1-F))',
        nameLocation: 'middle',
        nameGap: 50,
        min: yMin,
        max: yMax,
        axisLabel: {
          formatter: (value: number) => value.toFixed(2)
        },
        splitLine: {
          show: true,
          lineStyle: {
            type: 'dashed',
            opacity: 0.3,
          },
        },
      },
      series: [
        {
          name: '数据点',
          type: 'scatter',
          data: scatterData,
          symbolSize: 10,
          itemStyle: {
            color: '#1976d2',
          },
        },
        {
          name: '拟合直线',
          type: 'line',
          data: lineData,
          symbol: 'none',
          lineStyle: {
            color: '#d32f2f',
            width: 2,
          },
        },
      ],
      toolbox: {
        feature: {
          saveAsImage: {
            title: '保存图片',
            pixelRatio: 2,
          },
          dataZoom: {
            title: {
              zoom: '缩放',
              back: '还原',
            },
          },
        },
        right: 20,
        top: 10,
      },
    }
  }, [data, shape, scale])

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        威布尔概率纸
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        数据点应近似分布在一条直线上，表明数据符合威布尔分布
      </Typography>
      <Box sx={{ height: 500 }}>
        <ReactECharts
          option={option}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'canvas' }}
        />
      </Box>
      <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="caption" color="text.secondary">
          β = {shape.toFixed(3)} (斜率 = 形状参数)
        </Typography>
        <Typography variant="caption" color="text.secondary">
          η = {scale.toFixed(1)} (在 F=63.2% 处的截距)
        </Typography>
      </Box>
    </Paper>
  )
}

export default WeibullProbabilityPlot
