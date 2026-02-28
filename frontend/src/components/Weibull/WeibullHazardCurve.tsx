/**
 * 功率模块寿命分析软件 - 威布尔失效率曲线组件
 * @author GSH
 */
import React, { useMemo } from 'react'
import { Box, Typography, Paper, Chip } from '@mui/material'
import ReactECharts from 'echarts-for-react'
import type { WeibullCurveResult } from '@/types'

interface WeibullHazardCurveProps {
  curveData: WeibullCurveResult
  shape: number
}

export const WeibullHazardCurve: React.FC<WeibullHazardCurveProps> = ({
  curveData,
  shape,
}) => {
  // Determine hazard rate behavior based on shape parameter
  const hazardBehavior = useMemo(() => {
    if (shape < 1) {
      return {
        label: '递减失效率 (DFR)',
        description: '失效率随时间递减，早期失效特征',
        color: 'warning' as const,
      }
    } else if (Math.abs(shape - 1) < 0.1) {
      return {
        label: '恒定失效率 (CFR)',
        description: '失效率基本恒定，随机失效特征',
        color: 'info' as const,
      }
    } else {
      return {
        label: '递增失效率 (IFR)',
        description: '失效率随时间增加，磨损/老化特征',
        color: 'error' as const,
      }
    }
  }, [shape])

  const option = useMemo(() => {
    // Hazard rate data
    const hazardData = curveData.times.map((t, i) => [
      t,
      curveData.hazard_rate[i],
    ])

    // Find max hazard rate for y-axis scaling
    const maxHazard = Math.max(...curveData.hazard_rate)
    const yMax = maxHazard * 1.1

    return {
      title: {
        text: '失效率曲线 h(t)',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 600,
        },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const time = params[0].data[0].toFixed(1)
          const hazard = params[0].data[1].toExponential(3)
          return `时间: ${time} 小时/循环<br/>失效率: ${hazard} /小时`
        },
      },
      legend: {
        data: ['失效率'],
        top: 30,
      },
      grid: {
        left: 80,
        right: 40,
        top: 80,
        bottom: 60,
      },
      xAxis: {
        type: 'value',
        name: '时间 (小时/循环)',
        nameLocation: 'middle',
        nameGap: 30,
        min: 0,
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
        name: '失效率 (1/小时)',
        nameLocation: 'middle',
        nameGap: 60,
        min: 0,
        max: yMax,
        axisLabel: {
          formatter: (value: number) => {
            if (value === 0) return '0'
            if (value < 0.001 || value >= 1000) {
              return value.toExponential(1)
            }
            return value.toFixed(4)
          },
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
          name: '失效率',
          type: 'line',
          data: hazardData,
          symbol: 'none',
          smooth: true,
          lineStyle: {
            color: '#d32f2f',
            width: 2,
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(211, 47, 47, 0.3)' },
                { offset: 1, color: 'rgba(211, 47, 47, 0.05)' },
              ],
            },
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
  }, [curveData])

  return (
    <Paper sx={{ p: 2 }}>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Typography variant="subtitle1" fontWeight={600}>
          失效率函数 h(t)
        </Typography>
        <Chip
          label={hazardBehavior.label}
          color={hazardBehavior.color}
          size="small"
        />
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {hazardBehavior.description}。h(t) 表示在时间 t 时刻的瞬时失效率。
      </Typography>
      <Box sx={{ height: 400 }}>
        <ReactECharts
          option={option}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'canvas' }}
        />
      </Box>
      <Box sx={{ mt: 2 }}>
        <Typography variant="caption" color="text.secondary">
          β = {shape.toFixed(3)}: {' '}
          {shape < 1
            ? 'β &lt; 1 表示早期失效，失效率递减'
            : shape > 1
            ? 'β &gt; 1 表示磨损失效，失效率递增'
            : 'β ≈ 1 表示随机失效，失效率恒定'}
        </Typography>
      </Box>
    </Paper>
  )
}

export default WeibullHazardCurve
