/**
 * 功率模块寿命分析软件 - 威布尔可靠度曲线组件
 * @author GSH
 */
import React, { useMemo } from 'react'
import { Box, Typography, Paper } from '@mui/material'
import ReactECharts from 'echarts-for-react'
import type { WeibullCurveResult, WeibullFitResult } from '@/types'

interface WeibullReliabilityCurveProps {
  curveData: WeibullCurveResult
  fitResult: WeibullFitResult
  showConfidenceInterval?: boolean
}

export const WeibullReliabilityCurve: React.FC<WeibullReliabilityCurveProps> = ({
  curveData,
  fitResult,
  showConfidenceInterval = true,
}) => {
  const option = useMemo(() => {
    // Reliability data (convert to percentage)
    const reliabilityData = curveData.times.map((t, i) => [
      t,
      curveData.reliability[i] * 100,
    ])

    // B-life markers for markLine
    const bLifeMarkLineData: any[] = []
    const bLifeMarkPointData: any[] = []

    const maxTime = Math.max(...curveData.times)

    // B10 line
    if (fitResult.b10 <= maxTime) {
      bLifeMarkLineData.push({
        name: 'B10',
        xAxis: fitResult.b10,
        label: { formatter: 'B10', position: 'end' },
        lineStyle: { color: '#ff9800', type: 'dashed' },
      })
      bLifeMarkPointData.push({
        name: 'B10',
        coord: [fitResult.b10, 90],
        value: 'B10',
        itemStyle: { color: '#ff9800' },
      })
    }

    // B50 line
    if (fitResult.b50 <= maxTime) {
      bLifeMarkLineData.push({
        name: 'B50',
        xAxis: fitResult.b50,
        label: { formatter: 'B50', position: 'end' },
        lineStyle: { color: '#ff9800', type: 'dashed' },
      })
      bLifeMarkPointData.push({
        name: 'B50',
        coord: [fitResult.b50, 50],
        value: 'B50',
        itemStyle: { color: '#ff9800' },
      })
    }

    return {
      title: {
        text: '可靠度曲线 R(t)',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 600,
        },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          if (params && params[0]) {
            const time = params[0].data[0].toFixed(1)
            const reliability = params[0].data[1].toFixed(2)
            return `循环/时间: ${time}<br/>可靠度: ${reliability}%`
          }
          return ''
        },
      },
      legend: {
        data: ['可靠度'],
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
        name: '循环/时间',
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
        name: '可靠度 (%)',
        nameLocation: 'middle',
        nameGap: 50,
        min: 0,
        max: 100,
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
          name: '可靠度',
          type: 'line',
          data: reliabilityData,
          symbol: 'none',
          smooth: true,
          lineStyle: {
            color: '#1976d2',
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
                { offset: 0, color: 'rgba(25, 118, 210, 0.3)' },
                { offset: 1, color: 'rgba(25, 118, 210, 0.05)' },
              ],
            },
          },
          markLine: bLifeMarkLineData.length > 0 ? {
            symbol: 'none',
            data: bLifeMarkLineData,
          } : undefined,
          markPoint: bLifeMarkPointData.length > 0 ? {
            data: bLifeMarkPointData,
            symbol: 'circle',
            symbolSize: 10,
          } : undefined,
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
  }, [curveData, fitResult, showConfidenceInterval])

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        可靠度函数 R(t)
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        R(t) 表示设备在时间 t 之前正常工作的概率
      </Typography>
      <Box sx={{ height: 400 }}>
        <ReactECharts
          option={option}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'canvas' }}
        />
      </Box>
    </Paper>
  )
}

export default WeibullReliabilityCurve
