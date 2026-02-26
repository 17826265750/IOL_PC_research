import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Paper,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material'
import {
  BarChart as BarChartIcon,
  ShowChart as ShowChartIcon,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import { RainflowCycle } from '@/types'

interface Props {
  cycles: RainflowCycle[]
  binCount?: number
}

type ChartType = 'bar' | 'cumulative' | 'combined'

export const RainflowHistogram: React.FC<Props> = ({ cycles, binCount = 20 }) => {
  const [chartType, setChartType] = React.useState<ChartType>('bar')

  // Create histogram bins
  const createHistogram = () => {
    if (cycles.length === 0) {
      return { bins: [], counts: [], cumulative: [] }
    }

    const ranges = cycles.map((c) => c.range)
    const minRange = Math.min(...ranges)
    const maxRange = Math.max(...ranges)
    const binWidth = (maxRange - minRange) / binCount

    // Initialize bins
    const bins: string[] = []
    const counts: number[] = []
    const cumulative: number[] = []

    for (let i = 0; i < binCount; i++) {
      const binStart = minRange + i * binWidth
      const binEnd = binStart + binWidth
      bins.push(`${binStart.toFixed(1)}-${binEnd.toFixed(1)}`)

      // Count cycles in this bin
      const binCountValue = cycles
        .filter((c) => c.range >= binStart && c.range < binEnd)
        .reduce((sum, c) => sum + c.count, 0)

      counts.push(binCountValue)
    }

    // Calculate cumulative
    let sum = 0
    counts.forEach((count) => {
      sum += count
      cumulative.push(sum)
    })

    return { bins, counts, cumulative }
  }

  const histogram = createHistogram()

  const getBarOption = () => ({
    title: {
      text: '循环范围分布直方图',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 600,
      },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: (params: any) => {
        const param = params[0]
        return `范围: ${param.name}°C<br/>循环次数: ${param.value}`
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '15%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: histogram.bins,
      name: '温度范围 (°C)',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: {
        rotate: 45,
        fontSize: 10,
        interval: 'auto',
      },
    },
    yAxis: {
      type: 'value',
      name: '循环次数',
      nameLocation: 'middle',
      nameGap: 50,
    },
    series: [
      {
        name: '循环次数',
        type: 'bar',
        data: histogram.counts,
        itemStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: '#4287f5' },
              { offset: 1, color: '#63b3ed' },
            ],
          },
        },
        emphasis: {
          itemStyle: {
            color: '#3182ce',
          },
        },
      },
    ],
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: 'none',
        },
        restore: {},
        saveAsImage: {},
      },
      right: 10,
      top: 10,
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100,
      },
      {
        start: 0,
        end: 100,
      },
    ],
  })

  const getCumulativeOption = () => ({
    title: {
      text: '累积循环分布曲线',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 600,
      },
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const param = params[0]
        return `范围: ${param.name}°C<br/>累积循环: ${param.value.toFixed(0)}`
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '15%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: histogram.bins,
      name: '温度范围 (°C)',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: {
        rotate: 45,
        fontSize: 10,
        interval: 'auto',
      },
      boundaryGap: false,
    },
    yAxis: {
      type: 'value',
      name: '累积循环次数',
      nameLocation: 'middle',
      nameGap: 50,
    },
    series: [
      {
        name: '累积循环',
        type: 'line',
        data: histogram.cumulative,
        smooth: true,
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(66, 135, 245, 0.3)' },
              { offset: 1, color: 'rgba(66, 135, 245, 0.05)' },
            ],
          },
        },
        lineStyle: {
          color: '#4287f5',
          width: 2,
        },
        itemStyle: {
          color: '#4287f5',
        },
      },
    ],
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: 'none',
        },
        restore: {},
        saveAsImage: {},
      },
      right: 10,
      top: 10,
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100,
      },
      {
        start: 0,
        end: 100,
      },
    ],
  })

  const getCombinedOption = () => ({
    title: {
      text: '循环分布与累积曲线',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 600,
      },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
      },
    },
    legend: {
      data: ['循环次数', '累积循环'],
      top: 30,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '20%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: histogram.bins,
      name: '温度范围 (°C)',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: {
        rotate: 45,
        fontSize: 10,
        interval: 'auto',
      },
    },
    yAxis: [
      {
        type: 'value',
        name: '循环次数',
        nameLocation: 'middle',
        nameGap: 50,
        position: 'left',
      },
      {
        type: 'value',
        name: '累积循环',
        nameLocation: 'middle',
        nameGap: 50,
        position: 'right',
      },
    ],
    series: [
      {
        name: '循环次数',
        type: 'bar',
        data: histogram.counts,
        itemStyle: {
          color: 'rgba(66, 135, 245, 0.6)',
        },
      },
      {
        name: '累积循环',
        type: 'line',
        yAxisIndex: 1,
        data: histogram.cumulative,
        smooth: true,
        lineStyle: {
          color: '#e53e3e',
          width: 2,
        },
        itemStyle: {
          color: '#e53e3e',
        },
      },
    ],
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: 'none',
        },
        restore: {},
        saveAsImage: {},
      },
      right: 10,
      top: 10,
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100,
      },
      {
        start: 0,
        end: 100,
      },
    ],
  })

  const getOption = () => {
    switch (chartType) {
      case 'cumulative':
        return getCumulativeOption()
      case 'bar':
      default:
        return getBarOption()
    }
  }

  if (cycles.length === 0) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography color="text.secondary">
              暂无循环数据，请先输入时间历程数据并进行分析
            </Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">循环分布直方图</Typography>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={(_, value) => value && setChartType(value)}
            size="small"
          >
            <ToggleButton value="bar">
              <BarChartIcon fontSize="small" />
              柱状图
            </ToggleButton>
            <ToggleButton value="cumulative">
              <ShowChartIcon fontSize="small" />
              累积曲线
            </ToggleButton>
            <ToggleButton value="combined">
              <BarChartIcon fontSize="small" />
              组合图
            </ToggleButton>
          </ToggleButtonGroup>
        </Stack>

        <Paper sx={{ p: 2, mb: 2 }}>
          <ReactECharts option={getOption()} style={{ height: '350px' }} />
        </Paper>

        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              总循环数
            </Typography>
            <Typography variant="h5">
              {histogram.counts.reduce((a, b) => a + b, 0).toLocaleString()}
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              最大循环频率
            </Typography>
            <Typography variant="h5">
              {Math.max(...histogram.counts).toLocaleString()}
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              主要范围区间
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {
                histogram.bins[
                  histogram.counts.indexOf(Math.max(...histogram.counts))
                ]
              }
              °C
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              分箱数量
            </Typography>
            <Typography variant="h5">{binCount}</Typography>
          </Paper>
        </Stack>
      </CardContent>
    </Card>
  )
}

export default RainflowHistogram
