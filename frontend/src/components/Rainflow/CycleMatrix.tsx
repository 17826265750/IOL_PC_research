import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Paper,
} from '@mui/material'
import {
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import { RainflowCycle } from '@/types'

interface Props {
  cycles: RainflowCycle[]
  onExport?: () => void
}

export const CycleMatrix: React.FC<Props> = ({ cycles, onExport }) => {
  // Create a matrix from the cycles
  const createCycleMatrix = () => {
    if (cycles.length === 0) return { xData: [], yData: [], data: [], maxCount: 0 }

    // Get unique ranges and means
    const ranges = [...new Set(cycles.map((c) => Math.round(c.range * 10) / 10))].sort(
      (a, b) => a - b
    )
    const means = [...new Set(cycles.map((c) => Math.round(c.mean * 10) / 10))].sort(
      (a, b) => a - b
    )

    // Create matrix data
    const data: number[][] = []
    let maxCount = 0

    cycles.forEach((cycle) => {
      const xIndex = ranges.indexOf(Math.round(cycle.range * 10) / 10)
      const yIndex = means.indexOf(Math.round(cycle.mean * 10) / 10)
      if (xIndex !== -1 && yIndex !== -1) {
        data.push([xIndex, yIndex, cycle.count])
        if (cycle.count > maxCount) {
          maxCount = cycle.count
        }
      }
    })

    return {
      xData: ranges.map((r) => r.toFixed(1)),
      yData: means.map((m) => m.toFixed(1)),
      data,
      maxCount,
    }
  }

  const matrix = createCycleMatrix()

  const getOption = () => ({
    title: {
      text: '雨流计数循环矩阵',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 600,
      },
    },
    tooltip: {
      position: 'top',
      formatter: (params: any) => {
        const range = matrix.xData[params.value[0]]
        const mean = matrix.yData[params.value[1]]
        const count = params.value[2]
        return `范围: ${range}°C<br/>均值: ${mean}°C<br/>循环次数: ${count}`
      },
    },
    grid: {
      height: '70%',
      top: '15%',
    },
    xAxis: {
      type: 'category',
      data: matrix.xData,
      name: '温度范围 (°C)',
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: {
        rotate: 45,
        fontSize: 10,
      },
    },
    yAxis: {
      type: 'category',
      data: matrix.yData,
      name: '平均温度 (°C)',
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: {
        fontSize: 10,
      },
    },
    visualMap: {
      min: 0,
      max: matrix.maxCount || 10,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '5%',
      inRange: {
        color: ['#f0f9ff', '#c6e48b', '#7fcdbb', '#41b6c4', '#2c7fb8', '#253494'],
      },
      text: ['高', '低'],
      textStyle: {
        color: '#333',
      },
    },
    series: [
      {
        type: 'heatmap',
        data: matrix.data,
        label: {
          show: false,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
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
      right: 20,
      top: 10,
    },
  })

  const handleExport = () => {
    if (onExport) {
      onExport()
      return
    }

    // Default CSV export
    const headers = ['范围 (°C)', '均值 (°C)', '循环次数', '类型']
    const rows = cycles.map((c) => [
      c.range.toFixed(2),
      c.mean.toFixed(2),
      c.count.toString(),
      c.type,
    ])

    let csv = headers.join(',') + '\n'
    rows.forEach((row) => {
      csv += row.join(',') + '\n'
    })

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'cycle_matrix.csv'
    link.click()
    URL.revokeObjectURL(link.href)
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
          <Typography variant="h6">循环矩阵热力图</Typography>
          <Button
            variant="outlined"
            size="small"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
          >
            导出CSV
          </Button>
        </Stack>

        <Paper sx={{ p: 2, mb: 2 }}>
          <ReactECharts option={getOption()} style={{ height: '400px' }} />
        </Paper>

        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              总循环数
            </Typography>
            <Typography variant="h5">
              {cycles.reduce((sum, c) => sum + c.count, 0).toLocaleString()}
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              最大范围
            </Typography>
            <Typography variant="h5">
              {Math.max(...cycles.map((c) => c.range)).toFixed(1)}°C
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              最小范围
            </Typography>
            <Typography variant="h5">
              {Math.min(...cycles.map((c) => c.range)).toFixed(1)}°C
            </Typography>
          </Paper>
          <Paper sx={{ p: 2, flex: 1, minWidth: 150 }}>
            <Typography variant="caption" color="text.secondary">
              唯一循环数
            </Typography>
            <Typography variant="h5">{cycles.length}</Typography>
          </Paper>
        </Stack>
      </CardContent>
    </Card>
  )
}

export default CycleMatrix
