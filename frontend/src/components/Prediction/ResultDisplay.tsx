import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Alert,
  CircularProgress,
} from '@mui/material'
import {
  Download,
  Assessment,
  Timeline,
  ShowChart,
  Info,
  CheckCircle,
  Warning,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import type { PredictionResult } from '@/types'
import type { ExportFormat } from '@/types'

interface ResultDisplayProps {
  result: PredictionResult | null
  loading?: boolean
  error?: string | null
  onExport?: (format: ExportFormat) => void
}

const MODEL_NAMES: Record<string, { zh: string; en: string }> = {
  coffin_manson: { zh: 'Coffin-Manson', en: 'Coffin-Manson' },
  coffin_manson_arrhenius: { zh: 'Coffin-Manson-Arrhenius', en: 'Coffin-Manson-Arrhenius' },
  norris_landzberg: { zh: 'Norris-Landzberg', en: 'Norris-Landzberg' },
  cips2008: { zh: 'CIPS 2008 (Bayerer)', en: 'CIPS 2008 (Bayerer)' },
  lesit: { zh: 'LESIT', en: 'LESIT' },
}

const formatNumber = (num: number, decimals = 2): string => {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(decimals)}M`
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(decimals)}K`
  }
  return num.toFixed(decimals)
}

const formatHours = (hours: number): string => {
  if (hours >= 8760) {
    const years = hours / 8760
    return `${formatNumber(years, 1)} 年 (years)`
  }
  if (hours >= 24) {
    const days = hours / 24
    return `${formatNumber(days, 1)} 天 (days)`
  }
  return `${formatNumber(hours, 1)} 小时 (hours)`
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, loading, error, onExport }) => {
  const getLifetimeStatus = (cycles: number): { color: 'success' | 'warning' | 'error'; label: string } => {
    if (cycles >= 100000) {
      return { color: 'success', label: '优秀' }
    }
    if (cycles >= 20000) {
      return { color: 'warning', label: '良好' }
    }
    return { color: 'error', label: '需关注' }
  }

  const getContributionData = (result: PredictionResult) => {
    // Calculate factor contributions based on cycle results
    const contributions = result.cycleResults.map((cr) => ({
      name: `Cycle ${cr.index + 1}`,
      value: cr.damagePerCycle * 100,
      deltaT: cr.deltaT,
    }))

    return contributions
  }

  const getPieOption = (result: PredictionResult) => {
    const contributions = getContributionData(result)

    // Use scientific notation for small values
    const formatValue = (val: number) => {
      if (val < 0.01) {
        return val.toExponential(2)
      }
      return val.toFixed(4)
    }

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: { name: string; value: number; percent: number }) => {
          return `${params.name}<br/>值: ${formatValue(params.value)}%`
        },
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        textStyle: { fontSize: 12 },
      },
      series: [
        {
          name: '损伤贡献 / Damage Contribution',
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2,
          },
          label: {
            show: true,
            position: 'outside',
            formatter: (params: { name: string; percent: number }) => {
              return `${params.percent.toFixed(1)}%`
            },
            fontSize: 11,
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 14,
              fontWeight: 'bold',
            },
          },
          labelLine: {
            show: true,
          },
          data: contributions.map((c, idx) => ({
            value: Math.max(c.value, 0.001), // Minimum value for visibility
            name: `${c.name} (ΔT: ${c.deltaT.toFixed(1)}°C)`,
            itemStyle: {
              color: ['#1976d2', '#388e3c', '#f57c00', '#d32f2f', '#7b1fa2'][idx % 5]
            }
          })),
        },
      ],
    }
  }

  const getBarOption = (result: PredictionResult) => {
    const contributions = getContributionData(result)

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: (params: Array<{ name: string; value: number; data: { value: number; itemStyle: { color: string } } }>) => {
          if (params && params.length > 0) {
            const p = params[0]
            return `${p.name}<br/>损伤: ${p.value.toExponential(4)}`
          }
          return ''
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: contributions.map((c) => c.name),
        axisLabel: {
          fontSize: 10,
        },
      },
      yAxis: {
        type: 'value',
        name: '损伤/循环',
        axisLabel: {
          formatter: (value: number) => value.toExponential(1),
        },
      },
      series: [
        {
          name: '损伤贡献',
          type: 'bar',
          data: contributions.map((c) => ({
            value: c.value,
            itemStyle: {
              color: c.deltaT > 80 ? '#ef5350' : c.deltaT > 60 ? '#ffa726' : '#66bb6a',
            },
          })),
          label: {
            show: true,
            position: 'top',
            formatter: (params: { value: number }) => params.value.toExponential(2),
            fontSize: 9,
          },
        },
      ],
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 8 }}>
        <CircularProgress size={60} />
        <Typography sx={{ mt: 2 }}>正在计算预测结果... / Calculating predictions...</Typography>
      </Box>
    )
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        <Typography variant="body2">{error}</Typography>
      </Alert>
    )
  }

  if (!result) {
    return (
      <Paper
        sx={{
          p: 6,
          textAlign: 'center',
          border: 2,
          borderColor: 'divider',
          borderStyle: 'dashed',
        }}
      >
        <Assessment sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
        <Typography variant="h6" color="text.secondary" gutterBottom>
          等待预测 / Awaiting Prediction
        </Typography>
        <Typography variant="body2" color="text.secondary">
          请选择模型并设置参数后点击"计算"按钮
        </Typography>
        <Typography variant="caption" color="text.disabled">
          Select a model and set parameters, then click "Calculate"
        </Typography>
      </Paper>
    )
  }

  const status = getLifetimeStatus(result.predictedCycles)
  const modelName = MODEL_NAMES[result.modelType] || { zh: result.modelType, en: result.modelType }

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Timeline color="primary" />
          <Typography variant="h6">预测结果 / Prediction Results</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {(['csv', 'json', 'xlsx'] as ExportFormat[]).map((format) => (
            <Button
              key={format}
              size="small"
              variant="outlined"
              startIcon={<Download />}
              onClick={() => onExport?.(format)}
              disabled={!onExport}
            >
              {format.toUpperCase()}
            </Button>
          ))}
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        {/* Main Result */}
        <Grid item xs={12} md={6}>
          <Card
            sx={{
              height: '100%',
              border: 2,
              borderColor: `${status.color}.main`,
              background: `linear-gradient(135deg, ${status.color}.main 0%, ${status.color}.dark 100%)`,
              color: 'white',
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ShowChart sx={{ mr: 1 }} />
                <Typography variant="h4" fontWeight="bold">
                  {formatNumber(result.predictedCycles, 0)}
                </Typography>
              </Box>
              <Typography variant="body1" sx={{ opacity: 0.9 }}>
                预测循环次数 / Predicted Cycles
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip
                  icon={status.color === 'success' ? <CheckCircle /> : <Warning />}
                  label={status.label}
                  color={status.color === 'success' ? 'success' : 'warning'}
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.2)',
                    color: 'white',
                    '& .MuiChip-icon': { color: 'white' },
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Lifetime Hours */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="caption" color="text.secondary">
                预计寿命 / Estimated Lifetime
              </Typography>
              <Typography variant="h5" sx={{ mt: 1, fontWeight: 600, color: 'primary.main' }}>
                {formatHours(result.lifetimeHours)}
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                {result.lifetimeHours.toFixed(1)} h
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Model Info */}
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="caption" color="text.secondary">
                使用模型 / Model Used
              </Typography>
              <Box sx={{ mt: 1 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  {modelName.zh}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {modelName.en}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Confidence Interval */}
      {result.confidenceLower !== undefined && result.confidenceUpper !== undefined && (
        <Paper sx={{ p: 2, mb: 2, border: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Info color="info" sx={{ mr: 1, fontSize: 20 }} />
            <Typography variant="subtitle2">置信区间 / Confidence Interval (95%)</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              下限 / Lower: {formatNumber(result.confidenceLower, 0)}
            </Typography>
            <Box sx={{ flex: 1, height: 8, bgcolor: 'divider', borderRadius: 1, position: 'relative' }}>
              <Box
                sx={{
                  position: 'absolute',
                  left: `${((result.predictedCycles - result.confidenceLower) / result.predictedCycles) * 50}%`,
                  right: `${((result.confidenceUpper - result.predictedCycles) / result.confidenceUpper) * 50}%`,
                  height: '100%',
                  bgcolor: 'primary.main',
                  borderRadius: 1,
                }}
              />
              <Box
                sx={{
                  position: 'absolute',
                  left: '50%',
                  top: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 2,
                  height: 16,
                  bgcolor: 'primary.dark',
                }}
              />
            </Box>
            <Typography variant="body2" color="text.secondary">
              上限 / Upper: {formatNumber(result.confidenceUpper, 0)}
            </Typography>
          </Box>
        </Paper>
      )}

      {/* Charts */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                损伤分布 / Damage Distribution
              </Typography>
              <ReactECharts option={getPieOption(result)} style={{ height: 300 }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                各循环损伤 / Cycle Damage
              </Typography>
              <ReactECharts option={getBarOption(result)} style={{ height: 300 }} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Results Table */}
      <Paper sx={{ mb: 2 }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="subtitle2">详细结果 / Detailed Results</Typography>
        </Box>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>循环 / Cycle</TableCell>
                <TableCell>ΔT (°C)</TableCell>
                <TableCell align="right">预测寿命 / Nf</TableCell>
                <TableCell align="right">损伤/循环 / Damage/Cycle</TableCell>
                <TableCell align="right">累计损伤 / Cumulative</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {result.cycleResults.map((cycle, index) => {
                const cumulative = result.cycleResults
                  .slice(0, index + 1)
                  .reduce((sum, c) => sum + c.damagePerCycle, 0)

                return (
                  <TableRow key={cycle.index} hover>
                    <TableCell>{cycle.index + 1}</TableCell>
                    <TableCell>{cycle.deltaT.toFixed(2)}</TableCell>
                    <TableCell align="right">{formatNumber(cycle.cyclesToFailure, 0)}</TableCell>
                    <TableCell align="right">{cycle.damagePerCycle.toExponential(2)}</TableCell>
                    <TableCell align="right">{cumulative.toExponential(2)}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Timestamp */}
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          计算时间 / Calculation Time: {new Date(result.timestamp).toLocaleString('zh-CN')}
        </Typography>
      </Box>
    </Box>
  )
}

export default ResultDisplay
