import React from 'react'
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material'
import {
  BarChart,
  TrendingUp,
  TrendingDown,
  Info,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import type { SensitivityAnalysisResult } from '@/types'

interface SensitivityTornadoProps {
  data: SensitivityAnalysisResult | null
  loading?: boolean
}

const formatElasticity = (value: number): string => {
  const absValue = Math.abs(value)
  if (absValue >= 1) {
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}`
  }
  if (absValue >= 0.01) {
    return `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%`
  }
  return `${value > 0 ? '+' : ''}${(value * 1000).toFixed(1)}‰`
}

const getImpactLevel = (normalized: number): { label: string; color: string } => {
  if (normalized >= 0.7) return { label: '高 / High', color: '#d32f2f' }
  if (normalized >= 0.4) return { label: '中 / Medium', color: '#ed6c02' }
  return { label: '低 / Low', color: '#2e7d32' }
}

const PARAMETER_LABELS: Record<string, { zh: string; en: string; unit?: string }> = {
  deltaT: { zh: '温度摆动', en: 'ΔTj', unit: '°C' },
  Tjmax: { zh: '最高结温', en: 'Tjmax', unit: '°C' },
  theating: { zh: '加热时间', en: 'ton', unit: 's' },
  tcooling: { zh: '冷却时间', en: 'tcool', unit: 's' },
  I: { zh: '电流', en: 'I', unit: 'A' },
  V: { zh: '电压', en: 'V', unit: 'V' },
  D: { zh: '焊层厚度', en: 'D', unit: 'μm' },
  A: { zh: '模型常数', en: 'A' },
  n: { zh: '温度指数', en: 'n' },
  m: { zh: '时间指数', en: 'm' },
  Ea: { zh: '激活能', en: 'Ea', unit: 'eV' },
}

export const SensitivityTornado: React.FC<SensitivityTornadoProps> = ({ data, loading }) => {
  const getChartOption = () => {
    if (!data) return null

    // Sort sensitivities by absolute normalized value (descending)
    const sortedSensitivities = [...data.sensitivities].sort(
      (a, b) => Math.abs(b.normalized) - Math.abs(a.normalized)
    )

    return {
      title: {
        text: '参数敏感性分析 / Parameter Sensitivity Analysis',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const p = params[0]
          const param = sortedSensitivities[p.dataIndex]
          const label = PARAMETER_LABELS[param.parameter] || { zh: param.parameter, en: param.parameter }
          return `
            ${label.zh} (${label.en})<br/>
            弹性: ${formatElasticity(param.elasticity)}<br/>
            归一化: ${(param.normalized * 100).toFixed(1)}%
          `
        },
      },
      grid: {
        left: '3%',
        right: '8%',
        bottom: '3%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: '归一化敏感性 / Normalized Sensitivity',
        nameLocation: 'middle',
        nameGap: 30,
        max: 1,
        axisLabel: {
          formatter: '{value}',
        },
      },
      yAxis: {
        type: 'category',
        data: sortedSensitivities.map((s) => {
          const label = PARAMETER_LABELS[s.parameter]
          return label ? `${label.zh} (${label.en})` : s.parameter
        }),
        axisLabel: {
          fontSize: 11,
        },
      },
      series: [
        {
          type: 'bar',
          data: sortedSensitivities.map((s) => ({
            value: s.normalized,
            itemStyle: {
              color: Math.abs(s.normalized) >= 0.7
                ? '#ef5350'
                : Math.abs(s.normalized) >= 0.4
                  ? '#ffa726'
                  : '#66bb6a',
            },
          })),
          label: {
            show: true,
            position: 'right',
            formatter: (p: any) => `${(p.value * 100).toFixed(1)}%`,
          },
        },
      ],
    }
  }

  const getTornadoChartOption = () => {
    if (!data) return null

    // Sort tornado data by base value deviation
    const sortedTornado = [...data.tornadoData].sort((a, b) => {
      const aRange = Math.abs(a.max - a.min)
      const bRange = Math.abs(b.max - b.min)
      return bRange - aRange
    })

    return {
      title: {
        text: '龙卷风图 / Tornado Chart',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const idx = params[0].dataIndex
          const item = sortedTornado[idx]
          const label = PARAMETER_LABELS[item.parameter] || { zh: item.parameter, en: item.parameter }
          const range = Math.abs(item.max - item.min)
          const percentChange = (range / item.base) * 100
          return `
            ${label.zh} (${label.en})<br/>
            最小值: ${item.max.toFixed(0)} 周次<br/>
            基准值: ${item.base.toFixed(0)} 周次<br/>
            最大值: ${item.min.toFixed(0)} 周次<br/>
            变化范围: ${percentChange.toFixed(1)}%
          `
        },
      },
      grid: {
        left: '3%',
        right: '8%',
        bottom: '3%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: '预测寿命 / Predicted Lifetime (cycles)',
        nameLocation: 'middle',
        nameGap: 80,
        axisLabel: {
          formatter: (value: number) => {
            if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M'
            if (value >= 1000) return (value / 1000).toFixed(0) + 'k'
            return value.toFixed(0)
          },
        },
      },
      yAxis: {
        type: 'category',
        data: sortedTornado.map((t) => {
          const label = PARAMETER_LABELS[t.parameter]
          return label ? `${label.zh} (${label.en})` : t.parameter
        }),
        axisLabel: {
          fontSize: 11,
        },
      },
      series: [
        {
          type: 'bar',
          stack: 'tornado',
          data: sortedTornado.map((t) => ({
            value: Math.abs(t.min - t.base),
            itemStyle: {
              color: t.min < t.base ? '#d32f2f' : '#1976d2',
            },
          })),
        },
        {
          type: 'bar',
          stack: 'tornado',
          data: sortedTornado.map((t) => ({
            value: Math.abs(t.max - t.base),
            itemStyle: {
              color: t.max > t.base ? '#1976d2' : '#d32f2f',
            },
          })),
        },
      ],
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <Typography>加载中... / Loading...</Typography>
      </Box>
    )
  }

  if (!data) {
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
        <BarChart sx={{ fontSize: 48, color: 'text.disabled', mb: 2 }} />
        <Typography variant="h6" color="text.secondary">
          等待敏感性分析结果 / Awaiting Sensitivity Analysis
        </Typography>
      </Paper>
    )
  }

  const sortedSensitivities = [...data.sensitivities].sort(
    (a, b) => Math.abs(b.normalized) - Math.abs(a.normalized)
  )

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="caption" color="text.secondary">
                分析参数数量
              </Typography>
              <Typography variant="h5" sx={{ mt: 0.5 }}>
                {data.sensitivities.length}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Parameters Analyzed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="caption" color="text.secondary">
                基准预测
              </Typography>
              <Typography variant="h5" sx={{ mt: 0.5 }}>
                {data.basePrediction >= 1000
                  ? `${(data.basePrediction / 1000).toFixed(0)}k`
                  : data.basePrediction.toFixed(0)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Base Prediction (cycles)
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="caption" color="text.secondary">
                最高敏感性
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                {sortedSensitivities[0]?.elasticity < 0 ? (
                  <TrendingDown color="error" fontSize="small" />
                ) : (
                  <TrendingUp color="success" fontSize="small" />
                )}
                <Typography variant="h5">
                  {sortedSensitivities[0]
                    ? formatElasticity(sortedSensitivities[0].elasticity)
                    : '-'
                  }
                </Typography>
              </Box>
              <Typography variant="caption" color="text.secondary">
                Highest Sensitivity
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="caption" color="text.secondary">
                关键参数
              </Typography>
              <Typography variant="h5" sx={{ mt: 0.5, fontSize: '1.1rem' }}>
                {PARAMETER_LABELS[sortedSensitivities[0]?.parameter]?.zh || '-'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Key Parameter
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <ReactECharts option={getChartOption()} style={{ height: 400 }} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <ReactECharts option={getTornadoChartOption()} style={{ height: 400 }} />
          </Paper>
        </Grid>
      </Grid>

      {/* Detailed Table */}
      <Paper sx={{ mt: 2 }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center' }}>
          <Info color="info" sx={{ mr: 1 }} />
          <Typography variant="subtitle2">详细数据 / Detailed Data</Typography>
        </Box>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>排名 / Rank</TableCell>
                <TableCell>参数 / Parameter</TableCell>
                <TableCell align="right">弹性 / Elasticity</TableCell>
                <TableCell align="right">归一化 / Normalized</TableCell>
                <TableCell align="right">影响级别 / Impact</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedSensitivities.map((s, index) => {
                const impact = getImpactLevel(Math.abs(s.normalized))
                const label = PARAMETER_LABELS[s.parameter]

                return (
                  <TableRow key={s.parameter} hover>
                    <TableCell>
                      <Chip
                        label={`#${index + 1}`}
                        size="small"
                        color={index === 0 ? 'primary' : 'default'}
                        variant={index === 0 ? 'filled' : 'outlined'}
                      />
                    </TableCell>
                    <TableCell>
                      <Box>
                        <Typography variant="body2">
                          {label?.zh || s.parameter}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {label?.en || s.parameter}
                          {label?.unit && ` (${label.unit})`}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                        {s.elasticity < 0 ? (
                          <TrendingDown fontSize="small" color="error" />
                        ) : (
                          <TrendingUp fontSize="small" color="success" />
                        )}
                        <Typography variant="body2" fontFamily="monospace">
                          {formatElasticity(s.elasticity)}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" fontFamily="monospace">
                        {(s.normalized * 100).toFixed(1)}%
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Chip
                        label={impact.label}
                        size="small"
                        sx={{
                          backgroundColor: impact.color + '20',
                          color: impact.color,
                          border: `1px solid ${impact.color}`,
                        }}
                      />
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Interpretation */}
      <Box sx={{ mt: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
        <Typography variant="subtitle2" gutterBottom>
          分析说明 / Interpretation
        </Typography>
        <Typography variant="body2" color="text.secondary">
          弹性系数表示参数变化1%对预测结果的影响百分比。负值表示参数增加会降低寿命预测值。
          归一化值表示各参数相对影响程度，总和为1。
        </Typography>
        <Typography variant="caption" color="text.disabled" display="block" sx={{ mt: 0.5 }}>
          Elasticity shows the % change in prediction for a 1% change in parameter. Negative values mean
          increasing the parameter reduces lifetime. Normalized values show relative impact (sum = 1).
        </Typography>
      </Box>
    </Box>
  )
}

export default SensitivityTornado
