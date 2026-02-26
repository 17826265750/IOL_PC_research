import React, { useMemo } from 'react'
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  ToggleButtonGroup,
  ToggleButton,
  Card,
  CardContent,
  Grid,
} from '@mui/material'
import {
  ShowChart,
  Timeline,
  Thermostat,
  BarChart,
  ZoomIn,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import type { LifetimeModelType, LifetimeModelParams } from '@/types'

interface CurveData {
  parameter: string
  values: number[]
  lifetimes: number[]
  unit: string
}

interface LifetimeCurveProps {
  modelType: LifetimeModelType
  params: LifetimeModelParams
  baseParams: Record<string, number>
  compareModels?: LifetimeModelType[]
  onParamChange?: (param: string, value: number) => void
}

const PARAMETER_RANGES: Record<string, { min: number; max: number; steps: number; unit: string }> = {
  deltaT: { min: 10, max: 150, steps: 20, unit: '°C' },
  Tjmax: { min: 50, max: 200, steps: 20, unit: '°C' },
  theating: { min: 1, max: 600, steps: 20, unit: 's' },
  I: { min: 0, max: 200, steps: 20, unit: 'A' },
  V: { min: 0, max: 2000, steps: 20, unit: 'V' },
  D: { min: 50, max: 300, steps: 20, unit: 'μm' },
}

const MODEL_COLORS = [
  '#1976d2',
  '#9c27b0',
  '#2e7d32',
  '#ed6c02',
  '#d32f2f',
  '#0288d1',
]

const MODEL_LABELS: Record<string, string> = {
  coffin_manson: 'Coffin-Manson',
  coffin_manson_arrhenius: 'Coffin-Manson-Arrhenius',
  norris_landzberg: 'Norris-Landzberg',
  cips2008: 'CIPS 2008',
  lesit: 'LESIT',
}

// Simplified calculation functions for visualization
const calculateLifetime = (
  modelType: LifetimeModelType,
  params: Record<string, number>,
  deltaT: number,
  Tjmax: number,
  theating: number
): number => {
  const TjmaxK = Tjmax + 273.15
  const kB = 8.617e-5

  switch (modelType) {
    case 'coffin_manson':
      return (params.A || 100000) * Math.pow(deltaT, -(params.n || 1.5)) * Math.pow(theating, -(params.m || 0.5))

    case 'coffin_manson_arrhenius':
      return (
        (params.A || 100000) *
        Math.pow(deltaT, -(params.n || 1.5)) *
        Math.pow(theating, -(params.m || 0.5)) *
        Math.exp((params.Ea || 0.8) / (kB * TjmaxK))
      )

    case 'norris_landzberg':
      const f = 1 / (2 * theating)
      return (
        (params.A || 100000) *
        Math.pow(deltaT, -(params.n || 2)) *
        Math.pow(f, -(params.k || 0.33)) *
        Math.exp((params.Ea || 0.8) / (kB * TjmaxK))
      )

    case 'cips2008':
      const beta2 = params.beta2 || 2185
      const beta3 = params.beta3 || 0.532
      const I = params.I || 50
      const V = params.V || 600
      const D = params.D || 100
      return (
        (params.A || 124600) *
        Math.pow(deltaT, params.n || -4.416) *
        Math.exp(beta2 / TjmaxK) *
        Math.pow(theating, beta3) *
        Math.pow(I, params.beta4 || -0.736) *
        Math.pow(V, params.beta5 || -0.254) *
        Math.pow(D, params.beta6 || 0)
      )

    case 'lesit':
      return (
        (params.A || 100000) *
        Math.pow(deltaT, params.n || 1.3) *
        Math.exp((params.Ea || 0.5) / (kB * TjmaxK)) *
        Math.pow(theating, params.m || 0.4)
      )

    default:
      return 100000
  }
}

const generateCurveData = (
  modelType: LifetimeModelType,
  params: Record<string, number>,
  parameter: string,
  range: { min: number; max: number; steps: number }
): CurveData => {
  const values: number[] = []
  const lifetimes: number[] = []

  const step = (range.max - range.min) / range.steps
  const rangeWithUnit = PARAMETER_RANGES[parameter]

  for (let i = 0; i <= range.steps; i++) {
    const value = range.min + step * i
    values.push(value)

    let deltaT = params.deltaT || 80
    let Tjmax = params.Tjmax || 125
    let theating = params.theating || 60

    switch (parameter) {
      case 'deltaT':
        deltaT = value
        break
      case 'Tjmax':
        Tjmax = value
        break
      case 'theating':
        theating = value
        break
    }

    lifetimes.push(calculateLifetime(modelType, params, deltaT, Tjmax, theating))
  }

  return {
    parameter,
    values,
    lifetimes,
    unit: rangeWithUnit?.unit || '',
  }
}

export const LifetimeCurve: React.FC<LifetimeCurveProps> = ({
  modelType,
  params,
  baseParams,
  compareModels = [],
}) => {
  const [selectedParam, setSelectedParam] = React.useState('deltaT')
  const [scaleType, setScaleType] = React.useState<'linear' | 'log'>('log')

  const handleParamChange = (event: SelectChangeEvent<string>) => {
    setSelectedParam(event.target.value)
  }

  const handleScaleChange = (_: React.MouseEvent<HTMLElement>, newScale: 'linear' | 'log') => {
    if (newScale) {
      setScaleType(newScale)
    }
  }

  const curveData = useMemo(() => {
    const allParams: Record<string, number> = {
      deltaT: (baseParams.Tmax || 125) - (baseParams.Tmin || 40),
      Tjmax: baseParams.Tjmax || baseParams.Tmax || 125,
      theating: baseParams.theating || 60,
      ...baseParams,
      ...(params as unknown as Record<string, number>),
    }

    const mainData = generateCurveData(modelType, allParams, selectedParam, PARAMETER_RANGES[selectedParam])

    const compareData = compareModels.map((compareModel) => ({
      model: compareModel,
      data: generateCurveData(compareModel, allParams, selectedParam, PARAMETER_RANGES[selectedParam]),
    }))

    return { mainData, compareData }
  }, [modelType, params, baseParams, selectedParam, compareModels])

  const formatLifetimeValue = (value: number, forLogAxis = false): string => {
    if (!Number.isFinite(value)) return '--'
    if (value === 0) return '0'

    const absValue = Math.abs(value)

    if (forLogAxis) {
      return value.toExponential(1)
    }

    if (absValue >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
    if (absValue >= 1_000) return `${(value / 1_000).toFixed(1)}k`
    if (absValue >= 1) return value.toFixed(0)
    if (absValue >= 0.01) return value.toFixed(2)
    return value.toExponential(1)
  }

  const getChartOption = () => {
    const isYAxisLog = scaleType === 'log'

    const paramLabels: Record<string, { zh: string; en: string }> = {
      deltaT: { zh: '温度摆动 ΔTj (°C)', en: 'Temperature Swing ΔTj (°C)' },
      Tjmax: { zh: '最高结温 Tjmax (°C)', en: 'Max Junction Temp Tjmax (°C)' },
      theating: { zh: '加热时间 ton (s)', en: 'Heating Time ton (s)' },
      I: { zh: '电流 I (A)', en: 'Current I (A)' },
      V: { zh: '电压 V (V)', en: 'Voltage V (V)' },
      D: { zh: '焊层厚度 D (μm)', en: 'Solder Thickness D (μm)' },
    }

    const series = [
      {
        name: MODEL_LABELS[modelType],
        type: 'line',
        data: curveData.mainData.lifetimes.map((lf, i) => [curveData.mainData.values[i], lf]),
        smooth: true,
        lineStyle: { width: 3 },
        itemStyle: { color: MODEL_COLORS[0] },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: MODEL_COLORS[0] + '80' },
              { offset: 1, color: MODEL_COLORS[0] + '10' },
            ],
          },
        },
      },
      ...curveData.compareData.map((compare, idx) => ({
        name: MODEL_LABELS[compare.model],
        type: 'line' as const,
        data: compare.data.lifetimes.map((lf, i) => [compare.data.values[i], lf]),
        smooth: true,
        lineStyle: {
          width: 2,
          type: 'dashed',
        },
        itemStyle: { color: MODEL_COLORS[(idx + 1) % MODEL_COLORS.length] },
      })),
    ]

    return {
      title: {
        text: paramLabels[selectedParam]?.zh || selectedParam,
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          let result = `${params[0].value[0].toFixed(2)} ${curveData.mainData.unit}<br/>`
          params.forEach((p: any) => {
            result += `${p.marker}${p.seriesName}: ${formatLifetimeValue(p.value[1], isYAxisLog)}<br/>`
          })
          return result
        },
      },
      legend: {
        bottom: 10,
        data: [MODEL_LABELS[modelType], ...curveData.compareData.map((c) => MODEL_LABELS[c.model])],
      },
      grid: {
        left: '8%',
        right: '5%',
        bottom: '15%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: curveData.mainData.unit,
        nameLocation: 'middle',
        nameGap: 30,
        scale: true,
        axisLabel: {
          formatter: (value: number) => {
            if (value >= 1000) {
              return (value / 1000).toFixed(0) + 'k'
            }
            return value.toFixed(0)
          },
        },
        splitLine: {
          show: true,
          lineStyle: { type: 'dashed', opacity: 0.3 },
        },
      },
      yAxis: {
        type: isYAxisLog ? 'log' : 'value',
        name: '循环次数 Nf / Cycles',
        nameLocation: 'middle',
        nameGap: 60,
        scale: true,
        logBase: 10,
        axisLabel: {
          formatter: (value: number) => {
            return formatLifetimeValue(value, isYAxisLog)
          },
        },
        splitLine: {
          show: true,
          lineStyle: { type: 'dashed', opacity: 0.3 },
        },
      },
      series,
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
        },
        {
          type: 'inside',
          yAxisIndex: 0,
          filterMode: 'none',
        },
      ],
    }
  }

  return (
    <Paper sx={{ p: 2 }}>
      {/* Controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>参数 / Parameter</InputLabel>
            <Select value={selectedParam} label="参数 / Parameter" onChange={handleParamChange}>
              <MenuItem value="deltaT">ΔTj - 温度摆动</MenuItem>
              <MenuItem value="Tjmax">Tjmax - 最高结温</MenuItem>
              <MenuItem value="theating">ton - 加热时间</MenuItem>
            </Select>
          </FormControl>

          <ToggleButtonGroup
            value={scaleType}
            exclusive
            onChange={handleScaleChange}
            size="small"
          >
            <ToggleButton value="linear">
              <Typography variant="caption">线性 / Linear</Typography>
            </ToggleButton>
            <ToggleButton value="log">
              <Typography variant="caption">对数 / Log</Typography>
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <ZoomIn fontSize="small" color="action" />
          <Typography variant="caption" color="text.secondary">
            可缩放 / Zoomable
          </Typography>
        </Box>
      </Box>

      {/* Chart */}
      <ReactECharts option={getChartOption()} style={{ height: 400 }} />

      {/* Info Cards */}
      <Grid container spacing={2} sx={{ mt: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Thermostat fontSize="small" color="primary" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    X轴 / X-Axis
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {selectedParam === 'deltaT' ? '温度摆动' : selectedParam === 'Tjmax' ? '最高结温' : '加热时间'}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Timeline fontSize="small" color="primary" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Y轴 / Y-Axis
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    预测循环次数 Nf
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <BarChart fontSize="small" color="primary" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    缩放 / Scale
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {scaleType === 'log' ? '对数坐标' : '线性坐标'}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ShowChart fontSize="small" color="primary" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    模型 / Model
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {MODEL_LABELS[modelType]}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  )
}

export default LifetimeCurve
