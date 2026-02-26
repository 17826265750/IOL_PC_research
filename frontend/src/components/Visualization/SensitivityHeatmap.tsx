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
  Card,
  CardContent,
  Grid,
  Slider,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material'
import { Info } from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import type { LifetimeModelType, LifetimeModelParams } from '@/types'

interface HeatmapDataPoint {
  xValue: number
  yValue: number
  lifetime: number
}

interface SensitivityHeatmapProps {
  modelType: LifetimeModelType
  params: LifetimeModelParams
  baseParams: Record<string, number>
}

// Simplified lifetime calculation for visualization
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

const generateHeatmapData = (
  modelType: LifetimeModelType,
  params: Record<string, number>,
  xParam: string,
  yParam: string,
  xRange: { min: number; max: number; steps: number },
  yRange: { min: number; max: number; steps: number }
): { data: HeatmapDataPoint[]; xValues: number[]; yValues: number[]; minLifetime: number; maxLifetime: number } => {
  const data: HeatmapDataPoint[] = []
  const xValues: number[] = []
  const yValues: number[] = []
  let minLifetime = Infinity
  let maxLifetime = -Infinity

  const allParams = { ...params }
  const xStep = (xRange.max - xRange.min) / xRange.steps
  const yStep = (yRange.max - yRange.min) / yRange.steps

  for (let i = 0; i <= yRange.steps; i++) {
    const yValue = yRange.min + yStep * i
    yValues.push(yValue)

    for (let j = 0; j <= xRange.steps; j++) {
      const xValue = xRange.min + xStep * j
      if (i === 0) xValues.push(xValue)

      let deltaT = allParams.deltaT || 80
      let Tjmax = allParams.Tjmax || 125
      let theating = allParams.theating || 60

      // Apply parameter variations
      if (xParam === 'deltaT') deltaT = xValue
      else if (xParam === 'Tjmax') Tjmax = xValue
      else if (xParam === 'theating') theating = xValue

      if (yParam === 'deltaT') deltaT = yValue
      else if (yParam === 'Tjmax') Tjmax = yValue
      else if (yParam === 'theating') theating = yValue

      const lifetime = calculateLifetime(modelType, allParams, deltaT, Tjmax, theating)
      minLifetime = Math.min(minLifetime, lifetime)
      maxLifetime = Math.max(maxLifetime, lifetime)

      data.push({ xValue, yValue, lifetime })
    }
  }

  return { data, xValues, yValues, minLifetime, maxLifetime }
}

const PARAMETER_OPTIONS = [
  { value: 'deltaT', label: '温度摆动 ΔTj', labelEn: 'ΔTj', unit: '°C', defaultMin: 30, defaultMax: 120 },
  { value: 'Tjmax', label: '最高结温 Tjmax', labelEn: 'Tjmax', unit: '°C', defaultMin: 80, defaultMax: 180 },
  { value: 'theating', label: '加热时间 ton', labelEn: 'ton', unit: 's', defaultMin: 10, defaultMax: 300 },
]

const COLOR_SCALES = [
  { value: 'viridis', label: 'Viridis' },
  { value: 'plasma', label: 'Plasma' },
  { value: 'inferno', label: 'Inferno' },
  { value: 'coolwarm', label: 'CoolWarm' },
]

const COLOR_SCALE_PALETTES: Record<string, string[]> = {
  viridis: ['#440154', '#482878', '#3e4a89', '#31688e', '#26838f', '#1f9d8a', '#35b779', '#6dcd59', '#b4de2c', '#fde725'],
  plasma: ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
  inferno: ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],
  coolwarm: ['#3b4cc0', '#6d83f3', '#a5b1ff', '#dcd6ff', '#f7f7f7', '#ffded8', '#f4a09b', '#d9585e', '#9c1734', '#4a031c'],
}

export const SensitivityHeatmap: React.FC<SensitivityHeatmapProps> = ({
  modelType,
  params,
  baseParams,
}) => {
  const [xParam, setXParam] = React.useState('deltaT')
  const [yParam, setYParam] = React.useState('Tjmax')
  const [colorScale, setColorScale] = React.useState('viridis')
  const [xRange, setXRange] = React.useState({ min: 30, max: 120 })
  const [yRange, setYRange] = React.useState({ min: 80, max: 180 })
  const [resolution, setResolution] = React.useState(25)

  const allParams = useMemo(() => {
    return {
      deltaT: (baseParams.Tmax || 125) - (baseParams.Tmin || 40),
      Tjmax: baseParams.Tjmax || baseParams.Tmax || 125,
      theating: baseParams.theating || 60,
      ...baseParams,
      ...(params as unknown as Record<string, number>),
    }
  }, [params, baseParams])

  const heatmapData = useMemo(() => {
    return generateHeatmapData(
      modelType,
      allParams,
      xParam,
      yParam,
      { min: xRange.min, max: xRange.max, steps: resolution },
      { min: yRange.min, max: yRange.max, steps: resolution }
    )
  }, [modelType, allParams, xParam, yParam, xRange, yRange, resolution])

  const getChartOption = () => {
    const palette = COLOR_SCALE_PALETTES[colorScale]

    return {
      title: {
        text: '双参数敏感性热力图 / Two-Parameter Sensitivity Heatmap',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const xLabel = PARAMETER_OPTIONS.find((p) => p.value === xParam)
          const yLabel = PARAMETER_OPTIONS.find((p) => p.value === yParam)
          const value = Array.isArray(params.value) ? params.value : [params.value]
          return `
            ${xLabel?.label}: ${value[0]?.toFixed(1) || 0}${xLabel?.unit}<br/>
            ${yLabel?.label}: ${value[1]?.toFixed(1) || 0}${yLabel?.unit}<br/>
            预测寿命: ${Math.pow(10, value[2] || 0).toExponential(2)} 周次
          `
        },
      },
      grid: {
        height: '70%',
        top: '15%',
      },
      xAxis: {
        type: 'category',
        data: heatmapData.xValues.map((v) => v.toFixed(1)),
        name: PARAMETER_OPTIONS.find((p) => p.value === xParam)?.labelEn || xParam,
        nameLocation: 'middle',
        nameGap: 30,
        axisLabel: {
          interval: 'auto',
          rotate: 0,
          fontSize: 10,
        },
        splitArea: {
          show: true,
        },
      },
      yAxis: {
        type: 'category',
        data: heatmapData.yValues.map((v) => v.toFixed(1)),
        name: PARAMETER_OPTIONS.find((p) => p.value === yParam)?.labelEn || yParam,
        nameLocation: 'middle',
        nameGap: 40,
        axisLabel: {
          fontSize: 10,
        },
        splitArea: {
          show: true,
        },
      },
      visualMap: {
        min: Math.log10(heatmapData.minLifetime),
        max: Math.log10(heatmapData.maxLifetime),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: palette,
        },
        text: ['高 / High', '低 / Low'],
        textStyle: {
          fontSize: 10,
        },
        formatter: (value: number) => {
          return `${Math.pow(10, value).toExponential(1)}`
        },
      },
      series: [
        {
          name: '预测寿命',
          type: 'heatmap',
          data: heatmapData.data.map((d) => [
            d.xValue.toFixed(1),
            d.yValue.toFixed(1),
            Math.log10(d.lifetime),
          ]),
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
    }
  }

  const handleXParamChange = (event: SelectChangeEvent<string>) => {
    const newParam = event.target.value
    setXParam(newParam)
    const option = PARAMETER_OPTIONS.find((p) => p.value === newParam)
    if (option) {
      setXRange({ min: option.defaultMin, max: option.defaultMax })
    }
  }

  const handleYParamChange = (event: SelectChangeEvent<string>) => {
    const newParam = event.target.value
    setYParam(newParam)
    const option = PARAMETER_OPTIONS.find((p) => p.value === newParam)
    if (option) {
      setYRange({ min: option.defaultMin, max: option.defaultMax })
    }
  }

  return (
    <Paper sx={{ p: 2 }}>
      {/* Controls */}
      <Box sx={{ mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>X轴参数 / X Parameter</InputLabel>
              <Select value={xParam} label="X轴参数 / X Parameter" onChange={handleXParamChange}>
                {PARAMETER_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label} ({option.labelEn})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Y轴参数 / Y Parameter</InputLabel>
              <Select value={yParam} label="Y轴参数 / Y Parameter" onChange={handleYParamChange}>
                {PARAMETER_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label} ({option.labelEn})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>配色方案 / Color Scale</InputLabel>
              <Select
                value={colorScale}
                label="配色方案 / Color Scale"
                onChange={(e) => setColorScale(e.target.value)}
              >
                {COLOR_SCALES.map((scale) => (
                  <MenuItem key={scale.value} value={scale.value}>
                    {scale.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <ToggleButtonGroup
              value={resolution}
              exclusive
              onChange={(_, newValue) => newValue && setResolution(newValue)}
              size="small"
              fullWidth
            >
              <ToggleButton value={15}>低 / Low</ToggleButton>
              <ToggleButton value={25}>中 / Med</ToggleButton>
              <ToggleButton value={40}>高 / High</ToggleButton>
            </ToggleButtonGroup>
          </Grid>
        </Grid>

        {/* Range Sliders */}
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} sm={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                X轴范围 / X Range: {xRange.min} - {xRange.max}
                {PARAMETER_OPTIONS.find((p) => p.value === xParam)?.unit}
              </Typography>
              <Slider
                size="small"
                value={[xRange.min, xRange.max]}
                onChange={(_, newValue) => {
                  const values = Array.isArray(newValue) ? newValue : [newValue, newValue]
                  setXRange({ min: values[0] as number, max: values[1] as number })
                }}
                min={0}
                max={200}
                valueLabelDisplay="auto"
              />
            </Box>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Y轴范围 / Y Range: {yRange.min} - {yRange.max}
                {PARAMETER_OPTIONS.find((p) => p.value === yParam)?.unit}
              </Typography>
              <Slider
                size="small"
                value={[yRange.min, yRange.max]}
                onChange={(_, newValue) => {
                  const values = Array.isArray(newValue) ? newValue : [newValue, newValue]
                  setYRange({ min: values[0] as number, max: values[1] as number })
                }}
                min={0}
                max={250}
                valueLabelDisplay="auto"
              />
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Chart */}
      <ReactECharts option={getChartOption()} style={{ height: 500 }} />

      {/* Info Cards */}
      <Grid container spacing={2} sx={{ mt: 2 }}>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary">
                最低寿命 / Min
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {heatmapData.minLifetime.toExponential(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary">
                最高寿命 / Max
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {heatmapData.maxLifetime.toExponential(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary">
                寿命比 / Ratio
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {(heatmapData.maxLifetime / heatmapData.minLifetime).toFixed(1)}x
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={6} sm={3}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
              <Typography variant="caption" color="text.secondary">
                分辨率 / Resolution
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {resolution} × {resolution}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Interpretation */}
      <Box sx={{ mt: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1, display: 'flex', gap: 1 }}>
        <Info color="info" fontSize="small" />
        <Box sx={{ flex: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            分析说明 / Interpretation
          </Typography>
          <Typography variant="body2" color="text.secondary">
            热力图显示两个参数对预测寿命的综合影响。颜色越暖表示预测寿命越高，颜色越冷表示预测寿命越低。
            可拖动滑块调整参数范围，或切换不同的参数组合进行分析。
          </Typography>
          <Typography variant="caption" color="text.disabled" display="block" sx={{ mt: 0.5 }}>
            The heatmap shows the combined effect of two parameters on predicted lifetime. Warmer colors indicate
            higher lifetime, cooler colors indicate lower lifetime. Adjust ranges with sliders or switch parameter
            combinations.
          </Typography>
        </Box>
      </Box>
    </Paper>
  )
}

export default SensitivityHeatmap
