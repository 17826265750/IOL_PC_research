/**
 * 功率模块寿命分析软件 - 雨流计数页面
 * @author GSH
 */
import React, { useMemo, useCallback } from 'react'
import {
  Box,
  Container,
  Stack,
  Typography,
  Alert,
  CircularProgress,
  Grid,
  Paper,
  Tabs,
  Tab,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material'
import ReactECharts from 'echarts-for-react'
import { TimeHistoryInput } from '@/components/Rainflow/TimeHistoryInput'
import { apiService } from '@/services/api'
import type { RainflowResult } from '@/types'
import { useRainflowStore } from '@/stores/useRainflowStore'
import type { InputMode, ZthMode } from '@/stores/useRainflowStore'

/* ------------------------------------------------------------------ */
/*  TabPanel                                                           */
/* ------------------------------------------------------------------ */
function TabPanel({
  children,
  value,
  index,
}: {
  children?: React.ReactNode
  value: number
  index: number
}) {
  return (
    <Box hidden={value !== index} sx={{ pt: 2 }}>
      {value === index && children}
    </Box>
  )
}

/* ------------------------------------------------------------------ */
/*  Model parameter config for each registered lifetime model          */
/* ------------------------------------------------------------------ */
interface ModelParamDef {
  key: string
  label: string
  defaultValue: string
  unit?: string
}

const MODEL_PARAMS_CONFIG: Record<
  string,
  { label: string; formula: string; params: ModelParamDef[] }
> = {
  'coffin-manson': {
    label: 'Coffin-Manson',
    formula: 'Nf = A × (ΔTj)^(−α)',
    params: [
      { key: 'A', label: 'A (系数)', defaultValue: '3.025e14' },
      { key: 'alpha', label: 'α (指数)', defaultValue: '5.039' },
    ],
  },
  'coffin-manson-arrhenius': {
    label: 'Coffin-Manson-Arrhenius',
    formula: 'Nf = A × (ΔTj)^(−α) × Ea/(kB·Tj_mean)',
    params: [
      { key: 'A', label: 'A (系数)', defaultValue: '1e6' },
      { key: 'alpha', label: 'α (指数)', defaultValue: '2.0' },
      { key: 'Ea', label: 'Ea (活化能)', defaultValue: '0.8', unit: 'eV' },
    ],
  },
  'norris-landzberg': {
    label: 'Norris-Landzberg',
    formula: 'Nf = A × (ΔTj)^(−α) × f^β × exp(Ea/(kB·Tj_max))',
    params: [
      { key: 'A', label: 'A (系数)', defaultValue: '1e6' },
      { key: 'alpha', label: 'α (指数)', defaultValue: '2.0' },
      { key: 'beta', label: 'β (频率指数)', defaultValue: '0.333' },
      { key: 'Ea', label: 'Ea (活化能)', defaultValue: '0.5', unit: 'eV' },
      { key: 'f', label: 'f (频率)', defaultValue: '0.01', unit: 'Hz' },
    ],
  },
  'cips-2008': {
    label: 'CIPS 2008 (Bayerer)',
    formula:
      'Nf = K × ΔTj^β₁ × exp(β₂/Tj_max) × t_on^β₃ × I^β₄ × V^β₅ × D^β₆',
    params: [
      { key: 'K', label: 'K (系数)', defaultValue: '9.3e14' },
      { key: 'beta1', label: 'β₁ (ΔTj)', defaultValue: '-4.416' },
      { key: 'beta2', label: 'β₂ (Tj_max)', defaultValue: '1285' },
      { key: 'beta3', label: 'β₃ (t_on)', defaultValue: '-0.463' },
      { key: 'beta4', label: 'β₄ (I)', defaultValue: '-0.716' },
      { key: 'beta5', label: 'β₅ (V)', defaultValue: '-0.761' },
      { key: 'beta6', label: 'β₆ (D)', defaultValue: '-0.5' },
      { key: 't_on', label: 't_on', defaultValue: '3', unit: 's' },
      { key: 'I', label: 'I (电流)', defaultValue: '100', unit: 'A' },
      { key: 'V', label: 'V (电压)', defaultValue: '600', unit: 'V' },
      { key: 'D', label: 'D (键合线直径)', defaultValue: '400', unit: 'μm' },
    ],
  },
  lesit: {
    label: 'LESIT',
    formula: 'Nf = A × (ΔTj)^α × exp(Q/(R·Tj_min))',
    params: [
      { key: 'A', label: 'A (系数)', defaultValue: '3.025e5' },
      { key: 'alpha', label: 'α (指数)', defaultValue: '-5.039' },
      { key: 'Q', label: 'Q (活化能)', defaultValue: '0.8', unit: 'eV' },
    ],
  },
}

const DEFAULT_ZTH_SELF = '0.05,0.001\n0.15,0.01\n0.30,0.1\n0.50,1.0'
const DEFAULT_ZTH_CROSS = '0.01,0.005\n0.03,0.05'

/* ================================================================== */
/*  Main Component                                                     */
/* ================================================================== */
export const RainflowCounting: React.FC = () => {
  /* ---------- Store (global state — survives navigation) ---------- */
  const s = useRainflowStore()
  const patch = s.patch

  /* ================================================================ */
  /*  Parse helpers                                                    */
  /* ================================================================ */
  const parseFoster = useCallback(
    (text: string) =>
      text
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => {
          const [r, t] = l.split(/[,\s]+/).map(Number)
          return { R: r, tau: t }
        })
        .filter((e) => e.R > 0 && e.tau > 0),
    [],
  )

  const parseNumbers = useCallback(
    (text: string): number[] =>
      text
        .split(/[,\n\s]+/)
        .map(Number)
        .filter((n) => !isNaN(n)),
    [],
  )

  /* ================================================================ */
  /*  Pipeline runner                                                  */
  /* ================================================================ */
  const commonPayload = useCallback(
    () => ({
      binCount: 20,
      nBand: parseInt(s.nBand) || 20,
      // yMin/yMax are determined by backend from thermal_summary
      // to ensure from_to_matrix uses current data's temperature range
      ignoreBelow: parseFloat(s.ignoreBelow) || 0,
      rearrange: s.rearrange,
    }),
    [s.nBand, s.ignoreBelow, s.rearrange],
  )

  const runPipeline = useCallback(
    async (payload: Parameters<typeof apiService.runRainflowPipeline>[0]) => {
      patch({ loading: true, error: null })
      try {
        const resp = await apiService.runRainflowPipeline(payload)
        if (resp.success && resp.data) {
          const data = resp.data as RainflowResult
          const updates: Record<string, unknown> = { result: data }

          // Store computed Tj series
          if (data.originalData?.length) {
            updates.tjSeries = data.originalData
          }
          // Store multi-source Tj series
          if (data.allJunctionTemperatures) {
            updates.allTjSeries = data.allJunctionTemperatures
          }

          // Auto-fill yMin/yMax from thermal summary (always update)
          if (data.thermalSummary) {
            updates.yMin = data.thermalSummary.tj_min.toFixed(1)
            updates.yMax = data.thermalSummary.tj_max.toFixed(1)
          }

          patch(updates as Partial<typeof s>)

          // Warn if no cycles detected
          if (data.totalCycles === 0 && data.originalData?.length > 0) {
            patch({
              error:
                '⚠ 未检测到温度循环——数据可能为常数或波动太小。请检查输入数据或热阻抗参数。',
            })
          }
        } else {
          patch({ error: resp.error || '分析失败' })
        }
      } catch (err) {
        patch({ error: err instanceof Error ? err.message : '请求失败' })
      } finally {
        patch({ loading: false })
      }
    },
    [patch, s.yMin, s.yMax],
  )

  /* ---- Tab 0 handlers ---- */
  const handlePowerSubmit = useCallback(
    async (data: number[]) => {
      const foster = parseFoster(s.fosterInput)
      const zthValues = parseNumbers(s.zthInput)

      const payload: Parameters<typeof apiService.runRainflowPipeline>[0] = {
        powerCurve: data,
        ambientTemperature: parseFloat(s.ambientTemp) || 25,
        dt: parseFloat(s.dt) || 1.0,
        ...commonPayload(),
      }
      if (s.zthMode === 'foster' && foster.length > 0) {
        payload.fosterParams = foster
      } else if (s.zthMode === 'sampled' && zthValues.length > 0) {
        payload.thermalImpedanceCurve = zthValues
        payload.responseType = 'step'
      }
      // Always use model-based damage
      if (s.lifetimeModel) {
        const mp: Record<string, number> = {}
        Object.entries(s.modelParamsInput).forEach(([k, v]) => {
          const n = parseFloat(v); if (!isNaN(n)) mp[k] = n
        })
        payload.lifetimeModel = s.lifetimeModel
        payload.modelParams = mp
        payload.safetyFactor = parseFloat(s.safetyFactor) || 1.0
      }
      await runPipeline(payload)
      patch({ tab: 1 })
    },
    [
      parseFoster, parseNumbers,
      s.fosterInput, s.zthInput,
      s.ambientTemp, s.dt, s.zthMode,
      s.lifetimeModel, s.modelParamsInput, s.safetyFactor,
      commonPayload, runPipeline, patch,
    ],
  )

  const handleTjSubmit = useCallback(
    async (data: number[]) => {
      patch({ tjSeries: data })
      const base: Parameters<typeof apiService.runRainflowPipeline>[0] = {
        junctionTemperature: data,
        ...commonPayload(),
      }
      if (s.lifetimeModel) {
        const mp: Record<string, number> = {}
        Object.entries(s.modelParamsInput).forEach(([k, v]) => {
          const n = parseFloat(v); if (!isNaN(n)) mp[k] = n
        })
        base.lifetimeModel = s.lifetimeModel
        base.modelParams = mp
        base.safetyFactor = parseFloat(s.safetyFactor) || 1.0
      }
      await runPipeline(base)
      patch({ tab: 1 })
    },
    [s.lifetimeModel, s.modelParamsInput, s.safetyFactor, commonPayload, runPipeline, patch],
  )

  const handleMultiSourceSubmit = useCallback(async () => {
    const n = s.sourceCount
    const powerCurves = s.sourcePowerData
      .slice(0, n)
      .map((text) => parseNumbers(text))
    if (powerCurves.some((pc) => pc.length === 0)) {
      patch({ error: '请为每个热源输入功耗数据' })
      return
    }
    const zthMatrix: Array<Array<Array<{ R: number; tau: number }>>> = []
    for (let i = 0; i < n; i++) {
      const row: Array<Array<{ R: number; tau: number }>> = []
      for (let j = 0; j < n; j++) {
        row.push(parseFoster(s.zthMatrixInput[i]?.[j] || ''))
      }
      zthMatrix.push(row)
    }
    const payload: Parameters<typeof apiService.runRainflowPipeline>[0] = {
      powerCurves,
      zthMatrix,
      sourceNames: s.sourceNames.slice(0, n),
      targetNode: s.targetNode,
      ambientTemperature: parseFloat(s.ambientTemp) || 25,
      dt: parseFloat(s.dt) || 1.0,
      ...commonPayload(),
    }
    if (s.lifetimeModel) {
      const mp: Record<string, number> = {}
      Object.entries(s.modelParamsInput).forEach(([k, v]) => {
        const num = parseFloat(v); if (!isNaN(num)) mp[k] = num
      })
      payload.lifetimeModel = s.lifetimeModel
      payload.modelParams = mp
      payload.safetyFactor = parseFloat(s.safetyFactor) || 1.0
    }
    await runPipeline(payload)
    patch({ tab: 1 })
  }, [
    s.sourceCount, s.sourcePowerData, s.sourceNames, s.zthMatrixInput,
    s.targetNode, s.ambientTemp, s.dt,
    s.lifetimeModel, s.modelParamsInput, s.safetyFactor,
    parseNumbers, parseFoster, commonPayload, runPipeline, patch,
  ])

  /* ---- Tab 1 handler ---- */
  const handleRecompute = useCallback(async () => {
    if (!s.tjSeries) {
      patch({ error: '请先在「功耗变温度」页输入数据' })
      return
    }
    const base: Parameters<typeof apiService.runRainflowPipeline>[0] = {
      junctionTemperature: s.tjSeries,
      ...commonPayload(),
    }
    if (s.lifetimeModel) {
      const mp: Record<string, number> = {}
      Object.entries(s.modelParamsInput).forEach(([k, v]) => {
        const n = parseFloat(v); if (!isNaN(n)) mp[k] = n
      })
      base.lifetimeModel = s.lifetimeModel
      base.modelParams = mp
      base.safetyFactor = parseFloat(s.safetyFactor) || 1.0
    }
    await runPipeline(base)
  }, [
    s.tjSeries, s.lifetimeModel, s.modelParamsInput,
    s.safetyFactor,
    commonPayload, runPipeline, patch,
  ])

  /* ---- Tab 2 handler ---- */
  const handleDamageCalc = useCallback(async () => {
    if (!s.tjSeries) {
      patch({ error: '请先在「功耗变温度」页输入数据' })
      return
    }
    if (!s.lifetimeModel) {
      patch({ error: '请选择寿命模型' })
      return
    }
    const base: Parameters<typeof apiService.runRainflowPipeline>[0] = {
      junctionTemperature: s.tjSeries,
      ...commonPayload(),
    }
    const mp: Record<string, number> = {}
    Object.entries(s.modelParamsInput).forEach(([k, v]) => {
      const n = parseFloat(v); if (!isNaN(n)) mp[k] = n
    })
    base.lifetimeModel = s.lifetimeModel
    base.modelParams = mp
    base.safetyFactor = parseFloat(s.safetyFactor) || 1.0
    await runPipeline(base)
  }, [
    s.tjSeries, s.lifetimeModel, s.modelParamsInput,
    s.safetyFactor,
    commonPayload, runPipeline, patch,
  ])

  /* ================================================================ */
  /*  Chart options                                                    */
  /* ================================================================ */

  /* Tj curve (single + multi-source) */
  const tjChartOption = useMemo(() => {
    const dtVal = parseFloat(s.dt) || 1.0
    const colors = ['#1565c0', '#d32f2f', '#388e3c', '#f57c00']

    // Multi-source: overlay all nodes
    if (s.allTjSeries && Object.keys(s.allTjSeries).length > 0) {
      const names = Object.keys(s.allTjSeries)
      return {
        title: {
          text: '多热源结温曲线 Tj(t)',
          left: 'center',
          textStyle: { fontSize: 14 },
        },
        tooltip: { trigger: 'axis' as const },
        legend: { data: names, bottom: 0 },
        grid: { left: 60, right: 20, bottom: 40 },
        xAxis: { type: 'value' as const, name: '时间 (s)' },
        yAxis: { type: 'value' as const, name: 'Tj (°C)' },
        series: names.map((name, idx) => ({
          type: 'line' as const,
          name,
          data: s.allTjSeries![name].map((v, i) => [i * dtVal, v]),
          smooth: false,
          lineStyle: { width: 1.5, color: colors[idx % colors.length] },
          symbol: 'none',
        })),
      }
    }

    // Single source
    if (!s.tjSeries?.length) return null
    return {
      title: {
        text: '结温曲线 Tj(t)',
        left: 'center',
        textStyle: { fontSize: 14 },
      },
      tooltip: { trigger: 'axis' as const },
      grid: { left: 60, right: 20, bottom: 40 },
      xAxis: { type: 'value' as const, name: '时间 (s)' },
      yAxis: { type: 'value' as const, name: 'Tj (°C)' },
      series: [
        {
          type: 'line' as const,
          data: s.tjSeries.map((v, i) => [i * dtVal, v]),
          smooth: false,
          lineStyle: { width: 1, color: '#1565c0' },
          symbol: 'none',
        },
      ],
    }
  }, [s.tjSeries, s.allTjSeries, s.dt])

  /* From-To matrix heatmap */
  const fromToOption = useMemo(() => {
    if (!s.result?.fromToMatrix) return null
    const { matrix, n_band } = s.result.fromToMatrix
    const heatData: number[][] = []
    let maxVal = 1
    for (let from = 0; from < n_band; from++) {
      for (let to = 0; to < n_band; to++) {
        const v = matrix[from][to]
        if (v > 0) {
          heatData.push([to, from, v])
          maxVal = Math.max(maxVal, v)
        }
      }
    }
    const labels = Array.from({ length: n_band }, (_, i) => String(i + 1))
    return {
      title: {
        text: 'Rainflow matrix',
        left: 'center',
        textStyle: { fontSize: 14 },
      },
      tooltip: {
        position: 'top' as const,
        formatter: (p: any) =>
          `From ${p.data[1] + 1} → To ${p.data[0] + 1}: ${p.data[2]}`,
      },
      grid: { left: 50, right: 100, top: 40, bottom: 30 },
      xAxis: {
        type: 'category' as const,
        data: labels,
        name: 'To',
        position: 'top' as const,
        splitArea: { show: true },
      },
      yAxis: {
        type: 'category' as const,
        data: labels,
        name: 'From',
        inverse: true,
        splitArea: { show: true },
      },
      visualMap: {
        min: 0,
        max: maxVal,
        calculable: true,
        orient: 'vertical' as const,
        right: 0,
        top: 'center',
        inRange: { color: ['#ffffff', '#ffe0b2', '#ff8f00', '#d32f2f'] },
      },
      series: [
        {
          type: 'heatmap' as const,
          data: heatData,
          label: { show: heatData.length < 600, fontSize: 9 },
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,.5)' },
          },
        },
      ],
    }
  }, [s.result?.fromToMatrix])

  /* Amplitude histogram */
  const histOption = useMemo(() => {
    if (!s.result?.amplitudeHistogram) return null
    const { bin_centers, counts_total } = s.result.amplitudeHistogram
    if (!bin_centers?.length) return null
    return {
      title: {
        text: '温度幅值分布',
        left: 'center',
        textStyle: { fontSize: 14 },
      },
      tooltip: { trigger: 'axis' as const },
      grid: { left: 50, right: 20, bottom: 50 },
      xAxis: {
        type: 'category' as const,
        data: bin_centers.map((v: number) => v.toFixed(2)),
        name: '温度幅值 (°C)',
        axisLabel: { rotate: 45 },
      },
      yAxis: { type: 'value' as const, name: '次数' },
      series: [
        {
          type: 'bar' as const,
          data: counts_total,
          itemStyle: { color: '#1565c0' },
        },
      ],
    }
  }, [s.result?.amplitudeHistogram])

  /* Residual plot */
  const residualOption = useMemo(() => {
    if (!s.result?.residual?.length) return null
    return {
      title: {
        text: '温度残余',
        left: 'center',
        textStyle: { fontSize: 14 },
      },
      tooltip: { trigger: 'axis' as const },
      grid: { left: 60, right: 20, bottom: 40 },
      xAxis: {
        type: 'category' as const,
        data: s.result.residual.map((_: number, i: number) => i),
      },
      yAxis: { type: 'value' as const, name: 'Temp (°C)' },
      series: [
        {
          type: 'line' as const,
          data: s.result.residual,
          lineStyle: { width: 1.5, color: '#1565c0' },
          symbol: 'circle',
          symbolSize: 4,
        },
      ],
    }
  }, [s.result?.residual])

  /* ================================================================ */
  /*  CSV export                                                       */
  /* ================================================================ */
  const handleExportMatrix = useCallback(() => {
    if (!s.result?.fromToMatrix) return
    const { matrix, band_values, n_band } = s.result.fromToMatrix
    const header = [
      'From\\To',
      ...band_values.map((v: number) => v.toFixed(2)),
    ].join(',')
    const rows = matrix.map((row: number[], i: number) =>
      [band_values[i].toFixed(2), ...row.map(String)].join(','),
    )
    const csv = [header, ...rows].join('\n')
    const blob = new Blob(['\ufeff' + csv], {
      type: 'text/csv;charset=utf-8;',
    })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `from_to_matrix_${n_band}band_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(link.href)
  }, [s.result?.fromToMatrix])

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */
  return (
    <Container maxWidth="xl">
      <Stack spacing={2} sx={{ py: 2 }}>
        <Typography variant="h4">功率循环寿命评估</Typography>
        <Typography variant="body2" color="text.secondary">
          功耗曲线 + 热阻抗 → 结温 → 雨流计数 → Miner 累积损伤 → 寿命评估
        </Typography>

        <Tabs value={s.tab} onChange={(_, v) => patch({ tab: v })}>
          <Tab label="功耗变温度" />
          <Tab label="雨流计数法" />
          <Tab label="寿命计算" />
        </Tabs>

        {s.loading && (
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <CircularProgress size={28} />
            <Typography sx={{ mt: 1 }}>计算中…</Typography>
          </Paper>
        )}
        {s.error && (
          <Alert
            severity={s.error.startsWith('⚠') ? 'warning' : 'error'}
            onClose={() => patch({ error: null })}
          >
            {s.error}
          </Alert>
        )}

        {/* ── Persistent Tj chart (shown on ALL tabs when data exists) ── */}
        {tjChartOption && (
          <Paper sx={{ p: 2 }}>
            <ReactECharts option={tjChartOption} style={{ height: 280 }} />
          </Paper>
        )}

        {/* ========================================================= */}
        {/*  Tab 0 — 功耗变温度                                        */}
        {/* ========================================================= */}
        <TabPanel value={s.tab} index={0}>
          <Stack spacing={2}>
            {/* Mode + basic params */}
            <Paper sx={{ p: 2 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>输入模式</InputLabel>
                    <Select
                      value={s.inputMode}
                      label="输入模式"
                      onChange={(e) =>
                        patch({ inputMode: e.target.value as InputMode })
                      }
                    >
                      <MenuItem value="power">功耗 P(t) + 热阻抗 Zth</MenuItem>
                      <MenuItem value="tj">直接输入结温 Tj(t)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                {s.inputMode === 'power' && (
                  <>
                    <Grid item xs={6} sm={2}>
                      <TextField
                        label="采样间隔 dt (s)"
                        size="small"
                        fullWidth
                        value={s.dt}
                        onChange={(e) => patch({ dt: e.target.value })}
                      />
                    </Grid>
                    <Grid item xs={6} sm={2}>
                      <TextField
                        label="环境温度 (°C)"
                        size="small"
                        fullWidth
                        value={s.ambientTemp}
                        onChange={(e) =>
                          patch({ ambientTemp: e.target.value })
                        }
                      />
                    </Grid>
                    {s.sourceCount <= 1 && (
                      <Grid item xs={12} sm={4}>
                        <FormControl fullWidth size="small">
                          <InputLabel>热阻抗类型</InputLabel>
                          <Select
                            value={s.zthMode}
                            label="热阻抗类型"
                            onChange={(e) =>
                              patch({ zthMode: e.target.value as ZthMode })
                            }
                          >
                            <MenuItem value="foster">Foster RC 网络</MenuItem>
                            <MenuItem value="sampled">采样 Zth 曲线</MenuItem>
                          </Select>
                        </FormControl>
                      </Grid>
                    )}
                  </>
                )}
              </Grid>
              {/* Source count row (power mode) */}
              {s.inputMode === 'power' && (
                <Grid container spacing={2} alignItems="center" sx={{ mt: 1 }}>
                  <Grid item xs={6} sm={2}>
                    <TextField
                      label="热源数量"
                      type="number"
                      size="small"
                      fullWidth
                      value={s.sourceCount}
                      onChange={(e) => {
                        const cnt = Math.max(1, Math.min(4, parseInt(e.target.value) || 1))
                        const names = ['IGBT', 'Diode', 'FWD', 'NTC'].slice(0, cnt)
                        const powerData = Array(cnt).fill('')
                        const zthMat: string[][] = []
                        for (let i = 0; i < cnt; i++) {
                          const row: string[] = []
                          for (let j = 0; j < cnt; j++) {
                            row.push(i === j ? DEFAULT_ZTH_SELF : DEFAULT_ZTH_CROSS)
                          }
                          zthMat.push(row)
                        }
                        patch({
                          sourceCount: cnt,
                          sourceNames: names,
                          sourcePowerData: powerData,
                          zthMatrixInput: zthMat,
                        })
                      }}
                      inputProps={{ min: 1, max: 4 }}
                    />
                  </Grid>
                  {s.sourceCount > 1 && (
                    <>
                      {Array.from({ length: s.sourceCount }, (_, i) => (
                        <Grid item xs={6} sm={2} key={i}>
                          <TextField
                            label={`热源 ${i + 1} 名称`}
                            size="small"
                            fullWidth
                            value={s.sourceNames[i] || ''}
                            onChange={(e) => {
                              const updated = [...s.sourceNames]
                              updated[i] = e.target.value
                              patch({ sourceNames: updated })
                            }}
                          />
                        </Grid>
                      ))}
                      <Grid item xs={6} sm={2}>
                        <FormControl fullWidth size="small">
                          <InputLabel>分析节点</InputLabel>
                          <Select
                            value={s.targetNode}
                            label="分析节点"
                            onChange={(e) =>
                              patch({ targetNode: e.target.value as number })
                            }
                          >
                            {s.sourceNames.slice(0, s.sourceCount).map((nm, i) => (
                              <MenuItem key={i} value={i}>{nm}</MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Grid>
                    </>
                  )}
                </Grid>
              )}
            </Paper>

            {/* Zth params (single source only) */}
            {s.inputMode === 'power' && s.sourceCount <= 1 && (
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  热阻抗参数{' '}
                  {s.zthMode === 'foster'
                    ? '(Foster RC 网络)'
                    : '(采样 Zth 阶跃响应)'}
                </Typography>
                {s.zthMode === 'foster' ? (
                  <>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 1 }}
                    >
                      每行一组 R (K/W), τ (s)，逗号或空格分隔。来自数据手册 Zth(t) =
                      Σ Rᵢ(1−e^(−t/τᵢ))
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      minRows={3}
                      maxRows={8}
                      size="small"
                      value={s.fosterInput}
                      onChange={(e) =>
                        patch({ fosterInput: e.target.value })
                      }
                      placeholder={'0.05, 0.001\n0.15, 0.01\n0.30, 0.1\n0.50, 1.0'}
                    />
                  </>
                ) : (
                  <>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 1 }}
                    >
                      输入 Zth 阶跃响应采样值（K/W），以逗号或换行分隔，采样间隔同 dt
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      minRows={3}
                      maxRows={8}
                      size="small"
                      value={s.zthInput}
                      onChange={(e) => patch({ zthInput: e.target.value })}
                      placeholder="0.01, 0.03, 0.08, 0.15, 0.25, 0.38, 0.50, ..."
                    />
                  </>
                )}
              </Paper>
            )}

            {/* Multi-source: Zth matrix + power data per source */}
            {s.inputMode === 'power' && s.sourceCount > 1 && (
              <>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    热阻抗矩阵 Zth (Foster RC)
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    每个单元格输入 Foster RC 参数，每行一组 R (K/W), τ (s)。
                    Zth[i][j] 表示热源 j 对节点 i 的热阻抗路径。
                  </Typography>
                  <Grid container spacing={1}>
                    {/* Column headers */}
                    <Grid item xs={2} />
                    {Array.from({ length: s.sourceCount }, (_, j) => (
                      <Grid item xs={Math.floor(10 / s.sourceCount)} key={`hdr-${j}`}>
                        <Typography variant="caption" fontWeight="bold" textAlign="center" display="block">
                          {s.sourceNames[j] || `Source ${j}`} →
                        </Typography>
                      </Grid>
                    ))}
                    {/* Matrix rows */}
                    {Array.from({ length: s.sourceCount }, (_, i) => (
                      <React.Fragment key={`row-${i}`}>
                        <Grid item xs={2} sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="caption" fontWeight="bold">
                            → {s.sourceNames[i] || `Node ${i}`}
                          </Typography>
                        </Grid>
                        {Array.from({ length: s.sourceCount }, (_, j) => (
                          <Grid item xs={Math.floor(10 / s.sourceCount)} key={`cell-${i}-${j}`}>
                            <TextField
                              fullWidth
                              multiline
                              minRows={2}
                              maxRows={4}
                              size="small"
                              value={s.zthMatrixInput[i]?.[j] || ''}
                              onChange={(e) => {
                                const mat = s.zthMatrixInput.map((r) => [...r])
                                if (!mat[i]) mat[i] = []
                                mat[i][j] = e.target.value
                                patch({ zthMatrixInput: mat })
                              }}
                              placeholder={i === j ? '0.05,0.001\n0.15,0.01' : '0.01,0.005'}
                              sx={{ '& textarea': { fontSize: '0.75rem' } }}
                            />
                          </Grid>
                        ))}
                      </React.Fragment>
                    ))}
                  </Grid>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    功耗时间序列（每个热源）
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    为每个热源分别输入功耗 P(t) (W)，以逗号/空格/换行分隔
                  </Typography>
                  <Grid container spacing={2}>
                    {Array.from({ length: s.sourceCount }, (_, i) => (
                      <Grid item xs={12} sm={6} key={i}>
                        <Typography variant="body2" fontWeight="bold" sx={{ mb: 0.5 }}>
                          {s.sourceNames[i] || `热源 ${i + 1}`}
                        </Typography>
                        <TextField
                          fullWidth
                          multiline
                          minRows={3}
                          maxRows={6}
                          size="small"
                          value={s.sourcePowerData[i] || ''}
                          onChange={(e) => {
                            const pd = [...s.sourcePowerData]
                            pd[i] = e.target.value
                            patch({ sourcePowerData: pd })
                          }}
                          placeholder="100, 200, 150, 80, ..."
                        />
                      </Grid>
                    ))}
                  </Grid>
                  <Button
                    variant="contained"
                    sx={{ mt: 2 }}
                    onClick={handleMultiSourceSubmit}
                    disabled={s.loading}
                  >
                    多热源计算
                  </Button>
                </Paper>
              </>
            )}

            {/* Data input (single source / direct Tj) */}
            {(s.sourceCount <= 1 || s.inputMode !== 'power') && (
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  {s.inputMode === 'power'
                    ? '输入功耗时间序列 P(t) (W)'
                    : '输入结温时间序列 Tj(t) (°C)'}
                </Typography>
                <TimeHistoryInput
                  onDataSubmit={
                    s.inputMode === 'power' ? handlePowerSubmit : handleTjSubmit
                  }
                />
              </Paper>
            )}
          </Stack>
        </TabPanel>

        {/* ========================================================= */}
        {/*  Tab 1 — 雨流计数法                                        */}
        {/* ========================================================= */}
        <TabPanel value={s.tab} index={1}>
          <Stack spacing={2}>
            {/* params row */}
            <Paper sx={{ p: 2 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={6} sm={2}>
                  <TextField
                    label="Nband"
                    type="number"
                    size="small"
                    fullWidth
                    value={s.nBand}
                    onChange={(e) => patch({ nBand: e.target.value })}
                  />
                </Grid>
                <Grid item xs={6} sm={2}>
                  <TextField
                    label="Ymin (°C)"
                    size="small"
                    fullWidth
                    value={s.yMin}
                    onChange={(e) => patch({ yMin: e.target.value })}
                    placeholder="自动"
                  />
                </Grid>
                <Grid item xs={6} sm={2}>
                  <TextField
                    label="Ymax (°C)"
                    size="small"
                    fullWidth
                    value={s.yMax}
                    onChange={(e) => patch({ yMax: e.target.value })}
                    placeholder="自动"
                  />
                </Grid>
                <Grid item xs={6} sm={2}>
                  <TextField
                    label="忽略 ΔTj < (°C)"
                    size="small"
                    fullWidth
                    value={s.ignoreBelow}
                    onChange={(e) =>
                      patch({ ignoreBelow: e.target.value })
                    }
                  />
                </Grid>
                <Grid item xs={6} sm={2}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={s.rearrange}
                        onChange={(e) =>
                          patch({ rearrange: e.target.checked })
                        }
                        size="small"
                      />
                    }
                    label="重排"
                  />
                </Grid>
                <Grid item xs={6} sm={2}>
                  <Stack direction="row" spacing={1}>
                    <Button
                      variant="contained"
                      size="small"
                      onClick={handleRecompute}
                      disabled={!s.tjSeries || s.loading}
                    >
                      计算
                    </Button>
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={handleExportMatrix}
                      disabled={!s.result?.fromToMatrix}
                    >
                      导出
                    </Button>
                  </Stack>
                </Grid>
              </Grid>
            </Paper>

            {!s.result && !s.tjSeries && (
              <Alert severity="info">
                请先在「功耗变温度」页输入数据并提交
              </Alert>
            )}

            {s.result && (
              <>
                {/* summary stats */}
                <Grid container spacing={2}>
                  {[
                    {
                      label: '总循环数',
                      value: s.result.totalCycles.toLocaleString(),
                    },
                    {
                      label: '最大 ΔTj',
                      value: `${s.result.maxRange.toFixed(1)} °C`,
                    },
                    {
                      label: '循环种类',
                      value: String(s.result.cycles.length),
                    },
                    {
                      label: '残余点数',
                      value: String(s.result.residual?.length ?? 0),
                    },
                  ].map((item) => (
                    <Grid item xs={6} sm={3} key={item.label}>
                      <Paper sx={{ p: 1.5, textAlign: 'center' }}>
                        <Typography variant="caption" color="text.secondary">
                          {item.label}
                        </Typography>
                        <Typography variant="h6">{item.value}</Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>

                {/* From-To matrix (left) + band table (right) */}
                <Grid container spacing={2}>
                  <Grid item xs={12} md={8}>
                    {fromToOption && (
                      <Paper sx={{ p: 1 }}>
                        <ReactECharts
                          option={fromToOption}
                          style={{ height: 500 }}
                        />
                      </Paper>
                    )}
                  </Grid>
                  <Grid item xs={12} md={4}>
                    {s.result.fromToMatrix && (
                      <Paper sx={{ p: 1 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          温度带值
                        </Typography>
                        <TableContainer sx={{ maxHeight: 500 }}>
                          <Table size="small" stickyHeader>
                            <TableHead>
                              <TableRow>
                                <TableCell>band_index</TableCell>
                                <TableCell>band_value</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {s.result.fromToMatrix.band_values.map(
                                (v: number, i: number) => (
                                  <TableRow key={i}>
                                    <TableCell>{i + 1}</TableCell>
                                    <TableCell>{v.toFixed(2)}</TableCell>
                                  </TableRow>
                                ),
                              )}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Paper>
                    )}
                  </Grid>
                </Grid>

                {/* histogram (left) + residual (right) */}
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    {histOption && (
                      <Paper sx={{ p: 1 }}>
                        <ReactECharts
                          option={histOption}
                          style={{ height: 300 }}
                        />
                      </Paper>
                    )}
                  </Grid>
                  <Grid item xs={12} md={6}>
                    {residualOption && (
                      <Paper sx={{ p: 1 }}>
                        <ReactECharts
                          option={residualOption}
                          style={{ height: 300 }}
                        />
                      </Paper>
                    )}
                  </Grid>
                </Grid>
              </>
            )}
          </Stack>
        </TabPanel>

        {/* ========================================================= */}
        {/*  Tab 2 — 寿命计算                                          */}
        {/* ========================================================= */}
        <TabPanel value={s.tab} index={2}>
          <Stack spacing={2}>
            {/* Model selector + safety factor */}
            <Paper sx={{ p: 2 }}>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={4}>
                  <FormControl fullWidth size="small">
                    <InputLabel>寿命模型</InputLabel>
                    <Select
                      value={s.lifetimeModel}
                      label="寿命模型"
                      onChange={(e) => {
                        const name = e.target.value as string
                        const cfg = MODEL_PARAMS_CONFIG[name]
                        if (cfg) {
                          const defaults: Record<string, string> = {}
                          cfg.params.forEach((p) => {
                            defaults[p.key] = p.defaultValue
                          })
                          patch({ lifetimeModel: name, modelParamsInput: defaults })
                        } else {
                          patch({ lifetimeModel: name })
                        }
                      }}
                    >
                      {Object.entries(MODEL_PARAMS_CONFIG).map(([key, cfg]) => (
                        <MenuItem key={key} value={key}>{cfg.label}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <TextField
                    label="安全系数 f_safe"
                    size="small"
                    fullWidth
                    value={s.safetyFactor}
                    onChange={(e) => patch({ safetyFactor: e.target.value })}
                  />
                </Grid>
              </Grid>
            </Paper>

            {/* Model params */}
            <Paper sx={{ p: 2 }}>
              {MODEL_PARAMS_CONFIG[s.lifetimeModel] && (
                <>
                  <Typography variant="subtitle1" gutterBottom>
                    {MODEL_PARAMS_CONFIG[s.lifetimeModel].label} 模型参数
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ mb: 2, fontFamily: 'monospace' }}
                  >
                    {MODEL_PARAMS_CONFIG[s.lifetimeModel].formula}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    ΔTj、Tj_max、Tj_min、Tj_mean 由雨流计数结果自动推导，无需手动输入
                  </Typography>
                  <Grid container spacing={2}>
                    {MODEL_PARAMS_CONFIG[s.lifetimeModel].params.map((p) => (
                      <Grid item xs={6} sm={3} key={p.key}>
                        <TextField
                          label={p.unit ? `${p.label} (${p.unit})` : p.label}
                          size="small"
                          fullWidth
                          value={s.modelParamsInput[p.key] ?? p.defaultValue}
                          onChange={(e) =>
                            patch({
                              modelParamsInput: {
                                ...s.modelParamsInput,
                                [p.key]: e.target.value,
                              },
                            })
                          }
                        />
                      </Grid>
                    ))}
                  </Grid>
                </>
              )}
              <Button
                variant="contained"
                sx={{ mt: 2 }}
                onClick={handleDamageCalc}
                disabled={!s.tjSeries || s.loading}
              >
                计算寿命
              </Button>
            </Paper>

            {!s.result && (
              <Alert severity="info">请先输入数据并计算</Alert>
            )}

            {s.result && (
              <>
                {/* Thermal summary */}
                {s.result.thermalSummary && (
                  <Grid container spacing={2}>
                    {[
                      {
                        label: 'Tj_max',
                        value: `${s.result.thermalSummary.tj_max.toFixed(1)} °C`,
                      },
                      {
                        label: 'Tj_min',
                        value: `${s.result.thermalSummary.tj_min.toFixed(1)} °C`,
                      },
                      {
                        label: 'Tj_mean',
                        value: `${s.result.thermalSummary.tj_mean.toFixed(1)} °C`,
                      },
                      {
                        label: 'ΔTj_max',
                        value: `${s.result.thermalSummary.delta_tj_max.toFixed(1)} °C`,
                      },
                    ].map((item) => (
                      <Grid item xs={6} sm={3} key={item.label}>
                        <Paper sx={{ p: 1.5, textAlign: 'center' }}>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                          >
                            {item.label}
                          </Typography>
                          <Typography variant="h6">{item.value}</Typography>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                )}

                {/* Miner damage stats */}
                {s.result.damage && (
                  <Grid container spacing={2}>
                    {[
                      {
                        label: '每工况块损伤 CDI',
                        value: Number(
                          s.result.damage.total_damage_per_block ?? 0,
                        ).toExponential(3),
                      },
                      {
                        label: '可重复工况块数 (1/CDI)',
                        value: s.result.damage.blocks_to_failure
                          ? Number(
                              s.result.damage.blocks_to_failure,
                            ).toFixed(1)
                          : '-',
                      },
                      ...(s.result.damage.safety_factor != null
                        ? [{
                            label: '安全系数 f_safe',
                            value: String(s.result.damage.safety_factor),
                          }]
                        : []),
                      ...(s.result.damage.model_used
                        ? [{
                            label: '使用模型',
                            value: String(s.result.damage.model_used),
                          }]
                        : []),
                    ].map((item) => (
                      <Grid item xs={6} sm={3} key={item.label}>
                        <Paper sx={{ p: 1.5, textAlign: 'center' }}>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                          >
                            {item.label}
                          </Typography>
                          <Typography variant="h6">{item.value}</Typography>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                )}

                {/* Model damage per-cycle details table */}
                {s.result.modelDamage?.cycle_details &&
                  s.result.modelDamage.cycle_details.length > 0 && (
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      模型寿命计算明细（每类循环）
                    </Typography>
                    <TableContainer sx={{ maxHeight: 400 }}>
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>#</TableCell>
                            <TableCell>ΔTj (°C)</TableCell>
                            <TableCell>平均温度 (°C)</TableCell>
                            <TableCell>循环次数</TableCell>
                            <TableCell>Nf</TableCell>
                            <TableCell>损伤 ni/Nfi</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {s.result.modelDamage.cycle_details.map((cd, i) => (
                            <TableRow key={i}>
                              <TableCell>{i + 1}</TableCell>
                              <TableCell>{cd.delta_tj.toFixed(2)}</TableCell>
                              <TableCell>{cd.mean_tj.toFixed(2)}</TableCell>
                              <TableCell>{cd.count}</TableCell>
                              <TableCell>{cd.nf.toExponential(3)}</TableCell>
                              <TableCell>{cd.damage.toExponential(3)}</TableCell>
                            </TableRow>
                          ))}
                          {/* Total damage summary row */}
                          <TableRow sx={{ '& td': { fontWeight: 'bold', borderTop: 2 } }}>
                            <TableCell colSpan={3}>合计</TableCell>
                            <TableCell>
                              {s.result.modelDamage.cycle_details
                                .reduce((sum, cd) => sum + cd.count, 0)}
                            </TableCell>
                            <TableCell>—</TableCell>
                            <TableCell>
                              {s.result.modelDamage.cycle_details
                                .reduce((sum, cd) => sum + cd.damage, 0)
                                .toExponential(3)}
                            </TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Paper>
                )}

                {/* No damage hint */}
                {!s.result.damage && s.result.totalCycles > 0 && (
                  <Alert severity="info">
                    上方已有雨流计数结果，点击「计算寿命」可计算 Miner 累积损伤。
                  </Alert>
                )}
              </>
            )}
          </Stack>
        </TabPanel>
      </Stack>
    </Container>
  )
}

export default RainflowCounting
