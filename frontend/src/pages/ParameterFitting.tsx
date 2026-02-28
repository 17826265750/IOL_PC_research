/**
 * 功率模块寿命分析软件 - 参数拟合页面
 * @author GSH
 */
import React, { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Chip,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Checkbox,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
  Save as SaveIcon,
  Upload as UploadIcon,
  Science as ScienceIcon,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import apiService from '@/services/api'
import type { LifetimeModelType } from '@/types'

interface ExperimentDataRow {
  id: number
  deltaTj: number
  Tjmax: number
  ton: number
  I: number
  V: number
  D: number
  Nf: number
}

interface FittingResult {
  parameters: Record<string, number>
  std_errors: Record<string, number>
  r_squared: number
  rmse: number
  confidence_intervals: Record<string, [number | null, number | null]>
  fixed_params?: Record<string, number>
  fixed_data_values?: Record<string, number>
  auto_fixed_info?: string[]
  couple_vd_to_k?: boolean
}

const MODEL_OPTIONS: { value: LifetimeModelType; label: string }[] = [
  { value: 'cips2008', label: 'CIPS 2008 (Bayerer)' },
  { value: 'coffin_manson', label: 'Coffin-Manson' },
  { value: 'coffin_manson_arrhenius', label: 'Coffin-Manson-Arrhenius' },
  { value: 'norris_landzberg', label: 'Norris-Landzberg' },
  { value: 'lesit', label: 'LESIT' },
]

// Per-model fitting configuration
type ColName = 'deltaTj' | 'Tjmax' | 'ton' | 'I' | 'V' | 'D'
type FixedName = 'ton' | 'I' | 'V' | 'D'

interface ModelFitCfg {
  columns: ColName[]        // visible data columns (Nf always shown)
  fixedParams: FixedName[]  // available fixed-parameter checkboxes
  resultParams: string[]    // parameter names returned by fitting
  paramCount: number        // total free params before fixing
  formula: string
}

const MODEL_FIT_CFG: Record<string, ModelFitCfg> = {
  coffin_manson: {
    columns: ['deltaTj'],
    fixedParams: [],
    resultParams: ['A', 'alpha'],
    paramCount: 2,
    formula: 'Nf = A × (ΔTj)^α',
  },
  coffin_manson_arrhenius: {
    columns: ['deltaTj', 'Tjmax'],
    fixedParams: [],
    resultParams: ['A', 'alpha', 'Ea'],
    paramCount: 3,
    formula: 'Nf = A × (ΔTj)^α × exp(Ea/(kB×Tj_mean))',
  },
  norris_landzberg: {
    columns: ['deltaTj', 'Tjmax', 'ton'],
    fixedParams: ['ton'],
    resultParams: ['A', 'alpha', 'beta', 'Ea'],
    paramCount: 4,
    formula: 'Nf = A × (ΔTj)^α × f^β × exp(Ea/(kB×Tj_max))',
  },
  lesit: {
    columns: ['deltaTj', 'Tjmax'],
    fixedParams: [],
    resultParams: ['A', 'alpha', 'Q'],
    paramCount: 3,
    formula: 'Nf = A × (ΔTj)^α × exp(Q/(R×Tj_min))',
  },
  cips2008: {
    columns: ['deltaTj', 'Tjmax', 'ton', 'I', 'V', 'D'],
    fixedParams: ['ton', 'I', 'V', 'D'],
    resultParams: ['K', 'β1', 'β2', 'β3', 'β4', 'β5', 'β6'],
    paramCount: 7,
    formula: 'Nf = K × (ΔTj)^β1 × exp(β2/Tj_max) × ton^β3 × I^β4 × V^β5 × D^β6',
  },
}

const COLUMN_LABELS: Record<ColName, { header: string; sub: string; unit: string; width: number }> = {
  deltaTj: { header: 'ΔTj (K)', sub: '温度摆幅', unit: '', width: 90 },
  Tjmax: { header: 'Tjmax (°C)', sub: '最高结温', unit: '', width: 90 },
  ton: { header: 'ton (s)', sub: '加热时间', unit: '', width: 80 },
  I: { header: 'I (A)', sub: '负载电流', unit: '', width: 80 },
  V: { header: 'V (V)', sub: '阻断电压', unit: '', width: 90 },
  D: { header: 'D (μm)', sub: '键合线直径', unit: '', width: 80 },
}

const FIXED_PARAM_LABELS: Record<FixedName, string> = {
  ton: 'ton (s)',
  I: 'I (A)',
  V: 'V (V)',
  D: 'D (μm)',
}

// Parameter display names for results
const PARAM_DISPLAY: Record<string, { label: string; unit: string }> = {
  K: { label: 'K', unit: '' },
  A: { label: 'A', unit: '' },
  alpha: { label: 'α', unit: '' },
  beta: { label: 'β', unit: '' },
  Ea: { label: 'Ea', unit: 'eV' },
  Q: { label: 'Q', unit: 'eV' },
  'β1': { label: 'β1', unit: '' },
  'β2': { label: 'β2', unit: 'K' },
  'β3': { label: 'β3', unit: '' },
  'β4': { label: 'β4', unit: '' },
  'β5': { label: 'β5', unit: '' },
  'β6': { label: 'β6', unit: '' },
}

// Local storage key for fitted parameters
const FITTED_PARAMS_KEY = 'cips_fitted_parameters'
const EXPERIMENT_DATA_KEY = 'cips_experiment_data'
const FIXED_PARAMS_KEY = 'cips_fixed_params'
const FITTING_RESULT_KEY = 'cips_fitting_result'
const SELECTED_MODEL_KEY = 'cips_selected_model'

// 固定参数配置
interface FixedParamConfig {
  ton: { enabled: boolean; value: number }
  I: { enabled: boolean; value: number }
  V: { enabled: boolean; value: number }
  D: { enabled: boolean; value: number }
}

const FIXED_PARAM_TO_FIELD: Record<keyof FixedParamConfig, keyof ExperimentDataRow> = {
  ton: 'ton',
  I: 'I',
  V: 'V',
  D: 'D',
}

export const ParameterFitting: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<LifetimeModelType>(() => {
    const saved = localStorage.getItem(SELECTED_MODEL_KEY)
    return saved ? (saved as LifetimeModelType) : 'cips2008'
  })

  // 从localStorage加载试验数据
  const [experimentData, setExperimentData] = useState<ExperimentDataRow[]>(() => {
    const saved = localStorage.getItem(EXPERIMENT_DATA_KEY)
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        return [
          { id: 1, deltaTj: 60, Tjmax: 100, ton: 2, I: 100, V: 1200, D: 300, Nf: 500000 },
          { id: 2, deltaTj: 70, Tjmax: 110, ton: 2, I: 100, V: 1200, D: 300, Nf: 300000 },
          { id: 3, deltaTj: 80, Tjmax: 125, ton: 2, I: 100, V: 1200, D: 300, Nf: 150000 },
          { id: 4, deltaTj: 90, Tjmax: 140, ton: 2, I: 100, V: 1200, D: 300, Nf: 80000 },
          { id: 5, deltaTj: 100, Tjmax: 150, ton: 2, I: 100, V: 1200, D: 300, Nf: 50000 },
          { id: 6, deltaTj: 110, Tjmax: 160, ton: 2, I: 100, V: 1200, D: 300, Nf: 30000 },
          { id: 7, deltaTj: 120, Tjmax: 175, ton: 2, I: 100, V: 1200, D: 300, Nf: 20000 },
        ]
      }
    }
    return [
      { id: 1, deltaTj: 60, Tjmax: 100, ton: 2, I: 100, V: 1200, D: 300, Nf: 500000 },
      { id: 2, deltaTj: 70, Tjmax: 110, ton: 2, I: 100, V: 1200, D: 300, Nf: 300000 },
      { id: 3, deltaTj: 80, Tjmax: 125, ton: 2, I: 100, V: 1200, D: 300, Nf: 150000 },
      { id: 4, deltaTj: 90, Tjmax: 140, ton: 2, I: 100, V: 1200, D: 300, Nf: 80000 },
      { id: 5, deltaTj: 100, Tjmax: 150, ton: 2, I: 100, V: 1200, D: 300, Nf: 50000 },
      { id: 6, deltaTj: 110, Tjmax: 160, ton: 2, I: 100, V: 1200, D: 300, Nf: 30000 },
      { id: 7, deltaTj: 120, Tjmax: 175, ton: 2, I: 100, V: 1200, D: 300, Nf: 20000 },
    ]
  })

  const [fixedParams, setFixedParams] = useState<FixedParamConfig>(() => {
    const saved = localStorage.getItem(FIXED_PARAMS_KEY)
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        // ignore
      }
    }
    return {
      ton: { enabled: false, value: 2 },
      I: { enabled: false, value: 100 },
      V: { enabled: false, value: 1200 },
      D: { enabled: false, value: 300 },
    }
  })
  const [fittingResult, setFittingResult] = useState<FittingResult | null>(() => {
    const saved = localStorage.getItem(FITTING_RESULT_KEY)
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        // ignore
      }
    }
    return null
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [savedParams, setSavedParams] = useState<Record<string, Record<string, number>>>(() => {
    const saved = localStorage.getItem(FITTED_PARAMS_KEY)
    return saved ? JSON.parse(saved) : {}
  })
  const [currentTab, setCurrentTab] = useState(0)

  // 保存状态到localStorage
  React.useEffect(() => {
    localStorage.setItem(EXPERIMENT_DATA_KEY, JSON.stringify(experimentData))
  }, [experimentData])

  React.useEffect(() => {
    localStorage.setItem(FIXED_PARAMS_KEY, JSON.stringify(fixedParams))
  }, [fixedParams])

  // 当固定参数启用或值变化时，同步覆盖试验数据对应列，保持列内常值
  React.useEffect(() => {
    setExperimentData((prev) => {
      let changed = false
      const next = prev.map((row) => {
        const updated: ExperimentDataRow = { ...row }

        ;(['ton', 'I', 'V', 'D'] as const).forEach((param) => {
          if (!fixedParams[param].enabled) return
          const field = FIXED_PARAM_TO_FIELD[param]
          const fixedValue = fixedParams[param].value
          if (updated[field] !== fixedValue) {
            updated[field] = fixedValue
            changed = true
          }
        })

        return updated
      })

      return changed ? next : prev
    })
  }, [fixedParams])

  React.useEffect(() => {
    if (fittingResult) {
      localStorage.setItem(FITTING_RESULT_KEY, JSON.stringify(fittingResult))
    } else {
      localStorage.removeItem(FITTING_RESULT_KEY)
    }
  }, [fittingResult])

  React.useEffect(() => {
    localStorage.setItem(SELECTED_MODEL_KEY, selectedModel)
  }, [selectedModel])

  // Derived model config
  const currentCfg = MODEL_FIT_CFG[selectedModel] || MODEL_FIT_CFG.cips2008
  const visibleCols = currentCfg.columns
  const availableFixed = currentCfg.fixedParams

  // 计算最少需要的数据点数
  const getMinDataPoints = () => {
    const fixedCount = availableFixed.filter((p) => fixedParams[p].enabled).length
    const fittedParamCount = currentCfg.paramCount - fixedCount
    return Math.max(2, fittedParamCount)
  }

  const getReducedParamCount = () => {
    return availableFixed.filter((p) => fixedParams[p].enabled).length
  }

  // 更新固定参数配置
  const toggleFixedParam = (param: keyof FixedParamConfig) => {
    setFixedParams(prev => ({
      ...prev,
      [param]: { ...prev[param], enabled: !prev[param].enabled }
    }))
  }

  const updateFixedParamValue = (param: keyof FixedParamConfig, value: number) => {
    setFixedParams(prev => ({
      ...prev,
      [param]: { ...prev[param], value }
    }))
  }

  const addDataRow = useCallback(() => {
    const newId = Math.max(...experimentData.map(d => d.id), 0) + 1
    const newRow: ExperimentDataRow = {
      id: newId,
      deltaTj: 80,
      Tjmax: 125,
      ton: fixedParams.ton.enabled ? fixedParams.ton.value : 2,
      I: fixedParams.I.enabled ? fixedParams.I.value : 100,
      V: fixedParams.V.enabled ? fixedParams.V.value : 1200,  // 原始电压值(V)
      D: fixedParams.D.enabled ? fixedParams.D.value : 300,
      Nf: 50000,
    }
    setExperimentData([...experimentData, newRow])
  }, [experimentData, fixedParams])

  const removeDataRow = useCallback((id: number) => {
    setExperimentData(experimentData.filter(d => d.id !== id))
  }, [experimentData])

  const updateDataRow = useCallback((id: number, field: keyof ExperimentDataRow, value: number) => {
    setExperimentData(experimentData.map(d =>
      d.id === id ? { ...d, [field]: value } : d
    ))
  }, [experimentData])

  const handleFit = useCallback(async () => {
    setLoading(true)
    setError(null)
    setFittingResult(null)

    try {
      // 准备固定参数
      const fixedParamsForApi: Record<string, number> = {}
      if (selectedModel === 'cips2008') {
        if (fixedParams.ton.enabled) fixedParamsForApi['β3'] = -0.462
        if (fixedParams.I.enabled) fixedParamsForApi['β4'] = -0.716
        if (fixedParams.V.enabled) fixedParamsForApi['β5'] = -0.761
        if (fixedParams.D.enabled) fixedParamsForApi['β6'] = -0.5
      } else if (selectedModel === 'norris_landzberg') {
        if (fixedParams.ton.enabled) fixedParamsForApi['beta'] = -0.33
      }

      // Convert data to format expected by backend
      const data = experimentData.map(row => ({
        dTj: row.deltaTj,
        Tj_max: row.Tjmax,
        t_on: fixedParams.ton.enabled ? fixedParams.ton.value : row.ton,
        I: fixedParams.I.enabled ? fixedParams.I.value : row.I,
        V: fixedParams.V.enabled ? fixedParams.V.value : row.V,
        D: fixedParams.D.enabled ? fixedParams.D.value : row.D,
        Nf: row.Nf,
      }))

      const response = await fetch('/api/analysis/fitting/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: selectedModel,
          experiment_data: data,
          fixed_params: Object.keys(fixedParamsForApi).length > 0 ? fixedParamsForApi : null,
        }),
      })

      const result = await response.json()

      if (response.ok) {
        setFittingResult(result)
      } else {
        // 处理错误信息 - 可能是字符串或对象
        let errorMsg = '拟合失败 / Fitting failed'
        if (typeof result.detail === 'string') {
          errorMsg = result.detail
        } else if (result.detail?.msg) {
          errorMsg = result.detail.msg
        } else if (result.error) {
          errorMsg = typeof result.error === 'string' ? result.error : JSON.stringify(result.error)
        }
        setError(errorMsg)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '网络错误 / Network error')
    } finally {
      setLoading(false)
    }
  }, [experimentData, fixedParams, selectedModel])

  const handleSaveParams = useCallback(() => {
    if (!fittingResult) return

    const paramsToSave: Record<string, number> = { ...fittingResult.parameters }

    // For CIPS-2008: add fixed beta values and data values
    if (selectedModel === 'cips2008') {
      const fixedBetas = fittingResult.fixed_params ?? {}
      ;['β3', 'β4', 'β5', 'β6'].forEach((betaKey) => {
        if (betaKey in fixedBetas && !(betaKey in paramsToSave)) {
          paramsToSave[betaKey] = 0
        }
      })
    }

    // Save fixed data values for all models
    const fixedDataValues = fittingResult.fixed_data_values ?? {}
    if (typeof fixedDataValues.ton === 'number') {
      paramsToSave.theating = fixedDataValues.ton
    }
    if (typeof fixedDataValues.I === 'number') {
      paramsToSave.I = fixedDataValues.I
    }
    if (typeof fixedDataValues.V === 'number') {
      paramsToSave.V = fixedDataValues.V
    }
    if (typeof fixedDataValues.D === 'number') {
      paramsToSave.D = fixedDataValues.D
    }

    const newSavedParams = {
      ...savedParams,
      [selectedModel]: paramsToSave,
    }
    setSavedParams(newSavedParams)
    localStorage.setItem(FITTED_PARAMS_KEY, JSON.stringify(newSavedParams))
    setSaveSuccess(true)
    setTimeout(() => setSaveSuccess(false), 2000)
  }, [fittingResult, savedParams, selectedModel])

  const navigate = useNavigate()

  const handleLoadParams = useCallback((model: string) => {
    const params = savedParams[model]
    if (params) {
      // Navigate to prediction page with state
      navigate('/prediction', { state: { loadFittedParams: true, model } })
    }
  }, [savedParams, navigate])

  const handleImportCSV = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target?.result as string
      const lines = text.split('\n')
      const headers = lines[0].split(',').map(h => h.trim().toLowerCase())

      const newData: ExperimentDataRow[] = []
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',')
        if (values.length < 7) continue

        const row: ExperimentDataRow = {
          id: i,
          deltaTj: parseFloat(values[headers.indexOf('δtj')] || values[0]) || 0,
          Tjmax: parseFloat(values[headers.indexOf('tjmax')] || values[1]) || 0,
          ton: parseFloat(values[headers.indexOf('ton')] || values[2]) || 0,
          I: parseFloat(values[headers.indexOf('i')] || values[3]) || 0,
          V: parseFloat(values[headers.indexOf('v')] || values[4]) || 0,
          D: parseFloat(values[headers.indexOf('d')] || values[5]) || 0,
          Nf: parseFloat(values[headers.indexOf('nf')] || values[6]) || 0,
        }
        newData.push(row)
      }

      if (newData.length > 0) {
        const syncedData = newData.map((row) => ({
          ...row,
          ton: fixedParams.ton.enabled ? fixedParams.ton.value : row.ton,
          I: fixedParams.I.enabled ? fixedParams.I.value : row.I,
          V: fixedParams.V.enabled ? fixedParams.V.value : row.V,
          D: fixedParams.D.enabled ? fixedParams.D.value : row.D,
        }))
        setExperimentData(syncedData)
      }
    }
    reader.readAsText(file)
    event.target.value = ''
  }, [fixedParams])

  const getScatterOption = () => {
    if (!fittingResult || !fittingResult.parameters) return {}

    const p = fittingResult.parameters
    const kB = 8.617e-5 // eV/K
    const R = 8.314     // J/(mol·K)
    const observed = experimentData.map(d => d.Nf)
    const predicted = experimentData.map((row) => {
      const dTj = row.deltaTj || 1
      const Tjmax_K = row.Tjmax + 273.15
      try {
        switch (selectedModel) {
          case 'coffin_manson':
            return (p.A ?? 1) * Math.pow(dTj, p.alpha ?? -5)
          case 'coffin_manson_arrhenius': {
            const Tm_K = row.Tjmax - dTj / 2 + 273.15
            return (p.A ?? 1) * Math.pow(dTj, p.alpha ?? -5) * Math.exp((p.Ea ?? 0.8) / (kB * Tm_K))
          }
          case 'norris_landzberg': {
            const f = 1 / (2 * (row.ton || 1))
            return (p.A ?? 1) * Math.pow(dTj, p.alpha ?? -5) * Math.pow(f, p.beta ?? -0.33) * Math.exp((p.Ea ?? 0.8) / (kB * Tjmax_K))
          }
          case 'lesit': {
            const Tmin_K = row.Tjmax - dTj + 273.15
            const Q_J = (p.Q ?? 0.8) * 96485  // eV → J/mol
            return (p.A ?? 1) * Math.pow(dTj, p.alpha ?? -5) * Math.exp(Q_J / (R * Tmin_K))
          }
          case 'cips2008':
          default: {
            const K = p.K ?? 1e10
            const b1 = p['β1'] ?? -4.5, b2 = p['β2'] ?? 1500, b3 = p['β3'] ?? -0.5
            return Math.exp(Math.log(K) + b1 * Math.log(dTj) + b2 / Tjmax_K + b3 * Math.log(row.ton || 1))
          }
        }
      } catch { return 0 }
    })

    const allValues = [...observed, ...predicted].filter(v => v > 0 && isFinite(v))
    if (allValues.length === 0) return {}

    const minVal = Math.min(...allValues) * 0.5
    const maxVal = Math.max(...allValues) * 2

    return {
      title: {
        text: '观测值 vs 预测值',
        left: 'center',
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: { data: number[] }) => {
          const obs = params.data?.[0] ?? 0
          const pred = params.data?.[1] ?? 0
          return `观测: ${obs.toExponential(2)}<br/>预测: ${pred.toExponential(2)}`
        },
      },
      xAxis: {
        type: 'log',
        name: '观测值 (Nf)',
        min: minVal,
        max: maxVal,
      },
      yAxis: {
        type: 'log',
        name: '预测值 (Nf)',
        min: minVal,
        max: maxVal,
      },
      series: [
        {
          type: 'scatter',
          data: observed.map((obs, i) => [obs, predicted[i]]),
          symbolSize: 10,
          itemStyle: { color: '#1976d2' },
        },
        {
          type: 'line',
          data: [[minVal, minVal], [maxVal, maxVal]],
          lineStyle: { type: 'dashed', color: '#999' },
          symbol: 'none',
        },
      ],
    }
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          参数拟合 / Parameter Fitting
        </Typography>
        <Typography variant="body2" color="text.secondary">
          基于功率循环试验数据拟合寿命模型参数
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Left Panel - Data Input */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">试验数据 / Experiment Data</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<UploadIcon />}
                  component="label"
                >
                  导入CSV
                  <input
                    type="file"
                    accept=".csv"
                    hidden
                    onChange={handleImportCSV}
                  />
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<AddIcon />}
                  onClick={addDataRow}
                >
                  添加行
                </Button>
              </Box>
            </Box>

            <TableContainer sx={{ maxHeight: 400 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    {visibleCols.map((col) => (
                      <TableCell key={col} sx={{ minWidth: COLUMN_LABELS[col].width }}>
                        <Box>
                          <Typography variant="caption" fontWeight="bold">{COLUMN_LABELS[col].header}</Typography>
                          <Typography variant="caption" display="block" color="text.secondary">{COLUMN_LABELS[col].sub}</Typography>
                        </Box>
                      </TableCell>
                    ))}
                    <TableCell sx={{ minWidth: 100 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">Nf (cycles)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">失效循环次数</Typography>
                      </Box>
                    </TableCell>
                    <TableCell>操作</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {experimentData.map((row) => (
                    <TableRow key={row.id} hover>
                      {visibleCols.map((col) => {
                        const fpKey = col as keyof FixedParamConfig
                        const fp = fixedParams[fpKey]
                        const isFixed = (availableFixed as readonly string[]).includes(col) && fp?.enabled
                        return (
                          <TableCell key={col}>
                            <TextField
                              size="small"
                              type="number"
                              value={isFixed ? fp.value : row[col as keyof ExperimentDataRow]}
                              onChange={(e) => updateDataRow(row.id, col as keyof ExperimentDataRow, parseFloat(e.target.value) || 0)}
                              disabled={!!isFixed}
                              sx={{ width: COLUMN_LABELS[col].width - 10 }}
                            />
                          </TableCell>
                        )
                      })}
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.Nf}
                          onChange={(e) => updateDataRow(row.id, 'Nf', parseFloat(e.target.value) || 0)}
                          sx={{ width: 100 }}
                        />
                      </TableCell>
                      <TableCell>
                        <IconButton
                          size="small"
                          onClick={() => removeDataRow(row.id)}
                          disabled={experimentData.length <= 1}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {error && (
              <Alert severity="error" sx={{ mt: 2 }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            {/* 固定参数配置 - only show when model has fixable params */}
            {availableFixed.length > 0 && (
              <Paper sx={{ p: 2, mt: 2, bgcolor: 'action.hover' }}>
                <Typography variant="subtitle2" gutterBottom>
                  固定参数配置 / Fixed Parameters
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                  默认拟合全部参数；可按需固定部分参数以减少所需数据点
                </Typography>
                <Grid container spacing={2}>
                  {availableFixed.map((param) => (
                    <Grid item xs={6} sm={3} key={param}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Checkbox
                          checked={fixedParams[param].enabled}
                          onChange={() => toggleFixedParam(param)}
                          size="small"
                        />
                        <TextField
                          size="small"
                          label={FIXED_PARAM_LABELS[param]}
                          value={fixedParams[param].value}
                          onChange={(e) => updateFixedParamValue(param, parseFloat(e.target.value) || 0)}
                          disabled={!fixedParams[param].enabled}
                          sx={{ flex: 1, minWidth: 80 }}
                        />
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            )}

            <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
              <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel>选择模型</InputLabel>
                <Select
                  value={selectedModel}
                  label="选择模型"
                  onChange={(e) => setSelectedModel(e.target.value as LifetimeModelType)}
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Button
                variant="contained"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                onClick={handleFit}
                disabled={loading || experimentData.length < getMinDataPoints()}
              >
                {loading ? '拟合中...' : '开始拟合'}
              </Button>
            </Box>

            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                数据要求 / Data Requirements:
              </Typography>
              <Typography variant="body2" component="div">
                • 当前需要至少 <strong>{getMinDataPoints()}</strong> 组试验数据（已约束/简化 {getReducedParamCount()} 个参数）
              </Typography>
              <Typography variant="body2" component="div">
                • 当前模型: {currentCfg.formula}
              </Typography>
              <Typography variant="body2" component="div">
                • 所需变量: {visibleCols.map(c => COLUMN_LABELS[c].header).join(', ')}, Nf
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                变量说明: ΔTj=温度摆幅(K), Tjmax=最高结温(°C), ton=加热时间(s), Nf=失效循环次数
              </Typography>
            </Alert>
          </Paper>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              拟合结果 / Fitting Results
            </Typography>

            {fittingResult ? (
              <Box>
                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={`R² = ${(fittingResult.r_squared ?? 0).toFixed(4)}`}
                    color={(fittingResult.r_squared ?? 0) > 0.9 ? 'success' : (fittingResult.r_squared ?? 0) > 0.7 ? 'warning' : 'error'}
                    sx={{ mr: 1 }}
                  />
                  <Chip
                    label={`RMSE = ${(fittingResult.rmse ?? 0).toExponential(2)}`}
                    color="primary"
                  />
                </Box>

                {/* 显示自动固定的参数 */}
                {fittingResult.auto_fixed_info && fittingResult.auto_fixed_info.length > 0 && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="caption" component="div">
                      <strong>自动固定参数:</strong>
                    </Typography>
                    <Typography variant="caption" component="div">
                      {fittingResult.auto_fixed_info.join(', ')}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                      固定参数项已耦合到 K 中，预测时使用当前 K 即可
                    </Typography>
                  </Alert>
                )}

                <Divider sx={{ my: 2 }} />

                <Typography variant="subtitle2" gutterBottom>
                  拟合参数:
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>参数</TableCell>
                        <TableCell>值</TableCell>
                        <TableCell>95% CI</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {currentCfg.resultParams.map((name) => {
                        const value = fittingResult.parameters?.[name]
                        if (value === undefined) return null
                        const ci = fittingResult.confidence_intervals?.[name]
                        const isFixed = fittingResult.fixed_params && name in fittingResult.fixed_params
                        const display = PARAM_DISPLAY[name] || { label: name, unit: '' }
                        const isScale = name === 'K' || name === 'A'
                        return (
                          <TableRow key={name} sx={isScale ? { bgcolor: 'success.light' } : isFixed ? { bgcolor: 'action.selected' } : {}}>
                            <TableCell>
                              {isScale ? <strong>{display.label}</strong> : display.label}
                              {display.unit && ` (${display.unit})`}
                            </TableCell>
                            <TableCell>
                              {isScale
                                ? <strong>{value.toExponential(4)}</strong>
                                : value.toFixed(4)}
                            </TableCell>
                            <TableCell>
                              {isFixed
                                ? '固定'
                                : ci && ci[0] != null && ci[1] != null
                                  ? isScale
                                    ? `[${(ci[0] as number).toExponential(2)}, ${(ci[1] as number).toExponential(2)}]`
                                    : `[${(ci[0] as number).toFixed(4)}, ${(ci[1] as number).toFixed(4)}]`
                                  : '-'}
                            </TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* 使用说明 */}
                <Alert severity="success" sx={{ mt: 2 }}>
                  <Typography variant="caption">
                    <strong>拟合完成:</strong> 点击“保存参数”后可在寿命预测页面直接加载使用。
                    当前模型: {currentCfg.formula}
                  </Typography>
                </Alert>

                <Button
                  variant="contained"
                  fullWidth
                  color={saveSuccess ? 'success' : 'primary'}
                  startIcon={<SaveIcon />}
                  onClick={handleSaveParams}
                  sx={{ mt: 2 }}
                >
                  {saveSuccess ? '保存成功!' : '保存参数'}
                </Button>
              </Box>
            ) : (
              <Alert severity="info">
                <Typography variant="body2" gutterBottom>
                  需要至少 <strong>{getMinDataPoints()}</strong> 组试验数据
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  当前: {experimentData.length} 组 | 模型: {currentCfg.formula}
                </Typography>
              </Alert>
            )}
          </Paper>

          {/* Saved Parameters */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              已保存的参数 / Saved Parameters
            </Typography>
            {Object.keys(savedParams).length > 0 ? (
              Object.entries(savedParams).map(([model, params]) => {
                const p = params as Record<string, number>
                const cfg = MODEL_FIT_CFG[model]
                const displayParams = cfg ? cfg.resultParams : Object.keys(p)
                const modelLabel = MODEL_OPTIONS.find(o => o.value === model)?.label || model
                return (
                  <Card key={model} sx={{ mb: 1 }}>
                    <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="subtitle2">{modelLabel}</Typography>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleLoadParams(model)}
                        >
                          加载到预测
                        </Button>
                      </Box>
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="caption" color="text.secondary" component="div">
                          {displayParams.map((pn) => {
                            const disp = PARAM_DISPLAY[pn] || { label: pn, unit: '' }
                            const v = p[pn]
                            const formatted = v == null ? 'N/A'
                              : (pn === 'K' || pn === 'A') ? v.toExponential(2)
                              : v.toFixed(3)
                            return `${disp.label}: ${formatted}`
                          }).join(' | ')}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                )
              })
            ) : (
              <Typography variant="body2" color="text.secondary">
                暂无保存的参数
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Charts */}
        {fittingResult && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                拟合效果 / Fit Quality
              </Typography>
              <ReactECharts option={getScatterOption()} style={{ height: 400 }} />
            </Paper>
          </Grid>
        )}

        {/* Experiment Methodology Guide */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mt: 2, bgcolor: 'grey.50' }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ScienceIcon color="primary" />
              功率循环试验方案指南 / Power Cycling Experiment Guide
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              基于{MODEL_OPTIONS.find(o => o.value === selectedModel)?.label || selectedModel}模型的参数拟合试验设计建议
            </Typography>

            <Grid container spacing={3}>
              {/* Model Formula */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      模型公式 / Model Formula
                    </Typography>
                    <Box sx={{
                      p: 2,
                      bgcolor: 'background.paper',
                      borderRadius: 1,
                      fontFamily: 'monospace',
                      fontSize: '0.85rem',
                      textAlign: 'center',
                      border: 1,
                      borderColor: 'divider',
                    }}>
                      {currentCfg.formula}
                    </Box>
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="caption">
                        <strong>说明:</strong> 电压V输入实际阻断电压等级（如600V、1200V、1700V）
                      </Typography>
                    </Alert>
                  </CardContent>
                </Card>
              </Grid>

              {/* Parameter Ranges */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      推荐参数范围 / Recommended Ranges (论文验证范围)
                    </Typography>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>参数</TableCell>
                          <TableCell>范围</TableCell>
                          <TableCell>说明</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>ΔTj (温度摆幅)</TableCell>
                          <TableCell>60 - 150 K</TableCell>
                          <TableCell>关键变量，需多组不同值</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Tjmax (最高结温)</TableCell>
                          <TableCell>125 - 250 °C</TableCell>
                          <TableCell>关键变量，需多组不同值</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>ton (加热时间)</TableCell>
                          <TableCell>1 - 60 s</TableCell>
                          <TableCell>短脉冲影响键合线，长脉冲影响焊料</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>I (负载电流)</TableCell>
                          <TableCell>器件额定值附近</TableCell>
                          <TableCell>同器件通常固定</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>V (阻断电压)</TableCell>
                          <TableCell>600 - 1700 V</TableCell>
                          <TableCell>芯片电压等级</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>D (键合线直径)</TableCell>
                          <TableCell>100 - 400 μm</TableCell>
                          <TableCell>同器件固定</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </Grid>

              {/* Experiment Design */}
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      试验设计方案 / Experiment Design
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                          1. 基础试验（最少7组）
                        </Typography>
                        <Typography variant="body2" component="div" color="text.secondary">
                          • 固定ton、I、V、D（同器件）<br/>
                          • 变化ΔTj：至少3-4个不同值<br/>
                          • 变化Tjmax：至少3-4个不同值<br/>
                          • 建议使用正交设计覆盖组合
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                          2. 扩展试验（可选）
                        </Typography>
                        <Typography variant="body2" component="div" color="text.secondary">
                          • 不同ton值测试时间影响<br/>
                          • 不同电流等级测试<br/>
                          • 不同电压等级器件对比<br/>
                          • 不同键合线直径器件对比
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                          3. 数据质量要求
                        </Typography>
                        <Typography variant="body2" component="div" color="text.secondary">
                          • 每组条件建议3个以上样本<br/>
                          • 失效判据需一致<br/>
                          • 记录完整的温度曲线<br/>
                          • R² &gt; 0.9 表示拟合良好
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Reference Values */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      CIPS 2008 论文参考值 / Reference Values
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                      基于多个IGBT模块厂商（Infineon, Semikron, Mitsubishi等）的测试数据拟合结果
                    </Typography>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>参数</TableCell>
                          <TableCell>参考值</TableCell>
                          <TableCell>物理意义</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>β1 (ΔTj指数)</TableCell>
                          <TableCell>-4.423</TableCell>
                          <TableCell>温度摆幅影响（负相关）</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>β2 (Tjmax系数)</TableCell>
                          <TableCell>1285</TableCell>
                          <TableCell>Arrhenius激活能相关</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>β3 (ton指数)</TableCell>
                          <TableCell>-0.462</TableCell>
                          <TableCell>加热时间影响</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>β4 (电流指数)</TableCell>
                          <TableCell>-0.716</TableCell>
                          <TableCell>负载电流/芯片面积影响</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>β5 (电压指数)</TableCell>
                          <TableCell>-0.761</TableCell>
                          <TableCell>阻断电压/芯片厚度影响</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>β6 (直径指数)</TableCell>
                          <TableCell>-0.5</TableCell>
                          <TableCell>键合线直径影响</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </Grid>

              {/* Tips */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      拟合优化建议 / Fitting Tips
                    </Typography>
                    <Box component="ul" sx={{ pl: 2, m: 0 }}>
                      <Typography component="li" variant="body2" color="text.secondary">
                        <strong>变量独立性:</strong> ΔTj和Tjmax应独立变化，避免完全相关
                      </Typography>
                      <Typography component="li" variant="body2" color="text.secondary">
                        <strong>范围覆盖:</strong> 测试条件应覆盖实际应用场景
                      </Typography>
                      <Typography component="li" variant="body2" color="text.secondary">
                        <strong>固定参数:</strong> 同款器件的D、ton、I、V通常固定，可减少拟合参数
                      </Typography>
                      <Typography component="li" variant="body2" color="text.secondary">
                        <strong>置信区间:</strong> 置信区间过宽表示数据不足或变量变化不够
                      </Typography>
                      <Typography component="li" variant="body2" color="text.secondary">
                        <strong>外推风险:</strong> 避免在测试范围外进行寿命预测
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Failure Criteria and Measurement */}
              <Grid item xs={12}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      失效判据与测量方法 / Failure Criteria &amp; Measurement
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                          失效判据 (Failure Criteria)
                        </Typography>
                        <Typography variant="body2" component="div" color="text.secondary">
                          • V<sub>CE(sat)</sub> 增加≥5% (初始值)<br/>
                          • 热阻 R<sub>th(j-c)</sub> 增加≥20%<br/>
                          • 器件开路/短路失效
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" fontWeight="bold" gutterBottom>
                          结温测量方法 (Junction Temperature Measurement)
                        </Typography>
                        <Typography variant="body2" component="div" color="text.secondary">
                          • 使用TSEP（温度敏感电参数）方法<br/>
                          • 典型TSEP: V<sub>CE</sub> @ 小电流 (~2 mV/K)<br/>
                          • 需要进行温度标定
                        </Typography>
                      </Grid>
                    </Grid>
                    <Alert severity="warning" sx={{ mt: 2 }}>
                      <Typography variant="caption">
                        <strong>注意:</strong> 本模型主要适用于键合线脱落（Bond Wire Lift-off）主导的失效模式。对于焊料层疲劳主导的失效，可能需要使用其他模型或调整参数。
                      </Typography>
                    </Alert>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Box sx={{ mt: 2, textAlign: 'right' }}>
              <Typography variant="caption" color="text.secondary">
                参考: Bayerer et al., "Model for Power Cycling lifetime of IGBT Modules - various factors influencing lifetime", CIPS 2008
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default ParameterFitting
