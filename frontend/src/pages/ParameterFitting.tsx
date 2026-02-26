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
  FormControlLabel,
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
}

const MODEL_OPTIONS: { value: LifetimeModelType; label: string }[] = [
  { value: 'cips2008', label: 'CIPS 2008 (Bayerer)' },
  { value: 'coffin_manson', label: 'Coffin-Manson' },
  { value: 'norris_landzberg', label: 'Norris-Landzberg' },
  { value: 'lesit', label: 'LESIT' },
]

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
      ton: { enabled: true, value: 2 },
      I: { enabled: true, value: 100 },
      V: { enabled: true, value: 1200 },
      D: { enabled: true, value: 300 },
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

  // 计算最少需要的数据点数（7个参数 - 固定参数数量）
  const getMinDataPoints = () => {
    const fixedCount = Object.values(fixedParams).filter(p => p.enabled).length
    return Math.max(3, 7 - fixedCount)
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
      // 准备固定参数（仅包含启用的固定参数）
      const fixedParamsForApi: Record<string, number> = {}
      if (fixedParams.ton.enabled) fixedParamsForApi['β3'] = -0.462  // 使用典型值
      if (fixedParams.I.enabled) fixedParamsForApi['β4'] = -0.716
      if (fixedParams.V.enabled) fixedParamsForApi['β5'] = -0.761
      if (fixedParams.D.enabled) fixedParamsForApi['β6'] = -0.5

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

      const response = await fetch('/api/analysis/fitting/cips2008', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
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
  }, [experimentData, fixedParams])

  const handleSaveParams = useCallback(() => {
    if (!fittingResult) return

    const newSavedParams = {
      ...savedParams,
      [selectedModel]: fittingResult.parameters,
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
        setExperimentData(newData)
      }
    }
    reader.readAsText(file)
    event.target.value = ''
  }, [])

  const getScatterOption = () => {
    if (!fittingResult || !fittingResult.parameters) return {}

    const params = fittingResult.parameters
    const observed = experimentData.map(d => d.Nf)
    const predicted = experimentData.map((row) => {
      // Simple prediction using fitted parameters with null checks
      const K = params.K ?? 1e10
      const beta1 = params['β1'] ?? params.beta1 ?? -4.5
      const beta2 = params['β2'] ?? params.beta2 ?? 1500
      const beta3 = params['β3'] ?? params.beta3 ?? -0.5

      return Math.exp(
        Math.log(K) +
        beta1 * Math.log(row.deltaTj || 1) +
        beta2 / (row.Tjmax + 273.15) +
        beta3 * Math.log(row.ton || 1)
      )
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
                    <TableCell sx={{ minWidth: 90 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">ΔTj (K)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">温度摆幅</Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ minWidth: 90 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">Tjmax (°C)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">最高结温</Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ minWidth: 80 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">ton (s)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">加热时间</Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ minWidth: 80 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">I (A)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">负载电流</Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ minWidth: 90 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">V (V)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">阻断电压</Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ minWidth: 80 }}>
                      <Box>
                        <Typography variant="caption" fontWeight="bold">D (μm)</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">键合线直径</Typography>
                      </Box>
                    </TableCell>
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
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.deltaTj}
                          onChange={(e) => updateDataRow(row.id, 'deltaTj', parseFloat(e.target.value) || 0)}
                          sx={{ width: 80 }}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.Tjmax}
                          onChange={(e) => updateDataRow(row.id, 'Tjmax', parseFloat(e.target.value) || 0)}
                          sx={{ width: 80 }}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.ton}
                          onChange={(e) => updateDataRow(row.id, 'ton', parseFloat(e.target.value) || 0)}
                          sx={{ width: 70 }}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.I}
                          onChange={(e) => updateDataRow(row.id, 'I', parseFloat(e.target.value) || 0)}
                          sx={{ width: 80 }}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.V}
                          onChange={(e) => updateDataRow(row.id, 'V', parseFloat(e.target.value) || 0)}
                          sx={{ width: 90 }}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={row.D}
                          onChange={(e) => updateDataRow(row.id, 'D', parseFloat(e.target.value) || 0)}
                          sx={{ width: 80 }}
                        />
                      </TableCell>
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

            {/* 固定参数配置 */}
            <Paper sx={{ p: 2, mt: 2, bgcolor: 'action.hover' }}>
              <Typography variant="subtitle2" gutterBottom>
                固定参数配置 / Fixed Parameters
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                勾选固定参数可减少所需数据点数量（同款器件通常固定: ton, I, V, D）
              </Typography>
              <Grid container spacing={2}>
                {(['ton', 'I', 'V', 'D'] as const).map((param) => (
                  <Grid item xs={6} sm={3} key={param}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Checkbox
                        checked={fixedParams[param].enabled}
                        onChange={() => toggleFixedParam(param)}
                        size="small"
                      />
                      <TextField
                        size="small"
                        label={param === 'ton' ? 'ton (s)' : param === 'I' ? 'I (A)' : param === 'V' ? 'V (V)' : 'D (μm)'}
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
                • 当前需要至少 <strong>{getMinDataPoints()}</strong> 组试验数据（已固定 {Object.values(fixedParams).filter(p => p.enabled).length} 个参数）
              </Typography>
              <Typography variant="body2" component="div">
                • 主要变量: ΔTj (温度摆幅) 和 Tjmax (最高结温) 应有不同值
              </Typography>
              <Typography variant="body2" component="div" color="text.secondary">
                • 电压V输入芯片阻断电压等级（如600V、1200V、1700V）
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                变量说明: ΔTj=温度摆幅(K), Tjmax=最高结温(°C), ton=加热时间(s), I=电流(A), V=阻断电压(V), D=键合线直径(μm), Nf=失效循环次数
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
                      固定参数已合并到 K_eff 中，预测时使用 K_eff 即可
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
                      {/* K_eff - 用于预测的有效K值 */}
                      {fittingResult.parameters?.K_eff != null && (
                        <TableRow sx={{ bgcolor: 'success.light' }}>
                          <TableCell><strong>K_eff</strong></TableCell>
                          <TableCell><strong>{(fittingResult.parameters.K_eff).toExponential(4)}</strong></TableCell>
                          <TableCell>
                            {fittingResult.confidence_intervals?.K_eff && fittingResult.confidence_intervals.K_eff[0] != null && fittingResult.confidence_intervals.K_eff[1] != null
                              ? `[${(fittingResult.confidence_intervals.K_eff[0] as number).toExponential(2)}, ${(fittingResult.confidence_intervals.K_eff[1] as number).toExponential(2)}]`
                              : '-'}
                          </TableCell>
                        </TableRow>
                      )}
                      {/* K 原始值 */}
                      {fittingResult.parameters?.K != null && (
                        <TableRow>
                          <TableCell>K (原始)</TableCell>
                          <TableCell>{(fittingResult.parameters.K).toExponential(4)}</TableCell>
                          <TableCell>
                            {fittingResult.confidence_intervals?.K && fittingResult.confidence_intervals.K[0] != null && fittingResult.confidence_intervals.K[1] != null
                              ? `[${(fittingResult.confidence_intervals.K[0] as number).toExponential(2)}, ${(fittingResult.confidence_intervals.K[1] as number).toExponential(2)}]`
                              : '-'}
                          </TableCell>
                        </TableRow>
                      )}
                      {/* β 参数 */}
                      {['β1', 'β2', 'β3', 'β4', 'β5', 'β6'].map((name) => {
                        const value = fittingResult.parameters?.[name]
                        if (value === undefined) return null
                        const ci = fittingResult.confidence_intervals?.[name]
                        const isFixed = fittingResult.fixed_params && name in fittingResult.fixed_params
                        return (
                          <TableRow key={name} sx={isFixed ? { bgcolor: 'action.selected' } : {}}>
                            <TableCell>{name}</TableCell>
                            <TableCell>{value.toFixed(4)}</TableCell>
                            <TableCell>
                              {isFixed
                                ? '固定'
                                : ci && ci[0] != null && ci[1] != null
                                  ? `[${ci[0].toFixed(4)}, ${ci[1].toFixed(4)}]`
                                  : '-'}
                            </TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* 使用说明 */}
                {fittingResult.parameters?.K_eff != null && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    <Typography variant="caption">
                      <strong>使用 K_eff 进行预测:</strong> 在寿命预测页面输入 K_eff 和拟合的 β1、β2 值，
                      以及固定的 ton、I、V、D 值即可。
                    </Typography>
                  </Alert>
                )}

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
                  当前: {experimentData.length} 组 | 固定参数: {Object.values(fixedParams).filter(p => p.enabled).length} 个
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                  变量: ΔTj, Tjmax, ton, I, V, D, Nf
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
                return (
                  <Card key={model} sx={{ mb: 1 }}>
                    <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="subtitle2">{model}</Typography>
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
                          K: {p.K?.toExponential(2) || 'N/A'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" component="div">
                          β1: {p['β1']?.toFixed(3) || p.beta1?.toFixed(3) || 'N/A'} |
                          β2: {p['β2']?.toFixed(1) || p.beta2?.toFixed(1) || 'N/A'} |
                          β3: {p['β3']?.toFixed(3) || p.beta3?.toFixed(3) || 'N/A'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" component="div">
                          β4: {p['β4']?.toFixed(3) || p.beta4?.toFixed(3) || 'N/A'} |
                          β5: {p['β5']?.toFixed(3) || p.beta5?.toFixed(3) || 'N/A'} |
                          β6: {p['β6']?.toFixed(3) || p.beta6?.toFixed(3) || 'N/A'}
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
              基于CIPS 2008 (Bayerer) 模型的参数拟合试验设计建议
            </Typography>

            <Grid container spacing={3}>
              {/* Model Formula */}
              <Grid item xs={12} md={6}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="primary" gutterBottom>
                      CIPS 2008 模型公式 / Model Formula
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
                      N<sub>f</sub> = K × (ΔT<sub>j</sub>)<sup>β1</sup> × exp(β2/T<sub>j,max</sub>) × t<sub>on</sub><sup>β3</sup> × I<sup>β4</sup> × V<sup>β5</sup> × D<sup>β6</sup>
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
