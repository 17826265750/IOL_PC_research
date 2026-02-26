import React, { useState, useCallback, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Stack,
  Chip,
} from '@mui/material'
import {
  Calculate,
  Refresh,
  History,
  TrendingUp,
  Save,
  CloudDownload,
} from '@mui/icons-material'
import { ModelSelector } from '@/components/Prediction/ModelSelector'
import { ParameterInput } from '@/components/Prediction/ParameterInput'
import { ResultDisplay } from '@/components/Prediction/ResultDisplay'
import { LifetimeCurve } from '@/components/Visualization/LifetimeCurve'
import apiService from '@/services/api'
import type { LifetimeModelType, LifetimeModelParams, PredictionResult, ExportFormat } from '@/types'

const FITTED_PARAMS_KEY = 'cips_fitted_parameters'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box>{children}</Box>}
    </div>
  )
}

export const Prediction: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [modelType, setModelType] = useState<LifetimeModelType>('cips2008')
  const [params, setParams] = useState<Record<string, number>>({})
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentTab, setCurrentTab] = useState(0)
  const [usingFittedParams, setUsingFittedParams] = useState(false)

  // Load fitted params for current model
  const handleLoadFittedParams = useCallback((modelToLoad?: LifetimeModelType | React.MouseEvent) => {
    const targetModel = (typeof modelToLoad === 'string' ? modelToLoad : modelType) as LifetimeModelType
    const saved = localStorage.getItem(FITTED_PARAMS_KEY)
    if (saved) {
      const allParams = JSON.parse(saved)
      const fittedParams = allParams[targetModel]
      if (fittedParams) {
        // Convert Greek letter param names to English (β1 -> beta1)
        const convertedParams: Record<string, number> = {}
        for (const [key, value] of Object.entries(fittedParams as Record<string, unknown>)) {
          const newKey = key.replace('β', 'beta')
          convertedParams[newKey] = typeof value === 'number' ? value : 0
        }
        if (typeof modelToLoad === 'string' && modelToLoad !== modelType) {
          setModelType(modelToLoad)
        }
        setParams(prev => ({ ...prev, ...convertedParams }))
        setUsingFittedParams(true)
      }
    }
  }, [modelType])

  // Handle navigation state
  useEffect(() => {
    const state = location.state as { loadFittedParams?: boolean; model?: LifetimeModelType } | null
    if (state?.loadFittedParams && state.model) {
      handleLoadFittedParams(state.model)
      // Clear state to prevent reloading on subsequent renders
      navigate(location.pathname, { replace: true, state: {} })
    }
  }, [location.state, location.pathname, navigate, handleLoadFittedParams])

  // Listen for fitted params from ParameterFitting page (fallback for same-page events if any)
  useEffect(() => {
    const handleLoadFittedParamsEvent = (event: CustomEvent) => {
      const { model, params: fittedParams } = event.detail
      if (model === modelType) {
        // Convert Greek letter param names to English (β1 -> beta1)
        const convertedParams: Record<string, number> = {}
        for (const [key, value] of Object.entries(fittedParams as Record<string, unknown>)) {
          const newKey = key.replace('β', 'beta')
          convertedParams[newKey] = typeof value === 'number' ? value : 0
        }
        setParams(prev => ({ ...prev, ...convertedParams }))
        setUsingFittedParams(true)
      }
    }

    window.addEventListener('loadFittedParams', handleLoadFittedParamsEvent as EventListener)
    return () => {
      window.removeEventListener('loadFittedParams', handleLoadFittedParamsEvent as EventListener)
    }
  }, [modelType])

  const handleModelChange = useCallback((model: LifetimeModelType) => {
    setModelType(model)
    setResult(null)
    setError(null)
    setUsingFittedParams(false)
  }, [])

  const handleParamsChange = useCallback((newParams: Record<string, number>) => {
    setParams(newParams)
    setUsingFittedParams(false)
  }, [])

  const handleCalculate = useCallback(async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Transform params to API format
      const apiParams: Record<string, unknown> = { ...params }
      const cycles = [
        {
          Tmax: params.Tmax || 125,
          Tmin: params.Tmin || 40,
          theating: params.theating || 60,
          tcooling: params.tcooling || 60,
        },
      ]

      const response = await apiService.predictLifetime({
        modelType,
        params: apiParams,
        cycles,
      })

      if (response.success && response.data) {
        setResult(response.data as PredictionResult)
        setCurrentTab(1)
      } else {
        setError(response.error || '预测失败 / Prediction failed')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '网络错误 / Network error')
    } finally {
      setLoading(false)
    }
  }, [modelType, params])

  const handleReset = useCallback(() => {
    setParams({})
    setResult(null)
    setError(null)
    setCurrentTab(0)
  }, [])

  const handleExport = useCallback(async (format: ExportFormat) => {
    if (!result) return

    try {
      const blob = await apiService.exportData({
        type: 'prediction',
        id: result.modelType + '_' + Date.now(),
        format,
        includeCharts: true,
      }) as unknown as Blob

      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `lifetime_prediction_${Date.now()}.${format === 'xlsx' ? 'xlsx' : format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }, [result])

  return (
    <Box>
      {/* Page Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          寿命预测 / Lifetime Prediction
        </Typography>
        <Typography variant="body2" color="text.secondary">
          基于CIPS 2008标准的功率器件寿命预测模型 / Power device lifetime prediction based on CIPS 2008
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Left Panel - Input */}
        <Grid item xs={12} lg={4}>
          <Stack spacing={2}>
            {/* Model Selector */}
            <Paper sx={{ p: 2 }}>
              <ModelSelector value={modelType} onChange={handleModelChange} />
            </Paper>

            {/* Parameter Input */}
            <Paper sx={{ p: 2 }}>
              <ParameterInput
                modelType={modelType}
                values={params}
                onChange={handleParamsChange}
                disabled={loading}
              />
            </Paper>

            {/* Action Buttons */}
            <Paper sx={{ p: 2 }}>
              <Stack spacing={1}>
                <Stack direction="row" spacing={1}>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={loading ? <CircularProgress size={16} /> : <Calculate />}
                    onClick={handleCalculate}
                    disabled={loading}
                    size="large"
                  >
                    {loading ? '计算中... / Calculating...' : '计算 / Calculate'}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<Refresh />}
                  onClick={handleReset}
                  disabled={loading}
                >
                  重置 / Reset
                </Button>
                </Stack>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={<CloudDownload />}
                  onClick={handleLoadFittedParams}
                  disabled={loading}
                  size="small"
                >
                  加载拟合参数 / Load Fitted Params
                </Button>
                {usingFittedParams && (
                  <Chip
                    label="使用拟合参数"
                    color="success"
                    size="small"
                    sx={{ mt: 1 }}
                  />
                )}
              </Stack>
            </Paper>

            {/* Quick Info */}
            {error && (
              <Alert severity="error" onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            {!error && !result && (
              <Alert severity="info">
                <Typography variant="body2">
                  选择模型并设置参数后点击"计算"按钮开始预测
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Select a model and configure parameters, then click "Calculate"
                </Typography>
              </Alert>
            )}
          </Stack>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2, minHeight: 600 }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
              <Tabs value={currentTab} onChange={(_, newValue) => setCurrentTab(newValue)}>
                <Tab label="结果 / Results" />
                <Tab label="曲线分析 / Curves" disabled={!result} />
              </Tabs>
            </Box>

            <TabPanel value={currentTab} index={0}>
              <ResultDisplay
                result={result}
                loading={loading}
                error={error}
                onExport={handleExport}
              />
            </TabPanel>

            <TabPanel value={currentTab} index={1}>
              {result && (
                <LifetimeCurve
                  modelType={modelType}
                  params={params as unknown as LifetimeModelParams}
                  baseParams={params}
                />
              )}
            </TabPanel>
          </Paper>
        </Grid>
      </Grid>

      {/* History / Recent Calculations Section */}
      {result && (
        <Paper sx={{ mt: 3, p: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <History sx={{ mr: 1 }} />
            <Typography variant="h6">快速分析 / Quick Analysis</Typography>
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  p: 2,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: 'action.hover',
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  <TrendingUp sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
                  寿命评估 / Lifetime Assessment
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  预测寿命: <strong>{(result.lifetimeHours / 8760).toFixed(2)} 年</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  循环次数: <strong>{result.predictedCycles.toExponential(2)}</strong>
                </Typography>
                {result.confidenceLower && result.confidenceUpper && (
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                    95%置信区间: {result.confidenceLower.toExponential(1)} -{' '}
                    {result.confidenceUpper.toExponential(1)}
                  </Typography>
                )}
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  p: 2,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  bgcolor: 'action.hover',
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  <Save sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
                  建议 / Recommendations
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {result.predictedCycles >= 100000
                    ? '当前工况下器件寿命表现优秀，可继续使用。'
                    : result.predictedCycles >= 20000
                      ? '当前工况下器件寿命表现良好，建议定期监测。'
                      : '当前工况下器件寿命较低，建议优化工作条件。'}
                </Typography>
                <Typography variant="caption" color="text.disabled" display="block" sx={{ mt: 1 }}>
                  {result.predictedCycles >= 100000
                    ? 'Device lifetime is excellent under current conditions.'
                    : result.predictedCycles >= 20000
                      ? 'Device lifetime is good. Regular monitoring recommended.'
                      : 'Device lifetime is low. Consider optimizing operating conditions.'}
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Box>
  )
}

export default Prediction
