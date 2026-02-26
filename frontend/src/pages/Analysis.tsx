import React, { useState, useCallback } from 'react'
import {
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Divider,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material'
import {
  Analytics,
  Speed,
  Assessment,
  Timeline,
  ShowChart,
} from '@mui/icons-material'
import { SensitivityTornado } from '@/components/Visualization/SensitivityTornado'
import { SensitivityHeatmap } from '@/components/Visualization/SensitivityHeatmap'
import { LifetimeCurve } from '@/components/Visualization/LifetimeCurve'
import { ModelSelector } from '@/components/Prediction/ModelSelector'
import apiService from '@/services/api'
import type {
  LifetimeModelType,
  LifetimeModelParams,
  SensitivityAnalysisResult,
  WeibullAnalysisResult,
} from '@/types'

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

// Sensitivity analysis parameter options
const SENSITIVITY_PARAMS = [
  { name: 'deltaT', label: '温度摆动', labelEn: 'ΔTj', unit: '°C', variation: { min: -30, max: 30 } },
  { name: 'Tjmax', label: '最高结温', labelEn: 'Tjmax', unit: '°C', variation: { min: -20, max: 30 } },
  { name: 'theating', label: '加热时间', labelEn: 'ton', unit: 's', variation: { min: -20, max: 50 } },
  { name: 'tcooling', label: '冷却时间', labelEn: 'tcool', unit: 's', variation: { min: -20, max: 50 } },
]

// Weibull sample data for demonstration
const WEIBULL_SAMPLE_DATA = {
  failures: [12500, 18300, 22100, 28400, 32500, 38900, 43200, 48600, 54200, 61800],
  censored: [65000, 70000],
}

export const Analysis: React.FC = () => {
  const [modelType, setModelType] = useState<LifetimeModelType>('cips2008')
  const [params] = useState<Record<string, number>>({})
  const [currentTab, setCurrentTab] = useState(0)
  const [compareModels, setCompareModels] = useState<LifetimeModelType[]>([])

  // Sensitivity analysis state
  const [sensitivityResult, setSensitivityResult] = useState<SensitivityAnalysisResult | null>(null)
  const [sensitivityLoading, setSensitivityLoading] = useState(false)
  const [, setSensitivityError] = useState<string | null>(null)

  // Weibull analysis state
  const [weibullResult, setWeibullResult] = useState<WeibullAnalysisResult | null>(null)
  const [weibullLoading, setWeibullLoading] = useState(false)
  const [weibullError, setWeibullError] = useState<string | null>(null)
  const [confidenceLevel, setConfidenceLevel] = useState(0.9)

  // Curve visualization state
  const [curveParams] = useState<Record<string, number>>({
    Tmax: 125,
    Tmin: 40,
    theating: 60,
  })

  const handleModelChange = useCallback((model: LifetimeModelType) => {
    setModelType(model)
    setSensitivityResult(null)
    setWeibullResult(null)
  }, [])

  const handleCompareModelsChange = (_: React.MouseEvent<HTMLElement>, models: string[]) => {
    setCompareModels(models as LifetimeModelType[])
  }

  const runSensitivityAnalysis = useCallback(async () => {
    setSensitivityLoading(true)
    setSensitivityError(null)

    try {
      const response = await apiService.performSensitivityAnalysis({
        modelType,
        baseParams: params,
        parametersToAnalyze: SENSITIVITY_PARAMS.map((p) => ({
          name: p.name,
          variation: p.variation,
          steps: 10,
        })),
      })

      if (response.success && response.data) {
        setSensitivityResult(response.data as SensitivityAnalysisResult)
      } else {
        setSensitivityError(response.error || '敏感性分析失败')
      }
    } catch (err) {
      setSensitivityError(err instanceof Error ? err.message : '网络错误')
    } finally {
      setSensitivityLoading(false)
    }
  }, [modelType, params])

  const runWeibullAnalysis = useCallback(async () => {
    setWeibullLoading(true)
    setWeibullError(null)

    try {
      const response = await apiService.performWeibullAnalysis({
        failures: WEIBULL_SAMPLE_DATA.failures,
        censored: WEIBULL_SAMPLE_DATA.censored,
        confidenceLevel,
      })

      if (response.success && response.data) {
        setWeibullResult(response.data as WeibullAnalysisResult)
      } else {
        setWeibullError(response.error || 'Weibull分析失败')
      }
    } catch (err) {
      setWeibullError(err instanceof Error ? err.message : '网络错误')
    } finally {
      setWeibullLoading(false)
    }
  }, [confidenceLevel])

  const getAvailableModels = (): string[] => {
    return ['coffin_manson', 'coffin_manson_arrhenius', 'norris_landzberg', 'cips2008', 'lesit'].filter(
      (m) => m !== modelType
    )
  }

  return (
    <Box>
      {/* Page Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          参数分析 / Parameter Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary">
          敏感性分析、Weibull分析和模型比较 / Sensitivity analysis, Weibull analysis, and model comparison
        </Typography>
      </Box>

      {/* Model Selection Bar */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <ModelSelector value={modelType} onChange={handleModelChange} />
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="subtitle2">比较模型 / Compare:</Typography>
              <ToggleButtonGroup
                value={compareModels}
                onChange={handleCompareModelsChange}
                size="small"
              >
                {getAvailableModels().map((model) => (
                  <ToggleButton key={model} value={model}>
                    {model === 'coffin_manson'
                      ? 'C-M'
                      : model === 'coffin_manson_arrhenius'
                        ? 'CMA'
                        : model === 'norris_landzberg'
                          ? 'N-L'
                          : model === 'cips2008'
                            ? 'CIPS'
                            : 'LESIT'}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Main Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={currentTab} onChange={(_, newValue) => setCurrentTab(newValue)}>
          <Tab
            icon={<Speed />}
            label="敏感性分析 / Sensitivity"
            iconPosition="start"
          />
          <Tab
            icon={<Assessment />}
            label="Weibull分析 / Weibull"
            iconPosition="start"
          />
          <Tab
            icon={<ShowChart />}
            label="寿命曲线 / Curves"
            iconPosition="start"
          />
        </Tabs>
      </Paper>

      {/* Sensitivity Analysis Tab */}
      <TabPanel value={currentTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={3}>
            <Stack spacing={2}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    <Analytics sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
                    敏感性设置
                  </Typography>
                  <Alert severity="info" sx={{ mt: 1 }}>
                    <Typography variant="caption">
                      分析参数变化对寿命预测的影响程度
                    </Typography>
                  </Alert>
                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={
                      sensitivityLoading ? <CircularProgress size={16} /> : <Speed />
                    }
                    onClick={runSensitivityAnalysis}
                    disabled={sensitivityLoading}
                    sx={{ mt: 2 }}
                  >
                    运行分析
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardContent>
                  <Typography variant="caption" color="text.secondary">
                    分析参数 / Parameters
                  </Typography>
                  <Box sx={{ mt: 1 }}>
                    {SENSITIVITY_PARAMS.map((param) => (
                      <Box
                        key={param.name}
                        sx={{ display: 'flex', justifyContent: 'space-between', py: 0.5 }}
                      >
                        <Typography variant="body2">{param.label}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {param.variation.min > 0 ? '+' : ''}{param.variation.min}% ~ +{param.variation.max}%
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          <Grid item xs={12} md={9}>
            <SensitivityTornado
              data={sensitivityResult}
              loading={sensitivityLoading}
            />
          </Grid>
        </Grid>
      </TabPanel>

      {/* Weibull Analysis Tab */}
      <TabPanel value={currentTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={3}>
            <Stack spacing={2}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    <Assessment sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
                    Weibull设置
                  </Typography>

                  <FormControl fullWidth size="small" sx={{ mt: 2 }}>
                    <InputLabel>置信度 / Confidence</InputLabel>
                    <Select
                      value={confidenceLevel}
                      label="置信度 / Confidence"
                      onChange={(e) => setConfidenceLevel(e.target.value as number)}
                    >
                      <MenuItem value={0.8}>80%</MenuItem>
                      <MenuItem value={0.9}>90%</MenuItem>
                      <MenuItem value={0.95}>95%</MenuItem>
                      <MenuItem value={0.99}>99%</MenuItem>
                    </Select>
                  </FormControl>

                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="caption" display="block">
                      失效数据 / Failures:
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {WEIBULL_SAMPLE_DATA.failures.length} 个样本
                    </Typography>
                  </Alert>

                  <Button
                    fullWidth
                    variant="contained"
                    startIcon={weibullLoading ? <CircularProgress size={16} /> : <Assessment />}
                    onClick={runWeibullAnalysis}
                    disabled={weibullLoading}
                    sx={{ mt: 2 }}
                  >
                    运行分析
                  </Button>
                </CardContent>
              </Card>

              {weibullResult && (
                <Card>
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>
                      分析结果 / Results
                    </Typography>
                    <Divider sx={{ my: 1 }} />
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        形状参数 β: <strong>{weibullResult.params.shape.toFixed(3)}</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        尺度参数 η: <strong>{weibullResult.params.scale.toExponential(2)}</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        MTTF: <strong>{weibullResult.MTTF.toExponential(2)}</strong>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        中位寿命: <strong>{weibullResult.medianLife.toExponential(2)}</strong>
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Stack>
          </Grid>

          <Grid item xs={12} md={9}>
            {weibullError && (
              <Alert severity="error" sx={{ mb: 2 }}>{weibullError}</Alert>
            )}

            {!weibullResult && !weibullLoading && (
              <Paper
                sx={{
                  p: 6,
                  textAlign: 'center',
                  border: 2,
                  borderColor: 'divider',
                  borderStyle: 'dashed',
                }}
              >
                <Assessment sx={{ fontSize: 48, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  等待Weibull分析 / Awaiting Weibull Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  点击"运行分析"按钮开始计算
                </Typography>
              </Paper>
            )}

            {weibullLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress size={60} />
              </Box>
            )}

            {weibullResult && (
              <Grid container spacing={2}>
                {/* Weibull Probability Plot */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Weibull概率图 / Probability Plot
                    </Typography>
                    <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        概率图图表区域 / Probability Plot Chart Area
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>

                {/* Reliability Function */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      可靠度函数 / Reliability Function
                    </Typography>
                    <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        可靠度曲线图表区域 / Reliability Curve Chart Area
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>

                {/* Failure Rate */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      失效率 / Failure Rate
                    </Typography>
                    <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        失效率曲线图表区域 / Failure Rate Chart Area
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>

                {/* PDF */}
                <Grid item xs={12} md={6}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      概率密度 / PDF
                    </Typography>
                    <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        概率密度图表区域 / PDF Chart Area
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            )}
          </Grid>
        </Grid>
      </TabPanel>

      {/* Lifetime Curves Tab */}
      <TabPanel value={currentTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <LifetimeCurve
              modelType={modelType}
              params={params as unknown as LifetimeModelParams}
              baseParams={curveParams}
              compareModels={compareModels}
            />
          </Grid>
        </Grid>

        {/* 2D Sensitivity Heatmap */}
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              <Timeline sx={{ fontSize: 20, verticalAlign: 'middle', mr: 0.5 }} />
              双参数敏感性 / Two-Parameter Sensitivity
            </Typography>
            <SensitivityHeatmap
              modelType={modelType}
              params={params as unknown as LifetimeModelParams}
              baseParams={curveParams}
            />
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  )
}

export default Analysis
