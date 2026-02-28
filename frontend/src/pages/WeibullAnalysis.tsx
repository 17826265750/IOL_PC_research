/**
 * 功率模块寿命分析软件 - 威布尔可靠性分析页面
 * @author GSH
 */
import React, { useCallback, useMemo } from 'react'
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
  Button,
  Divider,
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import { apiService } from '@/services/api'
import { useWeibullStore } from '@/stores/useWeibullStore'
import {
  WeibullInput,
  WeibullFitResultDisplay,
  WeibullProbabilityPlot,
  WeibullReliabilityCurve,
  WeibullHazardCurve,
  WeibullBLifeTable,
} from '@/components/Weibull'

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
/*  Helper Functions                                                   */
/* ------------------------------------------------------------------ */

/**
 * Parse comma or newline separated numbers
 */
function parseNumbers(input: string): number[] {
  if (!input.trim()) return []
  return input
    .split(/[,\n\s]+/)
    .map((s) => parseFloat(s.trim()))
    .filter((n) => !isNaN(n) && isFinite(n))
}

/* ------------------------------------------------------------------ */
/*  WeibullAnalysis Page                                               */
/* ------------------------------------------------------------------ */
export const WeibullAnalysis: React.FC = () => {
  const {
    tab,
    loading,
    error,
    failureTimesInput,
    censoredTimesInput,
    confidenceLevel,
    fitMethod,
    fitResult,
    probabilityPlotData,
    reliabilityCurveData,
    hazardCurveData,
    customBLifes,
    customBLifeResults,
    patch,
    reset,
  } = useWeibullStore()

  const failureTimes = useMemo(() => parseNumbers(failureTimesInput), [failureTimesInput])
  const censoredTimes = useMemo(() => parseNumbers(censoredTimesInput), [censoredTimesInput])

  /* ---------- Run Analysis ---------- */
  const runAnalysis = useCallback(async () => {
    if (failureTimes.length < 3) {
      patch({ error: '请至少输入3个失效时间数据点' })
      return
    }

    patch({ loading: true, error: null })

    try {
      // 1. Fit Weibull distribution
      // Map 'ls' to 'rry' for backend API
      const apiMethod = fitMethod === 'ls' ? 'rry' : fitMethod
      const fitResponse = await apiService.fitWeibull({
        failure_times: failureTimes,
        censored_times: censoredTimes.length > 0 ? censoredTimes : undefined,
        confidence_level: parseFloat(confidenceLevel),
        method: apiMethod,
      })

      if (!fitResponse.success || !fitResponse.data) {
        patch({ loading: false, error: fitResponse.error || '拟合失败' })
        return
      }

      const fitData = fitResponse.data as any
      patch({ fitResult: fitData })

      // 2. Get probability plot data
      const plotResponse = await apiService.getWeibullProbabilityPlot({
        failure_times: failureTimes,
        censored_times: censoredTimes.length > 0 ? censoredTimes : undefined,
      })

      if (plotResponse.success && plotResponse.data) {
        patch({ probabilityPlotData: plotResponse.data as any })
      }

      // 3. Get curve data (PDF, CDF, reliability, hazard)
      const maxTime = Math.max(...failureTimes, ...censoredTimes, 0) * 1.5
      const curveResponse = await apiService.getWeibullCurve({
        shape: fitData.shape,
        scale: fitData.scale,
        t_min: 0,
        t_max: maxTime || 1000,
        num_points: 200,
      })

      if (curveResponse.success && curveResponse.data) {
        const curveData = curveResponse.data as any
        patch({
          reliabilityCurveData: curveData,
          hazardCurveData: curveData,
        })
      }

      // 4. Calculate custom B-lifes if any
      if (customBLifes.length > 0) {
        const bLifeResponse = await apiService.calculateWeibullBLife({
          shape: fitData.shape,
          scale: fitData.scale,
          percentiles: customBLifes,
        })

        if (bLifeResponse.success && bLifeResponse.data) {
          const bLifeData = bLifeResponse.data as any
          patch({ customBLifeResults: bLifeData.b_lifes || {} })
        }
      }

      patch({ loading: false, tab: 1 }) // Switch to results tab
    } catch (err) {
      const message = err instanceof Error ? err.message : '分析过程发生错误'
      patch({ loading: false, error: message })
    }
  }, [failureTimes, censoredTimes, confidenceLevel, fitMethod, customBLifes, patch])

  /* ---------- Add/Remove Custom B-Life ---------- */
  const handleAddBLife = useCallback(
    (percentile: number, value: number) => {
      patch({
        customBLifes: [...customBLifes, percentile],
        customBLifeResults: { ...customBLifeResults, [percentile]: value },
      })
    },
    [customBLifes, customBLifeResults, patch]
  )

  const handleRemoveBLife = useCallback(
    (percentile: number) => {
      const newBLifes = customBLifes.filter((p) => p !== percentile)
      const newResults = { ...customBLifeResults }
      delete newResults[percentile]
      patch({
        customBLifes: newBLifes,
        customBLifeResults: newResults,
      })
    },
    [customBLifes, customBLifeResults, patch]
  )

  /* ---------- Tab Change ---------- */
  const handleTabChange = useCallback(
    (_: React.SyntheticEvent, newValue: number) => {
      patch({ tab: newValue })
    },
    [patch]
  )

  /* ---------- Reset ---------- */
  const handleReset = useCallback(() => {
    reset()
  }, [reset])

  /* ---------- PDF/CDF Chart Option ---------- */
  const pdfCdfOption = useMemo(() => {
    if (!reliabilityCurveData) return null

    const times = reliabilityCurveData.times
    const pdfData = times.map((t, i) => [t, reliabilityCurveData.pdf[i]])
    const cdfData = times.map((t, i) => [t, reliabilityCurveData.cdf[i] * 100])

    return {
      title: {
        text: '概率密度函数 (PDF) 与 累积分布函数 (CDF)',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 600 },
      },
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        data: ['PDF', 'CDF'],
        top: 30,
      },
      grid: {
        left: 80,
        right: 60,
        top: 80,
        bottom: 60,
      },
      xAxis: {
        type: 'value',
        name: '时间 (小时/循环)',
        nameLocation: 'middle',
        nameGap: 30,
      },
      yAxis: [
        {
          type: 'value',
          name: 'PDF',
          nameLocation: 'middle',
          nameGap: 50,
          position: 'left',
        },
        {
          type: 'value',
          name: 'CDF (%)',
          nameLocation: 'middle',
          nameGap: 50,
          position: 'right',
          min: 0,
          max: 100,
        },
      ],
      series: [
        {
          name: 'PDF',
          type: 'line',
          data: pdfData,
          symbol: 'none',
          smooth: true,
          lineStyle: { color: '#2196f3', width: 2 },
        },
        {
          name: 'CDF',
          type: 'line',
          yAxisIndex: 1,
          data: cdfData,
          symbol: 'none',
          smooth: true,
          lineStyle: { color: '#4caf50', width: 2 },
        },
      ],
      toolbox: {
        feature: {
          saveAsImage: { title: '保存图片', pixelRatio: 2 },
        },
        right: 20,
        top: 10,
      },
    }
  }, [reliabilityCurveData])

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      <Stack spacing={3}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h4" fontWeight={700}>
              威布尔可靠性分析
            </Typography>
            <Typography variant="body2" color="text.secondary">
              基于威布尔分布的失效数据分析与寿命评估
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={handleReset}
            >
              重置
            </Button>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
              onClick={runAnalysis}
              disabled={loading || failureTimes.length < 3}
            >
              {loading ? '分析中...' : '开始分析'}
            </Button>
          </Stack>
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" onClose={() => patch({ error: null })}>
            {error}
          </Alert>
        )}

        {/* Data count info */}
        {failureTimes.length > 0 && (
          <Alert severity="info" icon={false}>
            已输入 {failureTimes.length} 个失效时间数据
            {censoredTimes.length > 0 && `，${censoredTimes.length} 个删失数据`}
          </Alert>
        )}

        {/* Tabs */}
        <Paper sx={{ width: '100%' }}>
          <Tabs
            value={tab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="数据输入" />
            <Tab label="拟合结果" disabled={!fitResult} />
            <Tab label="概率图" disabled={!probabilityPlotData} />
            <Tab label="可靠度与失效率" disabled={!fitResult} />
            <Tab label="汇总报告" disabled={!fitResult} />
          </Tabs>

          {/* Tab 0: Data Input */}
          <TabPanel value={tab} index={0}>
            <Box sx={{ p: 2 }}>
              <WeibullInput />
            </Box>
          </TabPanel>

          {/* Tab 1: Fit Results */}
          <TabPanel value={tab} index={1}>
            <Box sx={{ p: 2 }}>
              {fitResult && (
                <>
                  <WeibullFitResultDisplay result={fitResult} />
                  <Divider sx={{ my: 3 }} />
                  <WeibullBLifeTable
                    fitResult={fitResult}
                    customBLifes={customBLifes}
                    customBLifeResults={customBLifeResults}
                    onAddBLife={handleAddBLife}
                    onRemoveBLife={handleRemoveBLife}
                  />
                </>
              )}
            </Box>
          </TabPanel>

          {/* Tab 2: Probability Plot */}
          <TabPanel value={tab} index={2}>
            <Box sx={{ p: 2 }}>
              {probabilityPlotData && fitResult && (
                <WeibullProbabilityPlot
                  data={probabilityPlotData}
                  shape={fitResult.shape}
                  scale={fitResult.scale}
                />
              )}
            </Box>
          </TabPanel>

          {/* Tab 3: Reliability & Hazard Curves */}
          <TabPanel value={tab} index={3}>
            <Box sx={{ p: 2 }}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  {reliabilityCurveData && fitResult && (
                    <WeibullReliabilityCurve
                      curveData={reliabilityCurveData}
                      fitResult={fitResult}
                    />
                  )}
                </Grid>
                <Grid item xs={12}>
                  {hazardCurveData && fitResult && (
                    <WeibullHazardCurve
                      curveData={hazardCurveData}
                      shape={fitResult.shape}
                    />
                  )}
                </Grid>
                <Grid item xs={12}>
                  {pdfCdfOption && (
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                        PDF与CDF曲线
                      </Typography>
                      <Box sx={{ height: 400 }}>
                        <ReactECharts
                          option={pdfCdfOption}
                          style={{ height: '100%', width: '100%' }}
                          opts={{ renderer: 'canvas' }}
                        />
                      </Box>
                    </Paper>
                  )}
                </Grid>
              </Grid>
            </Box>
          </TabPanel>

          {/* Tab 4: Summary Report */}
          <TabPanel value={tab} index={4}>
            <Box sx={{ p: 2 }}>
              {fitResult && (
                <Grid container spacing={3}>
                  {/* Summary Statistics */}
                  <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" fontWeight={600} gutterBottom>
                        威布尔分析汇总报告
                      </Typography>
                      <Divider sx={{ mb: 2 }} />

                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle2" color="text.secondary">
                            分布参数
                          </Typography>
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2">
                              形状参数 (β): <strong>{fitResult.shape.toFixed(4)}</strong>
                            </Typography>
                            <Typography variant="body2">
                              尺度参数 (η): <strong>{fitResult.scale.toFixed(2)}</strong> 小时/循环
                            </Typography>
                            <Typography variant="body2">
                              平均失效时间 (MTTF): <strong>{fitResult.mttf.toFixed(2)}</strong> 小时/循环
                            </Typography>
                          </Box>
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle2" color="text.secondary">
                            拟合质量
                          </Typography>
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2">
                              R²: <strong>{fitResult.r_squared.toFixed(4)}</strong>
                            </Typography>
                            <Typography variant="body2">
                              数据点数: <strong>{failureTimes.length}</strong>
                            </Typography>
                            {censoredTimes.length > 0 && (
                              <Typography variant="body2">
                                删失数据: <strong>{censoredTimes.length}</strong>
                              </Typography>
                            )}
                          </Box>
                        </Grid>

                        <Grid item xs={12}>
                          <Typography variant="subtitle2" color="text.secondary">
                            B寿命
                          </Typography>
                          <Box sx={{ mt: 1, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                            <Typography variant="body2">
                              B10: <strong>{fitResult.b10.toFixed(1)}</strong>
                            </Typography>
                            <Typography variant="body2">
                              B50: <strong>{fitResult.b50.toFixed(1)}</strong>
                            </Typography>
                            <Typography variant="body2">
                              B63.2: <strong>{fitResult.b63_2.toFixed(1)}</strong>
                            </Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    </Paper>
                  </Grid>

                  {/* Reliability Curve Summary */}
                  <Grid item xs={12} md={6}>
                    {reliabilityCurveData && (
                      <WeibullReliabilityCurve
                        curveData={reliabilityCurveData}
                        fitResult={fitResult}
                      />
                    )}
                  </Grid>

                  {/* Probability Plot Summary */}
                  <Grid item xs={12} md={6}>
                    {probabilityPlotData && (
                      <WeibullProbabilityPlot
                        data={probabilityPlotData}
                        shape={fitResult.shape}
                        scale={fitResult.scale}
                      />
                    )}
                  </Grid>
                </Grid>
              )}
            </Box>
          </TabPanel>
        </Paper>
      </Stack>
    </Container>
  )
}

export default WeibullAnalysis
