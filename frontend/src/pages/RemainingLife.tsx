import React, { useState, useEffect } from 'react'
import {
  Box,
  Container,
  Stack,
  Typography,
  Alert,
  CircularProgress,
  Paper,
  Divider,
  Grid,
  Card,
  CardContent,
  Button,
} from '@mui/material'
import {
  Info as InfoIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material'
import { DegradationInput, DegradationData } from '@/components/RemainingLife/DegradationInput'
import { HealthIndicator } from '@/components/RemainingLife/HealthIndicator'
import { apiService } from '@/services/api'
import { DamageAccumulationResult } from '@/types'
import ReactECharts from 'echarts-for-react'

interface RemainingLifeData {
  healthIndex: number
  remainingLifeHours: number
  trend: 'improving' | 'stable' | 'degrading'
  confidenceInterval: [number, number]
  predictedData: Array<{ timestamp: string; healthIndex: number }>
}

export const RemainingLife: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentDamage, setCurrentDamage] = useState<number>(0)
  const [degradationData, setDegradationData] = useState<DegradationData[]>([])
  const [lifeData, setLifeData] = useState<RemainingLifeData | null>(null)
  const [operatingHours, setOperatingHours] = useState<number>(0)

  const handleDegradationChange = (data: DegradationData[]) => {
    setDegradationData(data)
    if (data.length > 0) {
      const maxHours = Math.max(...data.map((d) => d.operatingHours))
      setOperatingHours(maxHours)
    }
  }

  const handleCalculate = async () => {
    if (currentDamage <= 0 && degradationData.length === 0) {
      setError('请输入当前损伤状态或退化数据')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Use damage-based calculation if no degradation data
      if (degradationData.length === 0) {
        const healthIndex = Math.max(0, 100 - currentDamage * 100)
        const remainingCapacity = Math.max(0, 1 - currentDamage)

        setLifeData({
          healthIndex,
          remainingLifeHours: remainingCapacity * 100000, // Assume 100k hours baseline
          trend: 'stable',
          confidenceInterval: [
            Math.max(0, healthIndex - 10),
            Math.min(100, healthIndex + 10),
          ],
          predictedData: [],
        })
      } else {
        // Calculate from degradation trend
        // Simple linear extrapolation
        const sortedData = [...degradationData].sort(
          (a, b) => a.operatingHours - b.operatingHours
        )

        const latestData = sortedData[sortedData.length - 1]
        const healthIndex = latestData.healthIndex ?? 100

        // Calculate trend
        let trend: 'improving' | 'stable' | 'degrading' = 'stable'
        if (sortedData.length >= 2) {
          const firstHealth = sortedData[0].healthIndex ?? 100
          const healthChange = firstHealth - healthIndex
          if (healthChange > 5) trend = 'degrading'
          else if (healthChange < -5) trend = 'improving'
        }

        // Extrapolate remaining life
        let remainingLifeHours = 0
        if (sortedData.length >= 2 && trend === 'degrading') {
          const degradationRate =
            (100 - healthIndex) / latestData.operatingHours
          remainingLifeHours = healthIndex / degradationRate
        } else {
          remainingLifeHours = (100 - healthIndex) * 1000
        }

        // Generate predicted data
        const predictedData: Array<{ timestamp: string; healthIndex: number }> = []
        const now = new Date()
        for (let i = 1; i <= 12; i++) {
          const futureDate = new Date(now.getTime() + i * 30 * 24 * 60 * 60 * 1000)
          const futureHealth = Math.max(
            0,
            healthIndex - (i * 5) // Assume 5% degradation per month
          )
          predictedData.push({
            timestamp: futureDate.toISOString(),
            healthIndex: futureHealth,
          })
        }

        setLifeData({
          healthIndex,
          remainingLifeHours,
          trend,
          confidenceInterval: [
            Math.max(0, healthIndex - 15),
            Math.min(100, healthIndex + 15),
          ],
          predictedData,
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '计算失败')
    } finally {
      setLoading(false)
    }
  }

  const getExtrapolationOption = () => {
    if (!lifeData || degradationData.length === 0) return null

    const historicalData = degradationData.map((d) => ({
      timestamp: d.timestamp,
      healthIndex: d.healthIndex ?? 100 - currentDamage * 100,
    }))

    const historicalDates = historicalData.map((d) => new Date(d.timestamp).toLocaleDateString())
    const historicalValues = historicalData.map((d) => d.healthIndex)

    const predictedDates = lifeData.predictedData.map((d) =>
      new Date(d.timestamp).toLocaleDateString()
    )
    const predictedValues = lifeData.predictedData.map((d) => d.healthIndex)

    return {
      title: {
        text: '健康指数趋势外推',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        data: ['历史数据', '预测趋势'],
        top: 30,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '20%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: [...historicalDates, ...predictedDates],
        axisLabel: {
          fontSize: 10,
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        name: '健康指数 (%)',
      },
      series: [
        {
          name: '历史数据',
          type: 'line',
          data: historicalValues.map((v, i) =>
            i < historicalDates.length - 1 ? v : null
          ),
          showSymbol: true,
          itemStyle: {
            color: '#4287f5',
          },
        },
        {
          name: '历史数据',
          type: 'line',
          data: [...historicalValues, ...Array(predictedDates.length).fill(null)],
          lineStyle: {
            type: 'solid',
            color: '#4287f5',
          },
          showSymbol: false,
        },
        {
          name: '预测趋势',
          type: 'line',
          data: [
            ...Array(historicalDates.length - 1).fill(null),
            historicalValues[historicalValues.length - 1],
            ...predictedValues,
          ],
          lineStyle: {
            type: 'dashed',
            color: '#e53e3e',
          },
          itemStyle: {
            color: '#e53e3e',
          },
        },
      ],
      markLine: {
        data: [
          { yAxis: 20, label: { formatter: '危险阈值' } },
          { yAxis: 50, label: { formatter: '维护阈值' } },
        ],
        lineStyle: {
          type: 'solid',
          color: '#ff9800',
        },
      },
    }
  }

  const extrapolationOption = getExtrapolationOption()

  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        {/* Page Header */}
        <Box>
          <Typography variant="h4" gutterBottom>
            剩余寿命评估
          </Typography>
          <Typography variant="body1" color="text.secondary">
            基于当前器件状态和历史退化数据，评估器件的健康状态并预测剩余使用寿命。
            结合累积损伤分析和退化趋势分析，为维护决策提供依据。
          </Typography>
        </Box>

        {/* Current Damage Input */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                当前损伤状态
              </Typography>
              <Stack spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  如果已经完成累积损伤分析，可以输入当前的累积损伤值进行剩余寿命评估。
                </Typography>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    累积损伤比 (D):
                  </Typography>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={currentDamage}
                    onChange={(e) => setCurrentDamage(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <Typography variant="h6" textAlign="center" sx={{ mt: 1 }}>
                    {(currentDamage * 100).toFixed(1)}%
                  </Typography>
                </Box>
                {currentDamage > 0 && (
                  <Alert severity={currentDamage > 0.8 ? 'error' : currentDamage > 0.5 ? 'warning' : 'info'}>
                    健康指数: {Math.max(0, 100 - currentDamage * 100).toFixed(1)}%
                  </Alert>
                )}
              </Stack>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <DegradationInput
              onDataChange={handleDegradationChange}
              currentDamage={currentDamage}
            />
          </Grid>
        </Grid>

        {/* Calculate Button */}
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleCalculate}
            disabled={loading || (currentDamage <= 0 && degradationData.length === 0)}
          >
            评估剩余寿命
          </Button>
        </Box>

        {/* Loading State */}
        {loading && (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <CircularProgress size={40} />
            <Typography sx={{ mt: 2 }}>正在评估剩余寿命...</Typography>
          </Paper>
        )}

        {/* Error State */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Results */}
        {lifeData && !loading && (
          <>
            <Divider />
            <Typography variant="h5">评估结果</Typography>

            <HealthIndicator
              data={lifeData}
              historicalData={degradationData.map((d) => ({
                timestamp: d.timestamp,
                healthIndex: d.healthIndex ?? 100 - currentDamage * 100,
              }))}
            />

            {/* Extrapolation Chart */}
            {extrapolationOption && (
              <Card>
                <CardContent>
                  <Stack direction="row" spacing={1} alignItems="center" mb={2}>
                    <TimelineIcon color="primary" />
                    <Typography variant="h6">趋势外推预测</Typography>
                  </Stack>
                  <ReactECharts option={extrapolationOption} style={{ height: '300px' }} />
                  <Box sx={{ mt: 2 }}>
                    <Alert severity="info" icon={<InfoIcon />}>
                      <Typography variant="body2">
                        基于历史退化数据进行的线性外推预测。实际剩余寿命可能受多种因素影响，
                        包括工作条件变化、维护措施等。建议定期更新数据以获得更准确的预测。
                      </Typography>
                    </Alert>
                  </Box>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  维护建议
                </Typography>
                <Stack spacing={2}>
                  {lifeData.healthIndex >= 80 && (
                    <Alert severity="success">
                      器件状态良好，建议按常规维护计划进行检查。
                    </Alert>
                  )}
                  {lifeData.healthIndex >= 50 && lifeData.healthIndex < 80 && (
                    <Alert severity="info">
                      器件状态正常，建议每 {Math.floor(lifeData.remainingLifeHours / 10)} 运行小时进行一次检查。
                    </Alert>
                  )}
                  {lifeData.healthIndex >= 25 && lifeData.healthIndex < 50 && (
                    <Alert severity="warning">
                      器件状态一般，建议在 {Math.floor(lifeData.remainingLifeHours / 2)} 运行小时内安排维护。
                    </Alert>
                  )}
                  {lifeData.healthIndex < 25 && (
                    <Alert severity="error">
                      器件状态较差，建议尽快安排更换或大修。
                    </Alert>
                  )}
                </Stack>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
              <Button
                variant="outlined"
                onClick={() => setLifeData(null)}
              >
                重新评估
              </Button>
              <Button
                variant="outlined"
                onClick={() => {
                  setLifeData(null)
                  setCurrentDamage(0)
                  setDegradationData([])
                }}
              >
                新建评估
              </Button>
            </Box>
          </>
        )}

        {/* Information Section */}
        <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
          <Typography variant="h6" gutterBottom>
            关于剩余寿命预测
          </Typography>
          <Typography variant="body2" paragraph>
            剩余寿命预测基于以下方法：
          </Typography>
          <ul style={{ marginTop: 0, paddingLeft: 20 }}>
            <li>
              <Typography variant="body2">
                <strong>基于损伤:</strong> 使用累积损伤比D，计算剩余损伤容量(1-D)，结合模型参数估算剩余循环数。
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>基于退化数据:</strong> 分析历史退化趋势，外推至失效阈值以估算剩余寿命。
              </Typography>
            </li>
            <li>
              <Typography variant="body2">
                <strong>健康指数:</strong> 综合考虑电气参数变化（如Vce上升）和运行时间，量化器件健康状态。
              </Typography>
            </li>
          </ul>
          <Typography variant="body2" color="text.secondary">
            注意: 预测结果基于当前数据和假设，实际寿命可能因运行条件变化、维护措施等因素而有所不同。
          </Typography>
        </Paper>
      </Stack>
    </Container>
  )
}

export default RemainingLife
