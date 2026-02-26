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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from '@mui/material'
import { MissionProfileInput, MissionStep } from '@/components/DamageAccumulation/MissionProfileInput'
import { DamageDisplay } from '@/components/DamageAccumulation/DamageDisplay'
import { apiService } from '@/services/api'
import { DamageAccumulationResult, LifetimeModelType } from '@/types'

export const DamageAccumulation: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DamageAccumulationResult | null>(null)
  const [missionSteps, setMissionSteps] = useState<MissionStep[]>([])
  const [selectedModel, setSelectedModel] = useState<LifetimeModelType>('cips2008')
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [modelParams, setModelParams] = useState<Record<string, unknown>>({})

  useEffect(() => {
    loadModels()
    loadDefaultParams()
  }, [])

  useEffect(() => {
    if (selectedModel) {
      loadDefaultParams()
    }
  }, [selectedModel])

  const loadModels = async () => {
    try {
      const response = await apiService.getModels()
      if (response.success && response.data) {
        const models = response.data as Array<{ type: string; name: string }>
        setAvailableModels(models.map((m) => m.type))
      }
    } catch (err) {
      console.error('Failed to load models', err)
    }
  }

  const loadDefaultParams = async () => {
    try {
      const response = await apiService.getModelDefaultParams(selectedModel)
      if (response.success && response.data) {
        setModelParams(response.data as Record<string, unknown>)
      }
    } catch (err) {
      console.error('Failed to load default params', err)
    }
  }

  const handleProfileChange = (steps: MissionStep[]) => {
    setMissionSteps(steps)
    setResult(null)
  }

  const handleCalculate = async () => {
    if (missionSteps.length === 0) {
      setError('请先定义任务剖面')
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Convert mission steps to cycles format
      const cycles = missionSteps.map((step) => ({
        Tmax: step.tjMax,
        Tmin: step.tjMax - step.deltaTj,
        theating: step.tOn,
      }))

      const response = await apiService.calculateDamage({
        modelType: selectedModel,
        params: modelParams,
        cycles: cycles.map((c) => ({ ...c, count: 1 })),
      })

      if (response.success && response.data) {
        setResult(response.data as DamageAccumulationResult)
      } else {
        setError(response.error || '计算失败')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '网络请求失败')
    } finally {
      setLoading(false)
    }
  }

  const modelLabels: Record<LifetimeModelType, string> = {
    coffin_manson: 'Coffin-Manson',
    coffin_manson_arrhenius: 'Coffin-Manson-Arrhenius',
    norris_landzberg: 'Norris-Landzberg',
    cips2008: 'CIPS 2008',
    lesit: 'LESIT',
  }

  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        {/* Page Header */}
        <Box>
          <Typography variant="h4" gutterBottom>
            累积损伤分析
          </Typography>
          <Typography variant="body1" color="text.secondary">
            基于Miner线性累积损伤理论，计算器件在给定任务剖面下的累积损伤，并评估剩余寿命。
            根据Miner理论，当累积损伤D达到1.0时，器件发生失效。
          </Typography>
        </Box>

        {/* Model Selection */}
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            寿命模型选择
          </Typography>
          <Stack direction="row" spacing={2} alignItems="center">
            <FormControl sx={{ minWidth: 250 }}>
              <InputLabel>选择模型</InputLabel>
              <Select
                value={selectedModel}
                label="选择模型"
                onChange={(e) => setSelectedModel(e.target.value as LifetimeModelType)}
                disabled={loading}
              >
                {availableModels.map((model) => (
                  <MenuItem key={model} value={model}>
                    {modelLabels[model as LifetimeModelType] || model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box sx={{ flex: 1 }}>
              <Typography variant="body2" color="text.secondary">
                当前使用: <strong>{modelLabels[selectedModel]}</strong>
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {selectedModel === 'cips2008' &&
                  'CIPS 2008模型基于LESIT研究所的功率循环实验数据，适用于IGBT模块的寿命预测。'}
                {selectedModel === 'coffin_manson' &&
                  'Coffin-Manson模型是经典的疲劳寿命预测模型，基于塑性应变幅与寿命的关系。'}
                {selectedModel === 'norris_landzberg' &&
                  'Norris-Landzberg模型考虑了温度循环频率的影响，适用于电子封装焊点疲劳分析。'}
              </Typography>
            </Box>
          </Stack>

          {/* Model Parameters Display */}
          {Object.keys(modelParams).length > 0 && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                模型参数:
              </Typography>
              <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
                {Object.entries(modelParams).map(([key, value]) => (
                  <Box key={key}>
                    <Typography variant="caption" color="text.secondary">
                      {key}:
                    </Typography>
                    <Typography variant="body2" sx={{ ml: 0.5 }}>
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Box>
          )}
        </Paper>

        {/* Mission Profile Input */}
        <MissionProfileInput
          onProfileChange={handleProfileChange}
          initialSteps={missionSteps}
        />

        {/* Calculate Button */}
        {missionSteps.length > 0 && !loading && !result && (
          <Box sx={{ display: 'flex', justifyContent: 'center' }}>
            <Button
              variant="contained"
              size="large"
              onClick={handleCalculate}
              disabled={missionSteps.length === 0}
            >
              计算累积损伤
            </Button>
          </Box>
        )}

        {/* Loading State */}
        {loading && (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <CircularProgress size={40} />
            <Typography sx={{ mt: 2 }}>正在计算累积损伤...</Typography>
          </Paper>
        )}

        {/* Error State */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Results */}
        {result && !loading && (
          <>
            <Divider />
            <DamageDisplay result={result} />

            {/* Recalculate Button */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
              <Button
                variant="outlined"
                onClick={() => setResult(null)}
              >
                重新计算
              </Button>
              <Button
                variant="outlined"
                onClick={() => {
                  setResult(null)
                  setMissionSteps([])
                }}
              >
                新建分析
              </Button>
            </Box>
          </>
        )}

        {/* Information Section */}
        <Paper sx={{ p: 2, bgcolor: 'info.main', color: 'info.contrastText' }}>
          <Typography variant="h6" gutterBottom>
            关于累积损伤理论
          </Typography>
          <Typography variant="body2" paragraph>
            Miner线性累积损伤理论假设每个循环对材料造成的损伤是独立的且可累加的。
            总损伤D的计算公式为:
          </Typography>
          <Box sx={{ bgcolor: 'rgba(255,255,255,0.1)', p: 1, borderRadius: 1, my: 1 }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              D = Σ (n_i / N_i)
            </Typography>
          </Box>
          <Typography variant="body2" paragraph>
            其中 n_i 是第i级应力水平下的实际循环次数，N_i 是该应力水平下的疲劳寿命（失效循环数）。
          </Typography>
          <Typography variant="body2">
            当 D ≥ 1 时，认为材料已发生疲劳失效。该理论简单实用，在工程上得到广泛应用，
            但未考虑加载顺序的影响，在某些情况下可能与实际有偏差。
          </Typography>
        </Paper>
      </Stack>
    </Container>
  )
}

export default DamageAccumulation
