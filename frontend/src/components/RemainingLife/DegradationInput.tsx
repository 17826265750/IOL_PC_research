import React, { useState, useRef } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  TextField,
  Button,
  Paper,
  Alert,
  Divider,
  InputAdornment,
} from '@mui/material'
import {
  Upload as UploadIcon,
  Clear as ClearIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material'

export interface DegradationData {
  timestamp: string
  operatingHours: number
  vceOn?: number // On-state voltage drop
  vceOff?: number // Off-state leakage
  thermalResistance?: number
  healthIndex?: number // 0-100
}

interface Props {
  onDataChange: (data: DegradationData[]) => void
  currentDamage?: number
}

export const DegradationInput: React.FC<Props> = ({ onDataChange, currentDamage = 0 }) => {
  const [currentOperatingHours, setCurrentOperatingHours] = useState<number>(0)
  const [currentVceOn, setCurrentVceOn] = useState<number>(0)
  const [currentVceOff, setCurrentVceOff] = useState<number>(0)
  const [degradationHistory, setDegradationHistory] = useState<DegradationData[]>([])
  const [manualEntry, setManualEntry] = useState('')
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleManualEntrySubmit = () => {
    try {
      const parsed = JSON.parse(manualEntry)
      if (Array.isArray(parsed)) {
        const data: DegradationData[] = parsed.map((item: any) => ({
          timestamp: item.timestamp || item.date || new Date().toISOString(),
          operatingHours: Number(item.operatingHours || item.hours || 0),
          vceOn: item.vceOn !== undefined ? Number(item.vceOn) : undefined,
          vceOff: item.vceOff !== undefined ? Number(item.vceOff) : undefined,
          thermalResistance:
            item.thermalResistance !== undefined ? Number(item.thermalResistance) : undefined,
          healthIndex: item.healthIndex !== undefined ? Number(item.healthIndex) : undefined,
        }))
        setDegradationHistory(data)
        onDataChange(data)
        setError(null)
      }
    } catch (e) {
      setError('JSON 格式错误，请检查输入')
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string
        const parsed = JSON.parse(content)
        if (Array.isArray(parsed)) {
          const data: DegradationData[] = parsed.map((item: any) => ({
            timestamp: item.timestamp || item.date || new Date().toISOString(),
            operatingHours: Number(item.operatingHours || item.hours || 0),
            vceOn: item.vceOn !== undefined ? Number(item.vceOn) : undefined,
            vceOff: item.vceOff !== undefined ? Number(item.vceOff) : undefined,
            thermalResistance:
              item.thermalResistance !== undefined ? Number(item.thermalResistance) : undefined,
            healthIndex: item.healthIndex !== undefined ? Number(item.healthIndex) : undefined,
          }))
          setDegradationHistory(data)
          onDataChange(data)
          setError(null)
        }
      } catch (err) {
        setError('文件解析失败，请检查文件格式')
      }
    }
    reader.readAsText(file)
  }

  const handleClear = () => {
    setDegradationHistory([])
    setCurrentOperatingHours(0)
    setCurrentVceOn(0)
    setCurrentVceOff(0)
    setManualEntry('')
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    onDataChange([])
  }

  const handleAddCurrentPoint = () => {
    if (currentOperatingHours <= 0) {
      setError('请输入有效的运行小时数')
      return
    }

    const newPoint: DegradationData = {
      timestamp: new Date().toISOString(),
      operatingHours: currentOperatingHours,
      vceOn: currentVceOn > 0 ? currentVceOn : undefined,
      vceOff: currentVceOff > 0 ? currentVceOff : undefined,
      healthIndex: Math.max(0, 100 - currentDamage * 100),
    }

    const updated = [...degradationHistory, newPoint]
    setDegradationHistory(updated)
    onDataChange(updated)
    setError(null)
  }

  // Calculate degradation trend
  const getDegradationTrend = () => {
    if (degradationHistory.length < 2) return null

    const vceOnData = degradationHistory.filter((d) => d.vceOn !== undefined)
    if (vceOnData.length < 2) return null

    // Simple linear regression
    const n = vceOnData.length
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0

    vceOnData.forEach((d) => {
      sumX += d.operatingHours
      sumY += d.vceOn!
      sumXY += d.operatingHours * d.vceOn!
      sumX2 += d.operatingHours * d.operatingHours
    })

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)

    if (slope > 0.0001) return 'degrading'
    if (slope < -0.0001) return 'improving'
    return 'stable'
  }

  const trend = getDegradationTrend()

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          退化数据输入
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Stack spacing={3}>
          {/* Current State Input */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              当前器件状态
            </Typography>
            <Stack spacing={2} direction="row" flexWrap="wrap" useFlexGap>
              <TextField
                label="运行小时数"
                type="number"
                value={currentOperatingHours || ''}
                onChange={(e) => setCurrentOperatingHours(parseFloat(e.target.value) || 0)}
                InputProps={{
                  endAdornment: <InputAdornment position="end">h</InputAdornment>,
                }}
                sx={{ minWidth: 200 }}
              />
              <TextField
                label="导通压降 Vce(on)"
                type="number"
                value={currentVceOn || ''}
                onChange={(e) => setCurrentVceOn(parseFloat(e.target.value) || 0)}
                InputProps={{
                  endAdornment: <InputAdornment position="end">V</InputAdornment>,
                }}
                sx={{ minWidth: 200 }}
              />
              <TextField
                label="关断漏电流"
                type="number"
                value={currentVceOff || ''}
                onChange={(e) => setCurrentVceOff(parseFloat(e.target.value) || 0)}
                InputProps={{
                  endAdornment: <InputAdornment position="end">mA</InputAdornment>,
                }}
                sx={{ minWidth: 200 }}
              />
            </Stack>
            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                onClick={handleAddCurrentPoint}
                disabled={currentOperatingHours <= 0}
              >
                添加数据点
              </Button>
            </Box>
          </Paper>

          <Divider />

          {/* History Upload */}
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              历史退化数据（可选）
            </Typography>
            <Stack spacing={2}>
              <Typography variant="body2" color="text.secondary">
                上传历史退化数据文件 (JSON 格式)，用于分析退化趋势和预测剩余寿命
              </Typography>

              <Box
                sx={{
                  border: 2,
                  borderColor: 'divider',
                  borderStyle: 'dashed',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                }}
              >
                <UploadIcon sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  拖放文件到此处或点击上传
                </Typography>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,.csv"
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
                <Button variant="outlined" onClick={() => fileInputRef.current?.click()}>
                  选择文件
                </Button>
              </Box>

              {/* JSON Manual Entry */}
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  或手动输入 JSON 数据
                </Typography>
                <TextField
                  multiline
                  rows={6}
                  fullWidth
                  placeholder={`[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "operatingHours": 1000,
    "vceOn": 1.65,
    "healthIndex": 95
  },
  ...
]`}
                  value={manualEntry}
                  onChange={(e) => setManualEntry(e.target.value)}
                />
                <Button
                  variant="outlined"
                  sx={{ mt: 1 }}
                  onClick={handleManualEntrySubmit}
                  disabled={!manualEntry.trim()}
                >
                  解析并添加
                </Button>
              </Box>
            </Stack>
          </Paper>

          {/* Data Summary */}
          {degradationHistory.length > 0 && (
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="subtitle2">已上传数据</Typography>
                <Button size="small" onClick={handleClear} startIcon={<ClearIcon />}>
                  清空
                </Button>
              </Stack>

              <Stack spacing={2} direction="row" flexWrap="wrap" useFlexGap>
                <Box sx={{ minWidth: 120 }}>
                  <Typography variant="caption" color="text.secondary">
                    数据点数
                  </Typography>
                  <Typography variant="h6">{degradationHistory.length}</Typography>
                </Box>
                <Box sx={{ minWidth: 120 }}>
                  <Typography variant="caption" color="text.secondary">
                    运行时间范围
                  </Typography>
                  <Typography variant="body2">
                    {Math.min(...degradationHistory.map((d) => d.operatingHours)).toLocaleString()} -{' '}
                    {Math.max(...degradationHistory.map((d) => d.operatingHours)).toLocaleString()} h
                  </Typography>
                </Box>
                {trend && (
                  <Box sx={{ minWidth: 120 }}>
                    <Typography variant="caption" color="text.secondary">
                      退化趋势
                    </Typography>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      {trend === 'degrading' && <TrendingUpIcon color="error" fontSize="small" />}
                      {trend === 'improving' && <TrendingDownIcon color="success" fontSize="small" />}
                      <Typography
                        variant="body2"
                        color={
                          trend === 'degrading' ? 'error.main' : trend === 'improving' ? 'success.main' : 'text.primary'
                        }
                      >
                        {trend === 'degrading' ? '加速退化' : trend === 'improving' ? '改善' : '稳定'}
                      </Typography>
                    </Stack>
                  </Box>
                )}
              </Stack>

              {/* Mini table preview */}
              {degradationHistory.length > 0 && (
                <Box sx={{ mt: 2, maxHeight: 150, overflow: 'auto' }}>
                  <Typography variant="caption" color="text.secondary">
                    最近记录:
                  </Typography>
                  {degradationHistory.slice(-5).map((d, i) => (
                    <Typography key={i} variant="caption" display="block">
                      {new Date(d.timestamp).toLocaleString()} - {d.operatingHours.toLocaleString()}h
                      {d.vceOn && ` - Vce: ${d.vceOn}V`}
                      {d.healthIndex !== undefined && ` - 健康: ${d.healthIndex.toFixed(0)}%`}
                    </Typography>
                  ))}
                </Box>
              )}
            </Paper>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

export default DegradationInput
