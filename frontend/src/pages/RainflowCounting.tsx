import React, { useState } from 'react'
import {
  Box,
  Container,
  Stack,
  Typography,
  Alert,
  CircularProgress,
  Grid,
  Paper,
  Divider,
} from '@mui/material'
import { TimeHistoryInput } from '@/components/Rainflow/TimeHistoryInput'
import { CycleMatrix } from '@/components/Rainflow/CycleMatrix'
import { RainflowHistogram } from '@/components/Rainflow/RainflowHistogram'
import { apiService } from '@/services/api'
import { RainflowResult } from '@/types'

export const RainflowCounting: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<RainflowResult | null>(null)
  const [inputData, setInputData] = useState<number[]>([])

  const handleDataSubmit = async (data: number[]) => {
    setLoading(true)
    setError(null)
    setInputData(data)

    try {
      const response = await apiService.performRainflowCounting({
        timeSeries: data,
        binCount: 20,
      })

      if (response.success && response.data) {
        setResult(response.data as RainflowResult)
      } else {
        setError(response.error || '分析失败')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '网络请求失败')
    } finally {
      setLoading(false)
    }
  }

  const handleExportMatrix = () => {
    if (!result) return

    const headers = ['范围 (°C)', '均值 (°C)', '循环次数', '类型']
    const rows = result.cycles.map((c) => [
      c.range.toFixed(2),
      c.mean.toFixed(2),
      c.count.toString(),
      c.type,
    ])

    let csv = headers.join(',') + '\n'
    rows.forEach((row) => {
      csv += row.join(',') + '\n'
    })

    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `rainflow_matrix_${Date.now()}.csv`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        {/* Page Header */}
        <Box>
          <Typography variant="h4" gutterBottom>
            雨流计数分析
          </Typography>
          <Typography variant="body1" color="text.secondary">
            雨流计数法是一种用于循环计数的方法，广泛用于疲劳寿命分析。该方法将复杂的时间历程转化为一系列封闭的应力-应变循环。
          </Typography>
        </Box>

        {/* Input Section */}
        <TimeHistoryInput onDataSubmit={handleDataSubmit} />

        {/* Loading State */}
        {loading && (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <CircularProgress size={40} />
            <Typography sx={{ mt: 2 }}>正在进行雨流计数分析...</Typography>
          </Paper>
        )}

        {/* Error State */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Results Section */}
        {result && !loading && (
          <>
            <Divider />
            <Typography variant="h5">分析结果</Typography>

            {/* Summary Statistics */}
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    数据点数
                  </Typography>
                  <Typography variant="h5">{result.originalData.length}</Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    总循环数
                  </Typography>
                  <Typography variant="h5">{result.totalCycles.toLocaleString()}</Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    唯一循环数
                  </Typography>
                  <Typography variant="h5">{result.cycles.length}</Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    最大范围
                  </Typography>
                  <Typography variant="h5">{result.maxRange.toFixed(1)}°C</Typography>
                </Paper>
              </Grid>
            </Grid>

            {/* Cycle Matrix */}
            <Box>
              <CycleMatrix cycles={result.cycles} onExport={handleExportMatrix} />
            </Box>

            {/* Histogram */}
            <Box>
              <RainflowHistogram cycles={result.cycles} binCount={result.binCount || 20} />
            </Box>

            {/* Detailed Results Table */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                循环详细数据
              </Typography>
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e0e0e0' }}>
                      <th style={{ padding: 8, textAlign: 'left' }}>序号</th>
                      <th style={{ padding: 8, textAlign: 'left' }}>范围 (°C)</th>
                      <th style={{ padding: 8, textAlign: 'left' }}>均值 (°C)</th>
                      <th style={{ padding: 8, textAlign: 'left' }}>循环次数</th>
                      <th style={{ padding: 8, textAlign: 'left' }}>类型</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.cycles.map((cycle, index) => (
                      <tr key={index} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: 8 }}>{index + 1}</td>
                        <td style={{ padding: 8 }}>{cycle.range.toFixed(2)}</td>
                        <td style={{ padding: 8 }}>{cycle.mean.toFixed(2)}</td>
                        <td style={{ padding: 8 }}>{cycle.count.toLocaleString()}</td>
                        <td style={{ padding: 8 }}>
                          {cycle.type === 'reversal' ? '反转' : '峰值'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Box>
            </Paper>
          </>
        )}
      </Stack>
    </Container>
  )
}

export default RainflowCounting
