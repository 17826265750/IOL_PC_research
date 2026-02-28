/**
 * 功率模块寿命分析软件 - 威布尔B寿命表格组件
 * @author GSH
 */
import React, { useState, useCallback } from 'react'
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Button,
  IconButton,
  Chip,
  Stack,
  Divider,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material'
import { apiService } from '@/services/api'
import type { WeibullFitResult } from '@/types'

interface WeibullBLifeTableProps {
  fitResult: WeibullFitResult
  customBLifes: number[]
  customBLifeResults: Record<number, number>
  onAddBLife: (percentile: number, value: number) => void
  onRemoveBLife: (percentile: number) => void
}

/**
 * Calculate B-life from Weibull parameters
 * B(P) = η * (-ln(1-P/100))^(1/β)
 */
function calculateBLife(shape: number, scale: number, percentile: number): number {
  const F = percentile / 100
  return scale * Math.pow(-Math.log(1 - F), 1 / shape)
}

/**
 * Format number with appropriate precision
 */
function formatNumber(value: number, decimals: number = 1): string {
  if (value >= 10000 || value < 0.01) {
    return value.toExponential(decimals)
  }
  return value.toFixed(decimals)
}

export const WeibullBLifeTable: React.FC<WeibullBLifeTableProps> = ({
  fitResult,
  customBLifes,
  customBLifeResults,
  onAddBLife,
  onRemoveBLife,
}) => {
  const [newPercentile, setNewPercentile] = useState('')
  const [error, setError] = useState<string | null>(null)

  const handleAddBLife = useCallback(() => {
    const percentile = parseFloat(newPercentile)
    if (isNaN(percentile) || percentile <= 0 || percentile >= 100) {
      setError('请输入0-100之间的有效数值')
      return
    }

    // Check if already exists
    if (customBLifes.includes(percentile)) {
      setError('该百分位已存在')
      return
    }

    // Calculate B-life
    const value = calculateBLife(fitResult.shape, fitResult.scale, percentile)
    onAddBLife(percentile, value)
    setNewPercentile('')
    setError(null)
  }, [newPercentile, customBLifes, fitResult, onAddBLife])

  const handleKeyPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        handleAddBLife()
      }
    },
    [handleAddBLife]
  )

  // Standard B-lifes
  const standardBLifes = [
    { percentile: 10, label: 'B10', description: '90%可靠度寿命（汽车行业常用）' },
    { percentile: 50, label: 'B50', description: '中位寿命' },
    { percentile: 63.2, label: 'B63.2', description: '特征寿命（等于η）' },
  ]

  // Get value for a B-life (from fit result or custom results)
  const getBLifeValue = (percentile: number): number => {
    if (percentile === 10) return fitResult.b10
    if (percentile === 50) return fitResult.b50
    if (percentile === 63.2) return fitResult.b63_2
    return customBLifeResults[percentile] ?? 0
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        B寿命（额定寿命）
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        B(P)寿命表示P%的设备失效时的运行时间，即(100-P)%可靠度寿命
      </Typography>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>寿命类型</TableCell>
              <TableCell align="right">失效百分比</TableCell>
              <TableCell align="right">可靠度</TableCell>
              <TableCell align="right">寿命值（小时/循环）</TableCell>
              <TableCell>说明</TableCell>
              <TableCell align="center">操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {/* Standard B-lifes */}
            {standardBLifes.map((b) => (
              <TableRow key={b.percentile}>
                <TableCell>
                  <Typography fontWeight={600}>{b.label}</Typography>
                </TableCell>
                <TableCell align="right">{b.percentile}%</TableCell>
                <TableCell align="right">{(100 - b.percentile).toFixed(1)}%</TableCell>
                <TableCell align="right">
                  <Typography fontWeight={600} color="primary">
                    {formatNumber(getBLifeValue(b.percentile))}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" color="text.secondary">
                    {b.description}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Chip label="标准" size="small" variant="outlined" />
                </TableCell>
              </TableRow>
            ))}

            {/* Custom B-lifes */}
            {customBLifes.sort((a, b) => a - b).map((percentile) => (
              <TableRow key={percentile}>
                <TableCell>
                  <Typography fontWeight={600}>B{percentile}</Typography>
                </TableCell>
                <TableCell align="right">{percentile}%</TableCell>
                <TableCell align="right">{(100 - percentile).toFixed(1)}%</TableCell>
                <TableCell align="right">
                  <Typography fontWeight={600}>
                    {formatNumber(getBLifeValue(percentile))}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2" color="text.secondary">
                    用户自定义
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <IconButton
                    size="small"
                    onClick={() => onRemoveBLife(percentile)}
                    color="error"
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Add custom B-life */}
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle2" gutterBottom>
        添加自定义B寿命
      </Typography>
      <Stack direction="row" spacing={1} alignItems="center">
        <TextField
          size="small"
          label="百分位 (%)"
          value={newPercentile}
          onChange={(e) => setNewPercentile(e.target.value)}
          onKeyPress={handleKeyPress}
          error={!!error}
          helperText={error}
          sx={{ width: 150 }}
          placeholder="例如: 5, 20, 90"
        />
        <Button
          variant="contained"
          size="small"
          startIcon={<AddIcon />}
          onClick={handleAddBLife}
          disabled={!newPercentile}
        >
          添加
        </Button>
      </Stack>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        输入1-99之间的任意百分位值，如5表示B5（95%可靠度寿命）
      </Typography>
    </Paper>
  )
}

export default WeibullBLifeTable
