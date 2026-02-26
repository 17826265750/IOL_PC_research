import React from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  LinearProgress,
  Alert,
  AlertTitle,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material'
import {
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material'
import { DamageAccumulationResult } from '@/types'

interface Props {
  result: DamageAccumulationResult | null
  loading?: boolean
}

const getDamageColor = (damage: number): string => {
  if (damage < 0.5) return '#4caf50' // green
  if (damage < 0.8) return '#ff9800' // orange
  return '#f44336' // red
}

const getDamageStatus = (damage: number): { color: string; text: string; icon: React.ReactElement } => {
  if (damage >= 1.0) {
    return {
      color: 'error',
      text: '已失效',
      icon: <ErrorIcon />,
    }
  }
  if (damage >= 0.8) {
    return {
      color: 'error',
      text: '严重损伤',
      icon: <WarningIcon />,
    }
  }
  if (damage >= 0.5) {
    return {
      color: 'warning',
      text: '中等损伤',
      icon: <WarningIcon />,
    }
  }
  return {
    color: 'success',
    text: '状态良好',
    icon: <CheckCircleIcon />,
  }
}

export const DamageDisplay: React.FC<Props> = ({ result, loading }) => {
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <Typography color="text.secondary">正在计算损伤...</Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography color="text.secondary">
              请输入任务剖面并选择模型进行损伤计算
            </Typography>
          </Box>
        </CardContent>
      </Card>
    )
  }

  const damageStatus = getDamageStatus(result.totalDamage)

  return (
    <Stack spacing={2}>
      {/* Critical Warning */}
      {result.totalDamage >= 1.0 && (
        <Alert severity="error" icon={<ErrorIcon />}>
          <AlertTitle>警告：累计损伤已达到或超过1.0</AlertTitle>
          根据Miner线性累积损伤理论，器件已发生失效。建议立即检查或更换。
        </Alert>
      )}

      {result.totalDamage >= 0.8 && result.totalDamage < 1.0 && (
        <Alert severity="warning" icon={<WarningIcon />}>
          <AlertTitle>注意：累计损伤接近临界值</AlertTitle>
          当前累计损伤为 {(result.totalDamage * 100).toFixed(1)}%，接近失效阈值。建议计划维护。
        </Alert>
      )}

      {/* Main Damage Display */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            累积损伤分析结果
          </Typography>

          <Stack spacing={3}>
            {/* Progress Bar */}
            <Box>
              <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                mb={1}
              >
                <Typography variant="body2" color="text.secondary">
                  累积损伤比 (D)
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  {React.cloneElement(damageStatus.icon, {
                    sx: { color: getDamageColor(result.totalDamage) },
                  })}
                  <Typography
                    variant="h5"
                    sx={{ color: getDamageColor(result.totalDamage), fontWeight: 600 }}
                  >
                    {(result.totalDamage * 100).toFixed(2)}%
                  </Typography>
                </Stack>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(result.totalDamage * 100, 100)}
                sx={{
                  height: 12,
                  borderRadius: 6,
                  backgroundColor: 'grey.200',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getDamageColor(result.totalDamage),
                    borderRadius: 6,
                  },
                }}
              />
              <Stack direction="row" justifyContent="space-between" mt={0.5}>
                <Typography variant="caption" color="text.secondary">
                  0%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  100% (失效)
                </Typography>
              </Stack>
            </Box>

            {/* Status Cards */}
            <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
              <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
                <Typography variant="caption" color="text.secondary">
                  当前状态
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center" mt={0.5}>
                  {React.cloneElement(damageStatus.icon, {
                    fontSize: 'small',
                    color: getDamageColor(result.totalDamage) as any,
                  })}
                  <Typography
                    variant="h6"
                    sx={{ color: getDamageColor(result.totalDamage) }}
                  >
                    {damageStatus.text}
                  </Typography>
                </Stack>
              </Paper>

              <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
                <Typography variant="caption" color="text.secondary">
                  剩余损伤容量
                </Typography>
                <Typography variant="h6" sx={{ color: 'success.main', mt: 0.5 }}>
                  {Math.max(0, result.remainingDamage * 100).toFixed(2)}%
                </Typography>
              </Paper>

              <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
                <Typography variant="caption" color="text.secondary">
                  剩余寿命
                </Typography>
                <Typography variant="h6" sx={{ color: 'info.main', mt: 0.5 }}>
                  {result.remainingLifePercent.toFixed(1)}%
                </Typography>
              </Paper>

              <Paper sx={{ p: 2, flex: 1, minWidth: 200 }}>
                <Typography variant="caption" color="text.secondary">
                  使用的模型
                </Typography>
                <Typography variant="body1" sx={{ fontWeight: 500, mt: 0.5 }}>
                  {result.modelType}
                </Typography>
              </Paper>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {/* Damage Breakdown Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            损伤明细
          </Typography>

          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>序号</TableCell>
                  <TableCell>循环数</TableCell>
                  <TableCell>失效循环数</TableCell>
                  <TableCell>损伤贡献</TableCell>
                  <TableCell>累积损伤</TableCell>
                  <TableCell>损伤占比</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {result.entries.map((entry, index) => (
                  <TableRow key={entry.cycleIndex}>
                    <TableCell>{index + 1}</TableCell>
                    <TableCell>{entry.cycles.toLocaleString()}</TableCell>
                    <TableCell>
                      {entry.cyclesToFailure > 0
                        ? entry.cyclesToFailure.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })
                        : 'N/A'}
                    </TableCell>
                    <TableCell>
                      <Typography
                        variant="body2"
                        sx={{
                          color:
                            entry.damage > 0.1
                              ? 'error.main'
                              : entry.damage > 0.05
                              ? 'warning.main'
                              : 'text.primary',
                          fontWeight: entry.damage > 0.1 ? 600 : 400,
                        }}
                      >
                        {entry.damage > 0 ? (entry.damage * 100).toFixed(4) : '0'}%
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography
                        variant="body2"
                        sx={{
                          color:
                            entry.cumulativeDamage > 0.8
                              ? 'error.main'
                              : entry.cumulativeDamage > 0.5
                              ? 'warning.main'
                              : 'text.primary',
                          fontWeight: entry.cumulativeDamage > 0.5 ? 600 : 400,
                        }}
                      >
                        {(entry.cumulativeDamage * 100).toFixed(2)}%
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={
                            result.totalDamage > 0
                              ? (entry.damage / result.totalDamage) * 100
                              : 0
                          }
                          sx={{ flex: 1, height: 6, borderRadius: 3 }}
                        />
                        <Typography variant="caption" sx={{ minWidth: 45 }}>
                          {result.totalDamage > 0
                            ? ((entry.damage / result.totalDamage) * 100).toFixed(1)
                            : '0'}
                          %
                        </Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            建议
          </Typography>
          <Stack spacing={1}>
            {result.totalDamage < 0.3 && (
              <Typography variant="body2" color="success.main">
                • 当前器件状态良好，可继续正常运行
              </Typography>
            )}
            {result.totalDamage >= 0.3 && result.totalDamage < 0.7 && (
              <>
                <Typography variant="body2" color="warning.main">
                  • 建议定期监测器件状态
                </Typography>
                <Typography variant="body2" color="warning.main">
                  • 考虑降低工作温度或减少温度变化幅度
                </Typography>
              </>
            )}
            {result.totalDamage >= 0.7 && result.totalDamage < 1.0 && (
              <>
                <Typography variant="body2" color="error.main">
                  • 建议尽快安排维护检查
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 考虑准备备件进行更换
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 优化工作条件以减少进一步损伤
                </Typography>
              </>
            )}
            {result.totalDamage >= 1.0 && (
              <>
                <Typography variant="body2" color="error.main">
                  • 器件已达到失效条件，应立即停止使用
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 进行全面的器件检查和测试
                </Typography>
                <Typography variant="body2" color="error.main">
                  • 更换失效器件后分析失效原因
                </Typography>
              </>
            )}
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  )
}

export default DamageDisplay
