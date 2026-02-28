/**
 * 功率模块寿命分析软件 - 威布尔分析数据输入组件
 * @author GSH
 */
import React, { useCallback, useRef } from 'react'
import {
  Box,
  TextField,
  Typography,
  Slider,
  Grid,
  Paper,
  Button,
  Chip,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Divider,
} from '@mui/material'
import { Upload as UploadIcon } from '@mui/icons-material'
import { useWeibullStore } from '@/stores/useWeibullStore'

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

export const WeibullInput: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const {
    failureTimesInput,
    censoredTimesInput,
    confidenceLevel,
    fitMethod,
    patch,
  } = useWeibullStore()

  const failureTimes = parseNumbers(failureTimesInput)
  const censoredTimes = parseNumbers(censoredTimesInput)

  const handleFailureTimesChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      patch({ failureTimesInput: e.target.value })
    },
    [patch]
  )

  const handleCensoredTimesChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      patch({ censoredTimesInput: e.target.value })
    },
    [patch]
  )

  const handleConfidenceLevelChange = useCallback(
    (_: Event, value: number | number[]) => {
      patch({ confidenceLevel: String(value) })
    },
    [patch]
  )

  const handleMethodChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      patch({ fitMethod: e.target.value as 'mle' | 'ls' })
    },
    [patch]
  )

  const handleFileUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return

      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target?.result as string
        if (content) {
          // Try to parse as CSV
          const lines = content.split(/\r?\n/).filter((line) => line.trim())

          // Check if it has header
          const firstLine = lines[0].toLowerCase()
          const hasHeader =
            firstLine.includes('failure') ||
            firstLine.includes('time') ||
            firstLine.includes('censored')

          const dataLines = hasHeader ? lines.slice(1) : lines

          // Check if we have censored data column
          if (dataLines[0]?.split(/[,\t;]/).length > 1) {
            // Two columns: failure times and censored times
            const failures: number[] = []
            const censored: number[] = []

            dataLines.forEach((line) => {
              const parts = line.split(/[,\t;]/)
              const failure = parseFloat(parts[0])
              const censor = parseFloat(parts[1])

              if (!isNaN(failure)) failures.push(failure)
              if (!isNaN(censor) && censor > 0) censored.push(censor)
            })

            patch({
              failureTimesInput: failures.join(', '),
              censoredTimesInput: censored.join(', '),
            })
          } else {
            // Single column: failure times only
            const times = dataLines
              .flatMap((line) => line.split(/[,\s]+/))
              .map((s) => parseFloat(s.trim()))
              .filter((n) => !isNaN(n) && isFinite(n))

            patch({ failureTimesInput: times.join(', ') })
          }
        }
      }
      reader.readAsText(file)

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    },
    [patch]
  )

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  // Confidence level marks
  const confidenceMarks = [
    { value: 0.8, label: '80%' },
    { value: 0.9, label: '90%' },
    { value: 0.95, label: '95%' },
    { value: 0.99, label: '99%' },
  ]

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Failure times input */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 2,
              }}
            >
              <Typography variant="subtitle1" fontWeight={600}>
                失效时间数据
              </Typography>
              <Chip
                label={`${failureTimes.length} 个数据点`}
                size="small"
                color={failureTimes.length > 0 ? 'primary' : 'default'}
              />
            </Box>
            <TextField
              multiline
              rows={6}
              fullWidth
              placeholder="输入失效时间，用逗号或换行分隔&#10;例如：100, 150, 200, 250, 300&#10;或每行一个数值"
              value={failureTimesInput}
              onChange={handleFailureTimesChange}
              sx={{
                '& .MuiInputBase-root': {
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                },
              }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              输入设备失效前运行的循环次数或时间（小时）
            </Typography>
          </Paper>
        </Grid>

        {/* Censored times input */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 2,
              }}
            >
              <Typography variant="subtitle1" fontWeight={600}>
                删失数据（可选）
              </Typography>
              <Chip
                label={`${censoredTimes.length} 个数据点`}
                size="small"
                color={censoredTimes.length > 0 ? 'secondary' : 'default'}
              />
            </Box>
            <TextField
              multiline
              rows={6}
              fullWidth
              placeholder="输入删失时间，用逗号或换行分隔&#10;删失数据指试验结束时仍未失效的样品运行时间"
              value={censoredTimesInput}
              onChange={handleCensoredTimesChange}
              sx={{
                '& .MuiInputBase-root': {
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                },
              }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              删失数据是指试验结束时仍未失效的样品的运行时间
            </Typography>
          </Paper>
        </Grid>

        {/* Confidence level slider */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              置信水平
            </Typography>
            <Box sx={{ px: 2 }}>
              <Slider
                value={parseFloat(confidenceLevel)}
                onChange={handleConfidenceLevelChange}
                min={0.8}
                max={0.99}
                step={0.01}
                marks={confidenceMarks}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              当前: {(parseFloat(confidenceLevel) * 100).toFixed(0)}% 置信水平
            </Typography>
          </Paper>
        </Grid>

        {/* Fitting method selection */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              拟合方法
            </Typography>
            <FormControl component="fieldset">
              <RadioGroup
                value={fitMethod}
                onChange={handleMethodChange}
              >
                <FormControlLabel
                  value="ls"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600}>
                        最小二乘法 (秩回归)
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        在威布尔概率纸上进行线性回归，直观易懂
                      </Typography>
                    </Box>
                  }
                />
                <FormControlLabel
                  value="mle"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600}>
                        极大似然估计 (MLE)
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        统计特性优良，适用于大数据集和删失数据
                      </Typography>
                    </Box>
                  }
                />
              </RadioGroup>
            </FormControl>
          </Paper>
        </Grid>

        {/* File upload */}
        <Grid item xs={12}>
          <input
            type="file"
            ref={fileInputRef}
            accept=".csv,.txt,.dat"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
            onClick={handleUploadClick}
            fullWidth
          >
            上传CSV文件
          </Button>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: 'block', textAlign: 'center' }}
          >
            支持CSV、TXT、DAT格式，可以是单列（失效时间）或双列（失效时间, 删失时间）
          </Typography>
        </Grid>
      </Grid>
    </Box>
  )
}

export default WeibullInput
