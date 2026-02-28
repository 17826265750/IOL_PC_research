import React, { useState, useRef } from 'react'
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  Alert,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
} from '@mui/material'
import {
  Upload as UploadIcon,
  AutoFixHigh as AutoFixHighIcon,
  Clear as ClearIcon,
  Download as DownloadIcon,
} from '@mui/icons-material'

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ py: 2 }}>{children}</Box>}
    </div>
  )
}

interface Props {
  onDataSubmit: (data: number[]) => void
}

export const TimeHistoryInput: React.FC<Props> = ({ onDataSubmit }) => {
  const [tabValue, setTabValue] = useState(0)
  const [csvInput, setCsvInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<number[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
    setError(null)
  }

  const parseCSVInput = (input: string): number[] => {
    const values = input
      .split(/[\n,;\t\s]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => parseFloat(s))
      .filter((n) => !isNaN(n))

    return values
  }

  const handleCSVInputChange = (input: string) => {
    setCsvInput(input)
    const parsed = parseCSVInput(input)
    setPreviewData(parsed.slice(0, 100))
    setError(null)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      const content = e.target?.result as string
      handleCSVInputChange(content)
      setTabValue(0)
    }
    reader.onerror = () => {
      setError('文件读取失败')
    }
    reader.readAsText(file)
  }

  const generateSampleData = () => {
    const sample: number[] = []

    for (let i = 0; i < 500; i++) {
      // Generate temperature cycles with varying patterns
      const cyclePosition = (i % 100) / 100
      const baseTemp = 25 + 100 * Math.sin(cyclePosition * Math.PI)

      // Add noise
      const noise = (Math.random() - 0.5) * 10
      sample.push(baseTemp + noise)
    }

    setCsvInput(sample.join('\n'))
    setPreviewData(sample.slice(0, 100))
    setError(null)
  }

  const handleSubmit = () => {
    const data = parseCSVInput(csvInput)

    if (data.length < 4) {
      setError('至少需要4个数据点进行雨流计数分析')
      return
    }

    if (data.length < 10) {
      setError('警告: 数据点较少，可能影响分析结果')
    }

    onDataSubmit(data)
  }

  const handleClear = () => {
    setCsvInput('')
    setPreviewData([])
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleDownloadSample = () => {
    const sampleData = generateSampleCSV()
    const blob = new Blob([sampleData], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'sample_time_history.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  const generateSampleCSV = () => {
    const sample: number[] = []
    let base = 25

    for (let i = 0; i < 500; i++) {
      const cyclePosition = (i % 100) / 100
      const baseTemp = 25 + 100 * Math.sin(cyclePosition * Math.PI)
      const noise = (Math.random() - 0.5) * 10
      sample.push(baseTemp + noise)
    }

    return sample.join('\n')
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          时间历程数据输入
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 2 }}>
          <Tab label="手动输入" />
          <Tab label="文件上传" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Stack spacing={2}>
            <TextField
              label="输入数据（每行一个值，或用逗号分隔）"
              multiline
              rows={8}
              value={csvInput}
              onChange={(e) => handleCSVInputChange(e.target.value)}
              placeholder="例如：
25.5
30.2
45.8
...
或：25.5, 30.2, 45.8, ..."
              fullWidth
            />

            <Stack direction="row" spacing={2} justifyContent="space-between">
              <Stack direction="row" spacing={2}>
                <Button
                  variant="outlined"
                  startIcon={<AutoFixHighIcon />}
                  onClick={generateSampleData}
                >
                  生成示例数据
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={handleDownloadSample}
                >
                  下载示例CSV
                </Button>
              </Stack>
              <Button
                variant="outlined"
                color="error"
                startIcon={<ClearIcon />}
                onClick={handleClear}
              >
                清空
              </Button>
            </Stack>

            {previewData.length > 0 && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  数据预览（前100个点）:
                </Typography>
                <Box
                  sx={{
                    maxHeight: 200,
                    overflow: 'auto',
                    p: 2,
                    bgcolor: 'action.hover',
                    borderRadius: 1,
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                  }}
                >
                  {previewData.join(', ')}
                  {csvInput.split(/[\n,;\t\s]+/).filter((s) => s.trim().length > 0).length >
                    100 && '...'}
                </Box>
                <Typography variant="caption" color="text.secondary">
                  共{' '}
                  {parseCSVInput(csvInput).length} 个数据点
                </Typography>
              </Box>
            )}
          </Stack>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Stack spacing={2} alignItems="center">
            <Box
              sx={{
                width: '100%',
                height: 150,
                border: 2,
                borderColor: 'divider',
                borderStyle: 'dashed',
                borderRadius: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'action.hover',
              }}
            >
              <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                拖放文件到此处或点击上传
              </Typography>
              <Typography variant="caption" color="text.secondary">
                支持 CSV, TXT 格式
              </Typography>
            </Box>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.txt"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
            />

            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                startIcon={<UploadIcon />}
                onClick={handleUploadClick}
              >
                选择文件
              </Button>
              <Button
                variant="outlined"
                startIcon={<AutoFixHighIcon />}
                onClick={generateSampleData}
              >
                生成示例数据
              </Button>
            </Stack>

            {previewData.length > 0 && (
              <Box sx={{ width: '100%' }}>
                <Typography variant="subtitle2" gutterBottom>
                  文件数据预览:
                </Typography>
                <TableContainer component={Paper} sx={{ maxHeight: 200 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>序号</TableCell>
                        <TableCell>数值</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {previewData.slice(0, 20).map((value, idx) => (
                        <TableRow key={idx}>
                          <TableCell>{idx + 1}</TableCell>
                          <TableCell>{value.toFixed(2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                {previewData.length > 20 && (
                  <Typography variant="caption" color="text.secondary">
                    显示前 20 条，共 {parseCSVInput(csvInput).length} 条记录
                  </Typography>
                )}
              </Box>
            )}
          </Stack>
        </TabPanel>

        <Box sx={{ mt: 3 }}>
          <Button
            variant="contained"
            size="large"
            fullWidth
            onClick={handleSubmit}
            disabled={parseCSVInput(csvInput).length < 4}
          >
            进行雨流计数分析
          </Button>
        </Box>
      </CardContent>
    </Card>
  )
}

export default TimeHistoryInput
