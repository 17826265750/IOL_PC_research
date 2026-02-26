import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Stack,
  TextField,
  Button,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  FolderOpen as FolderOpenIcon,
  Download as DownloadIcon,
} from '@mui/icons-material'

export interface MissionStep {
  id: string
  deltaTj: number // Temperature swing
  tjMax: number // Maximum junction temperature
  tOn: number // Heating time in seconds
  cycleCount: number // Number of cycles at this condition
  description?: string
}

interface Props {
  onProfileChange: (steps: MissionStep[]) => void
  initialSteps?: MissionStep[]
}

export const MissionProfileInput: React.FC<Props> = ({
  onProfileChange,
  initialSteps = [],
}) => {
  const [steps, setSteps] = useState<MissionStep[]>(initialSteps)
  const [editingStep, setEditingStep] = useState<MissionStep | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [loadDialogOpen, setLoadDialogOpen] = useState(false)
  const [savedProfiles, setSavedProfiles] = useState<Record<string, MissionStep[]>>(
    {}
  )
  const [profileName, setProfileName] = useState('')

  useEffect(() => {
    // Load saved profiles from localStorage
    const saved = localStorage.getItem('missionProfiles')
    if (saved) {
      try {
        setSavedProfiles(JSON.parse(saved))
      } catch (e) {
        console.error('Failed to load saved profiles', e)
      }
    }
  }, [])

  useEffect(() => {
    onProfileChange(steps)
  }, [steps, onProfileChange])

  const handleAddStep = () => {
    const newStep: MissionStep = {
      id: `step-${Date.now()}`,
      deltaTj: 80,
      tjMax: 125,
      tOn: 60,
      cycleCount: 1000,
      description: '',
    }
    setEditingStep(newStep)
    setDialogOpen(true)
  }

  const handleEditStep = (step: MissionStep) => {
    setEditingStep({ ...step })
    setDialogOpen(true)
  }

  const handleDeleteStep = (id: string) => {
    setSteps((prev) => prev.filter((s) => s.id !== id))
  }

  const handleSaveStep = () => {
    if (!editingStep) return

    if (editingStep.deltaTj <= 0 || editingStep.tjMax <= 0 || editingStep.tOn <= 0) {
      return
    }

    setSteps((prev) => {
      const existingIndex = prev.findIndex((s) => s.id === editingStep.id)
      if (existingIndex >= 0) {
        const updated = [...prev]
        updated[existingIndex] = editingStep
        return updated
      }
      return [...prev, editingStep]
    })

    setDialogOpen(false)
    setEditingStep(null)
  }

  const handleSaveProfile = () => {
    const name = profileName.trim() || `任务剖面-${new Date().toLocaleString()}`
    const updated = { ...savedProfiles, [name]: steps }
    setSavedProfiles(updated)
    localStorage.setItem('missionProfiles', JSON.stringify(updated))
    setProfileName('')
    setLoadDialogOpen(false)
  }

  const handleLoadProfile = (name: string) => {
    const profile = savedProfiles[name]
    if (profile) {
      setSteps(profile)
      setLoadDialogOpen(false)
    }
  }

  const handleDeleteProfile = (name: string) => {
    const updated = { ...savedProfiles }
    delete updated[name]
    setSavedProfiles(updated)
    localStorage.setItem('missionProfiles', JSON.stringify(updated))
  }

  const handleExportProfile = () => {
    const data = {
      name: profileName || '任务剖面',
      exportDate: new Date().toISOString(),
      steps: steps,
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `mission-profile-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  const handleImportProfile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string)
        if (data.steps && Array.isArray(data.steps)) {
          setSteps(data.steps)
          setProfileName(data.name || '')
        }
      } catch (err) {
        console.error('Failed to import profile', err)
      }
    }
    reader.readAsText(file)
  }

  const totalCycles = steps.reduce((sum, s) => sum + s.cycleCount, 0)

  return (
    <Card>
      <CardContent>
        <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">任务剖面定义</Typography>
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              size="small"
              startIcon={<FolderOpenIcon />}
              onClick={() => setLoadDialogOpen(true)}
            >
              加载
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<SaveIcon />}
              onClick={() => setDialogOpen(false)}
              onClickCapture={() => {
                if (steps.length > 0) {
                  setLoadDialogOpen(true)
                }
              }}
            >
              保存
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<DownloadIcon />}
              onClick={handleExportProfile}
              disabled={steps.length === 0}
            >
              导出
            </Button>
            <Button
              variant="outlined"
              size="small"
              component="label"
              disabled={steps.length === 0}
            >
              导入
              <input type="file" hidden accept=".json" onChange={handleImportProfile} />
            </Button>
          </Stack>
        </Stack>

        {steps.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography color="text.secondary" gutterBottom>
              尚未定义任务剖面步骤
            </Typography>
            <Button variant="contained" startIcon={<AddIcon />} onClick={handleAddStep}>
              添加步骤
            </Button>
          </Box>
        ) : (
          <>
            <TableContainer component={Paper} sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>步骤</TableCell>
                    <TableCell>ΔTj (°C)</TableCell>
                    <TableCell>Tj_max (°C)</TableCell>
                    <TableCell>t_on (s)</TableCell>
                    <TableCell>循环次数</TableCell>
                    <TableCell>说明</TableCell>
                    <TableCell align="right">操作</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {steps.map((step, index) => (
                    <TableRow key={step.id}>
                      <TableCell>{index + 1}</TableCell>
                      <TableCell>{step.deltaTj}</TableCell>
                      <TableCell>{step.tjMax}</TableCell>
                      <TableCell>{step.tOn}</TableCell>
                      <TableCell>{step.cycleCount.toLocaleString()}</TableCell>
                      <TableCell>{step.description || '-'}</TableCell>
                      <TableCell align="right">
                        <IconButton
                          size="small"
                          onClick={() => handleEditStep(step)}
                          color="primary"
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={() => handleDeleteStep(step.id)}
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

            <Stack direction="row" justifyContent="space-between" alignItems="center">
              <Typography variant="body2" color="text.secondary">
                总循环数: {totalCycles.toLocaleString()}
              </Typography>
              <Button variant="contained" startIcon={<AddIcon />} onClick={handleAddStep}>
                添加步骤
              </Button>
            </Stack>
          </>
        )}

        {/* Edit/Add Step Dialog */}
        <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>
            {editingStep && steps.find((s) => s.id === editingStep.id)
              ? '编辑步骤'
              : '添加步骤'}
          </DialogTitle>
          <DialogContent>
            <Stack spacing={2} sx={{ mt: 1 }}>
              <TextField
                label="温度变化 ΔTj (°C)"
                type="number"
                fullWidth
                value={editingStep?.deltaTj || ''}
                onChange={(e) =>
                  setEditingStep((prev) => ({ ...prev!, deltaTj: parseFloat(e.target.value) || 0 }))
                }
                inputProps={{ min: 0, step: 0.1 }}
              />
              <TextField
                label="最高结温 Tj_max (°C)"
                type="number"
                fullWidth
                value={editingStep?.tjMax || ''}
                onChange={(e) =>
                  setEditingStep((prev) => ({ ...prev!, tjMax: parseFloat(e.target.value) || 0 }))
                }
                inputProps={{ min: 0, step: 0.1 }}
              />
              <TextField
                label="加热时间 t_on (秒)"
                type="number"
                fullWidth
                value={editingStep?.tOn || ''}
                onChange={(e) =>
                  setEditingStep((prev) => ({ ...prev!, tOn: parseFloat(e.target.value) || 0 }))
                }
                inputProps={{ min: 0, step: 1 }}
              />
              <TextField
                label="循环次数"
                type="number"
                fullWidth
                value={editingStep?.cycleCount || ''}
                onChange={(e) =>
                  setEditingStep((prev) => ({
                    ...prev!,
                    cycleCount: parseInt(e.target.value) || 0,
                  }))
                }
                inputProps={{ min: 1, step: 1 }}
              />
              <TextField
                label="说明（可选）"
                fullWidth
                value={editingStep?.description || ''}
                onChange={(e) =>
                  setEditingStep((prev) => ({ ...prev!, description: e.target.value }))
                }
                placeholder="例如：启动工况、正常运行等"
              />
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDialogOpen(false)}>取消</Button>
            <Button variant="contained" onClick={handleSaveStep}>
              保存
            </Button>
          </DialogActions>
        </Dialog>

        {/* Save/Load Profile Dialog */}
        <Dialog
          open={loadDialogOpen}
          onClose={() => setLoadDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>保存/加载任务剖面</DialogTitle>
          <DialogContent>
            <Stack spacing={2}>
              <TextField
                label="剖面名称"
                fullWidth
                value={profileName}
                onChange={(e) => setProfileName(e.target.value)}
                placeholder="输入名称以保存当前剖面"
              />
              <Divider />
              <Typography variant="subtitle2">已保存的剖面:</Typography>
              {Object.keys(savedProfiles).length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  暂无保存的剖面
                </Typography>
              ) : (
                <Stack spacing={1}>
                  {Object.entries(savedProfiles).map(([name, profileSteps]) => (
                    <Paper
                      key={name}
                      variant="outlined"
                      sx={{
                        p: 1.5,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <Box>
                        <Typography variant="body2">{name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {profileSteps.length} 个步骤,{' '}
                          {profileSteps.reduce((s, st) => s + st.cycleCount, 0).toLocaleString()}{' '}
                          循环
                        </Typography>
                      </Box>
                      <Stack direction="row" spacing={0.5}>
                        <Button
                          size="small"
                          onClick={() => handleLoadProfile(name)}
                        >
                          加载
                        </Button>
                        <Button
                          size="small"
                          color="error"
                          onClick={() => handleDeleteProfile(name)}
                        >
                          删除
                        </Button>
                      </Stack>
                    </Paper>
                  ))}
                </Stack>
              )}
            </Stack>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setLoadDialogOpen(false)}>关闭</Button>
            <Button
              variant="contained"
              onClick={handleSaveProfile}
              disabled={!profileName.trim() || steps.length === 0}
            >
              保存当前剖面
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  )
}

export default MissionProfileInput
