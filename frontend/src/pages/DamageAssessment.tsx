/**
 * 功率模块寿命分析软件 - 累计损伤评估页面
 * @author GSH
 */
import React, { useState, useCallback, useEffect } from 'react'
import {
  Box,
  Container,
  Stack,
  Typography,
  Alert,
  CircularProgress,
  Paper,
  Grid,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Divider,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Calculate,
} from '@mui/icons-material'
import ReactECharts from 'echarts-for-react'
import apiService from '@/services/api'
import type { LifetimeModelType } from '@/types'

/* ────────── Model parameter configs (same structure as Prediction page) ────────── */

interface ParamDef {
  key: string
  label: string
  unit: string
  defaultValue: string
}

interface ModelCfg {
  label: string
  formula: string
  params: ParamDef[]
}

const MODEL_PARAMS_CONFIG: Record<string, ModelCfg> = {
  coffin_manson: {
    label: 'Coffin-Manson',
    formula: 'Nf = A × (ΔTj)^(-α)',
    params: [
      { key: 'A', label: 'A', unit: '', defaultValue: '3.025e14' },
      { key: 'alpha', label: 'α', unit: '', defaultValue: '5.039' },
    ],
  },
  coffin_manson_arrhenius: {
    label: 'Coffin-Manson-Arrhenius',
    formula: 'Nf = A × (ΔTj)^(-α) × exp(Ea/(kB×Tj_mean))',
    params: [
      { key: 'A', label: 'A', unit: '', defaultValue: '3.025e14' },
      { key: 'alpha', label: 'α', unit: '', defaultValue: '5.039' },
      { key: 'Ea', label: 'Ea', unit: 'eV', defaultValue: '0.8' },
    ],
  },
  norris_landzberg: {
    label: 'Norris-Landzberg',
    formula: 'Nf = A × (ΔTj)^(-α) × f^β × exp(Ea/(kB×Tj_max))',
    params: [
      { key: 'A', label: 'A', unit: '', defaultValue: '3.025e14' },
      { key: 'alpha', label: 'α', unit: '', defaultValue: '5.039' },
      { key: 'beta', label: 'β', unit: '', defaultValue: '-0.33' },
      { key: 'Ea', label: 'Ea', unit: 'eV', defaultValue: '0.8' },
      { key: 'f', label: '频率 f', unit: 'Hz', defaultValue: '6' },
    ],
  },
  cips2008: {
    label: 'CIPS 2008 (Bayerer)',
    formula: 'Nf = K×(ΔTj)^β1×exp(β2/Tj_max)×ton^β3×I^β4×V^β5×D^β6',
    params: [
      { key: 'K', label: 'K', unit: '', defaultValue: '1e17' },
      { key: 'beta1', label: 'β1', unit: '', defaultValue: '-4.423' },
      { key: 'beta2', label: 'β2', unit: 'K', defaultValue: '1285' },
      { key: 'beta3', label: 'β3', unit: '', defaultValue: '-0.462' },
      { key: 'beta4', label: 'β4', unit: '', defaultValue: '-0.716' },
      { key: 'beta5', label: 'β5', unit: '', defaultValue: '-0.761' },
      { key: 'beta6', label: 'β6', unit: '', defaultValue: '-0.5' },
    ],
  },
  lesit: {
    label: 'LESIT',
    formula: 'Nf = A × (ΔTj)^α × exp(Q/(R×Tj_min))',
    params: [
      { key: 'A', label: 'A', unit: '', defaultValue: '3.025e14' },
      { key: 'alpha', label: 'α', unit: '', defaultValue: '-5.039' },
      { key: 'Q', label: 'Q', unit: 'eV', defaultValue: '0.8' },
    ],
  },
}

/* ────────── Operating-condition row interface ────────── */

interface MissionRow {
  id: number
  Tmax: number
  Tmin: number
  theating: number  // seconds
  tcooling: number  // seconds
  I: number         // CIPS only
  V: number         // CIPS only
  D: number         // CIPS only
  count: number     // cycle count
}

interface DamageDetail {
  row: MissionRow
  nf: number
  damage: number // ni / Nfi
}

interface DamageResult {
  totalDamage: number
  blocksToFailure: number
  details: DamageDetail[]
  safetyFactor: number
}

const FITTED_PARAMS_KEY = 'cips_fitted_parameters'
const DA_MODEL_TYPE_KEY = 'da_model_type'
const DA_MODEL_PARAMS_KEY = 'da_model_params'
const DA_SAFETY_FACTOR_KEY = 'da_safety_factor'
const DA_ROWS_KEY = 'da_rows'
const DA_RESULT_KEY = 'da_result'

let rowId = 1
const makeRow = (): MissionRow => ({
  id: rowId++,
  Tmax: 125,
  Tmin: 40,
  theating: 2,
  tcooling: 2,
  I: 100,
  V: 1200,
  D: 300,
  count: 10000,
})

/* ────────── Component ────────── */

export const DamageAssessment: React.FC = () => {
  const [modelType, setModelType] = useState<string>(() => {
    return localStorage.getItem(DA_MODEL_TYPE_KEY) || 'cips2008'
  })
  const [modelParams, setModelParams] = useState<Record<string, string>>(() => {
    const saved = localStorage.getItem(DA_MODEL_PARAMS_KEY)
    if (saved) {
      try { return JSON.parse(saved) } catch { /* ignore */ }
    }
    const cfg = MODEL_PARAMS_CONFIG['cips2008']
    const d: Record<string, string> = {}
    cfg.params.forEach((p) => (d[p.key] = p.defaultValue))
    return d
  })
  const [safetyFactor, setSafetyFactor] = useState(() => {
    return localStorage.getItem(DA_SAFETY_FACTOR_KEY) || '1.0'
  })
  const [rows, setRows] = useState<MissionRow[]>(() => {
    const saved = localStorage.getItem(DA_ROWS_KEY)
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed) && parsed.length > 0) {
          rowId = Math.max(...parsed.map((r: MissionRow) => r.id), 0) + 1
          return parsed
        }
      } catch { /* ignore */ }
    }
    return [makeRow()]
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DamageResult | null>(() => {
    const saved = localStorage.getItem(DA_RESULT_KEY)
    if (saved) {
      try { return JSON.parse(saved) } catch { /* ignore */ }
    }
    return null
  })

  /* Persist state to localStorage */
  useEffect(() => { localStorage.setItem(DA_MODEL_TYPE_KEY, modelType) }, [modelType])
  useEffect(() => { localStorage.setItem(DA_MODEL_PARAMS_KEY, JSON.stringify(modelParams)) }, [modelParams])
  useEffect(() => { localStorage.setItem(DA_SAFETY_FACTOR_KEY, safetyFactor) }, [safetyFactor])
  useEffect(() => { localStorage.setItem(DA_ROWS_KEY, JSON.stringify(rows)) }, [rows])
  useEffect(() => {
    if (result) localStorage.setItem(DA_RESULT_KEY, JSON.stringify(result))
    else localStorage.removeItem(DA_RESULT_KEY)
  }, [result])

  /* Try loading fitted params on mount */
  useEffect(() => {
    const saved = localStorage.getItem(FITTED_PARAMS_KEY)
    if (saved) {
      try {
        const all = JSON.parse(saved)
        if (all[modelType]) {
          const fitted = all[modelType] as Record<string, unknown>
          const merged: Record<string, string> = { ...modelParams }
          Object.entries(fitted).forEach(([k, v]) => {
            const mapped = k.replace('β', 'beta').replace('ton', 'theating')
            merged[mapped] = String(v)
          })
          setModelParams(merged)
        }
      } catch { /* ignore */ }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleModelChange = useCallback((name: string) => {
    setModelType(name)
    const cfg = MODEL_PARAMS_CONFIG[name]
    if (cfg) {
      const d: Record<string, string> = {}
      cfg.params.forEach((p) => (d[p.key] = p.defaultValue))
      setModelParams(d)
    }
    setResult(null)
  }, [])

  const handleRowChange = useCallback(
    (id: number, field: keyof MissionRow, value: string) => {
      setRows((prev) =>
        prev.map((r) =>
          r.id === id ? { ...r, [field]: parseFloat(value) || 0 } : r,
        ),
      )
      setResult(null)
    },
    [],
  )

  const addRow = useCallback(() => {
    setRows((prev) => [...prev, makeRow()])
    setResult(null)
  }, [])

  const removeRow = useCallback((id: number) => {
    setRows((prev) => prev.filter((r) => r.id !== id))
    setResult(null)
  }, [])

  /* ── Compute Miner damage ── */

  const handleCalculate = useCallback(async () => {
    if (rows.length === 0) {
      setError('请至少添加一条工况')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const sf = parseFloat(safetyFactor) || 1.0
      const details: DamageDetail[] = []

      // Parse model params to numbers
      const numericParams: Record<string, number> = {}
      Object.entries(modelParams).forEach(([k, v]) => {
        const n = parseFloat(v)
        if (Number.isFinite(n)) numericParams[k] = n
      })

      // For each mission row, call /predict to get Nf
      for (const row of rows) {
        const apiParams: Record<string, unknown> = {
          Tmax: row.Tmax,
          Tmin: row.Tmin,
          theating: row.theating,
          tcooling: row.tcooling,
          ...numericParams,
        }
        // CIPS-specific params
        if (modelType === 'cips2008') {
          apiParams.I = row.I
          apiParams.V = row.V
          apiParams.D = row.D
        }

        const resp = await apiService.predictLifetime({
          modelType,
          params: apiParams,
          cycles: [{
            Tmax: row.Tmax,
            Tmin: row.Tmin,
            theating: row.theating,
          }],
        })

        if (!resp.success || !resp.data) {
          throw new Error(resp.error || `工况 Tmax=${row.Tmax}°C 计算失败`)
        }

        const nf = (resp.data as { predictedCycles: number }).predictedCycles
        const damage = row.count / nf

        details.push({ row, nf, damage })
      }

      const totalDamage = details.reduce((s, d) => s + d.damage, 0) * sf
      const blocksToFailure = totalDamage > 0 ? 1 / totalDamage : Infinity

      setResult({ totalDamage, blocksToFailure, details, safetyFactor: sf })
    } catch (err) {
      setError(err instanceof Error ? err.message : '计算失败')
    } finally {
      setLoading(false)
    }
  }, [rows, modelType, modelParams, safetyFactor])

  /* ── Damage pie chart ── */
  const getPieOption = () => {
    if (!result) return null
    return {
      title: { text: '各工况损伤贡献', left: 'center' },
      tooltip: { trigger: 'item', formatter: '{b}: {d}%' },
      series: [
        {
          type: 'pie',
          radius: ['40%', '65%'],
          data: result.details.map((d, i) => ({
            name: `工况${i + 1} (ΔTj=${(d.row.Tmax - d.row.Tmin).toFixed(0)}°C)`,
            value: parseFloat(d.damage.toExponential(4)),
          })),
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.5)' },
          },
          label: { formatter: '{b}\n{d}%' },
        },
      ],
    }
  }

  /* ── Health gauge ── */
  const healthIndex = result ? Math.max(0, (1 - result.totalDamage) * 100) : null

  const isCips = modelType === 'cips2008'
  const cfg = MODEL_PARAMS_CONFIG[modelType]

  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        {/* Header */}
        <Box>
          <Typography variant="h4" gutterBottom>
            累积损伤与剩余寿命评估
          </Typography>
          <Typography variant="body1" color="text.secondary">
            基于 Miner 线性累积损伤理论，输入任务剖面（多工况），选择寿命模型计算每一工况下的
            Nf，汇总 D = Σ(n_i / N_{'{f,i}'})，评估剩余寿命与健康状态。
          </Typography>
        </Box>

        {/* Model selector + params */}
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>寿命模型</Typography>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth size="small">
                <InputLabel>选择模型</InputLabel>
                <Select
                  value={modelType}
                  label="选择模型"
                  onChange={(e) => handleModelChange(e.target.value)}
                >
                  {Object.entries(MODEL_PARAMS_CONFIG).map(([k, c]) => (
                    <MenuItem key={k} value={k}>{c.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                label="安全系数 f_safe"
                size="small"
                fullWidth
                value={safetyFactor}
                onChange={(e) => setSafetyFactor(e.target.value)}
              />
            </Grid>
          </Grid>

          {cfg && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace', mb: 1 }}>
                {cfg.formula}
              </Typography>
              <Grid container spacing={2}>
                {cfg.params.map((p) => (
                  <Grid item xs={6} sm={3} key={p.key}>
                    <TextField
                      label={p.unit ? `${p.label} (${p.unit})` : p.label}
                      size="small"
                      fullWidth
                      value={modelParams[p.key] ?? p.defaultValue}
                      onChange={(e) =>
                        setModelParams((prev) => ({ ...prev, [p.key]: e.target.value }))
                      }
                    />
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
        </Paper>

        {/* Mission profile table */}
        <Paper sx={{ p: 2 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="h6">任务剖面</Typography>
            <Button startIcon={<AddIcon />} size="small" onClick={addRow}>
              添加工况
            </Button>
          </Stack>

          <TableContainer sx={{ maxHeight: 400 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Tmax (°C)</TableCell>
                  <TableCell>Tmin (°C)</TableCell>
                  <TableCell>ΔTj (°C)</TableCell>
                  <TableCell>加热时间 (s)</TableCell>
                  <TableCell>冷却时间 (s)</TableCell>
                  {isCips && <TableCell>I (A)</TableCell>}
                  {isCips && <TableCell>V (V)</TableCell>}
                  {isCips && <TableCell>D (μm)</TableCell>}
                  <TableCell>循环次数</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((r, i) => (
                  <TableRow key={r.id}>
                    <TableCell>{i + 1}</TableCell>
                    <TableCell>
                      <TextField
                        size="small" type="number" sx={{ width: 80 }}
                        value={r.Tmax}
                        onChange={(e) => handleRowChange(r.id, 'Tmax', e.target.value)}
                      />
                    </TableCell>
                    <TableCell>
                      <TextField
                        size="small" type="number" sx={{ width: 80 }}
                        value={r.Tmin}
                        onChange={(e) => handleRowChange(r.id, 'Tmin', e.target.value)}
                      />
                    </TableCell>
                    <TableCell>{(r.Tmax - r.Tmin).toFixed(1)}</TableCell>
                    <TableCell>
                      <TextField
                        size="small" type="number" sx={{ width: 80 }}
                        value={r.theating}
                        onChange={(e) => handleRowChange(r.id, 'theating', e.target.value)}
                      />
                    </TableCell>
                    <TableCell>
                      <TextField
                        size="small" type="number" sx={{ width: 80 }}
                        value={r.tcooling}
                        onChange={(e) => handleRowChange(r.id, 'tcooling', e.target.value)}
                      />
                    </TableCell>
                    {isCips && (
                      <TableCell>
                        <TextField
                          size="small" type="number" sx={{ width: 80 }}
                          value={r.I}
                          onChange={(e) => handleRowChange(r.id, 'I', e.target.value)}
                        />
                      </TableCell>
                    )}
                    {isCips && (
                      <TableCell>
                        <TextField
                          size="small" type="number" sx={{ width: 80 }}
                          value={r.V}
                          onChange={(e) => handleRowChange(r.id, 'V', e.target.value)}
                        />
                      </TableCell>
                    )}
                    {isCips && (
                      <TableCell>
                        <TextField
                          size="small" type="number" sx={{ width: 80 }}
                          value={r.D}
                          onChange={(e) => handleRowChange(r.id, 'D', e.target.value)}
                        />
                      </TableCell>
                    )}
                    <TableCell>
                      <TextField
                        size="small" type="number" sx={{ width: 100 }}
                        value={r.count}
                        onChange={(e) => handleRowChange(r.id, 'count', e.target.value)}
                      />
                    </TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => removeRow(r.id)}
                        disabled={rows.length <= 1}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button
              variant="contained"
              startIcon={loading ? <CircularProgress size={16} /> : <Calculate />}
              onClick={handleCalculate}
              disabled={loading || rows.length === 0}
              size="large"
            >
              {loading ? '计算中…' : '计算累积损伤'}
            </Button>
          </Box>
        </Paper>

        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* ── Results ── */}
        {result && (
          <>
            {/* Summary cards */}
            <Grid container spacing={2}>
              {[
                {
                  label: '累积损伤 D',
                  value: result.totalDamage.toExponential(4),
                  color: result.totalDamage >= 1 ? 'error.main' : result.totalDamage > 0.5 ? 'warning.main' : 'success.main',
                },
                {
                  label: '可承受工况块数 (1/D)',
                  value: Number.isFinite(result.blocksToFailure) ? result.blocksToFailure.toFixed(1) : '∞',
                  color: 'primary.main',
                },
                {
                  label: '健康指数',
                  value: `${healthIndex!.toFixed(1)}%`,
                  color: healthIndex! > 80 ? 'success.main' : healthIndex! > 50 ? 'warning.main' : 'error.main',
                },
                {
                  label: '安全系数',
                  value: result.safetyFactor.toFixed(2),
                  color: 'text.primary',
                },
              ].map((c) => (
                <Grid item xs={6} sm={3} key={c.label}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="text.secondary">{c.label}</Typography>
                    <Typography variant="h5" sx={{ color: c.color, fontWeight: 'bold' }}>
                      {c.value}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>

            {/* Health bar */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>健康状态</Typography>
              <LinearProgress
                variant="determinate"
                value={healthIndex!}
                color={healthIndex! > 80 ? 'success' : healthIndex! > 50 ? 'warning' : 'error'}
                sx={{ height: 20, borderRadius: 2 }}
              />
              <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
                <Typography variant="caption" color="text.secondary">失效 (D=1)</Typography>
                <Typography variant="caption" color="text.secondary">全新 (D=0)</Typography>
              </Stack>
              <Alert
                severity={
                  healthIndex! > 80 ? 'success' : healthIndex! > 50 ? 'info' : healthIndex! > 25 ? 'warning' : 'error'
                }
                sx={{ mt: 2 }}
              >
                {healthIndex! > 80 && '器件状态良好，可按常规维护计划运行。'}
                {healthIndex! > 50 && healthIndex! <= 80 && '器件状态正常，建议加强监测。'}
                {healthIndex! > 25 && healthIndex! <= 50 && '器件已有明显损伤，建议尽快安排维护。'}
                {healthIndex! <= 25 && '器件接近寿命终点，建议立即更换。'}
              </Alert>
            </Paper>

            {/* Detail table */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>各工况损伤明细</Typography>
              <TableContainer sx={{ maxHeight: 400 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>#</TableCell>
                      <TableCell>ΔTj (°C)</TableCell>
                      <TableCell>Tmax (°C)</TableCell>
                      <TableCell>Tmin (°C)</TableCell>
                      <TableCell>循环次数 n_i</TableCell>
                      <TableCell>Nf</TableCell>
                      <TableCell>损伤 n_i / Nf</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.details.map((d, i) => (
                      <TableRow key={i}>
                        <TableCell>{i + 1}</TableCell>
                        <TableCell>{(d.row.Tmax - d.row.Tmin).toFixed(1)}</TableCell>
                        <TableCell>{d.row.Tmax}</TableCell>
                        <TableCell>{d.row.Tmin}</TableCell>
                        <TableCell>{d.row.count}</TableCell>
                        <TableCell>{d.nf.toExponential(3)}</TableCell>
                        <TableCell>{d.damage.toExponential(4)}</TableCell>
                      </TableRow>
                    ))}
                    <TableRow sx={{ '& td': { fontWeight: 'bold', borderTop: 2 } }}>
                      <TableCell colSpan={4}>合计</TableCell>
                      <TableCell>{result.details.reduce((s, d) => s + d.row.count, 0)}</TableCell>
                      <TableCell>—</TableCell>
                      <TableCell>{result.totalDamage.toExponential(4)}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>

            {/* Pie chart */}
            {result.details.length > 1 && (
              <Paper sx={{ p: 2 }}>
                <ReactECharts option={getPieOption()!} style={{ height: 320 }} />
              </Paper>
            )}

            {/* Remaining life estimate */}
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>剩余寿命估算</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      剩余可承受工况块数
                    </Typography>
                    <Typography variant="h5">
                      {Number.isFinite(result.blocksToFailure) ? result.blocksToFailure.toFixed(1) : '∞'}
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      剩余循环数（估算）
                    </Typography>
                    <Typography variant="h5">
                      {Number.isFinite(result.blocksToFailure)
                        ? (
                            result.blocksToFailure *
                            result.details.reduce((s, d) => s + d.row.count, 0)
                          ).toExponential(3)
                        : '∞'}
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      剩余运行时间（年）
                    </Typography>
                    <Typography variant="h5">
                      {Number.isFinite(result.blocksToFailure)
                        ? (() => {
                            const totalCycleTime = result.details.reduce(
                              (s, d) => s + d.row.count * (d.row.theating + d.row.tcooling),
                              0,
                            )
                            const blockHours = totalCycleTime / 3600
                            return (result.blocksToFailure * blockHours / 8760).toFixed(2)
                          })()
                        : '∞'}
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </Paper>

            <Box sx={{ textAlign: 'center' }}>
              <Button variant="outlined" onClick={() => setResult(null)}>
                重新计算
              </Button>
            </Box>
          </>
        )}

        {/* Theory section */}
        <Paper sx={{ p: 2, bgcolor: 'info.main', color: 'info.contrastText' }}>
          <Typography variant="h6" gutterBottom>关于 Miner 线性累积损伤</Typography>
          <Typography variant="body2" paragraph>
            Miner 理论假设每个循环造成的损伤独立且可累加，总损伤 D = Σ (n_i / N_{'f,i'})。
            当 D ≥ 1 时器件发生失效。每一工况条件下的 N_{'f,i'} 由所选寿命模型计算。
          </Typography>
          <Typography variant="body2">
            安全系数 f_safe 乘以 D 后得到 D_safe，一般取 1.0~2.0。
          </Typography>
        </Paper>
      </Stack>
    </Container>
  )
}

export default DamageAssessment
