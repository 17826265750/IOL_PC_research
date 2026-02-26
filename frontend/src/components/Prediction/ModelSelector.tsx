import React from 'react'
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
} from '@mui/material'
import { TrendingUp, Functions } from '@mui/icons-material'
import type { LifetimeModelType } from '@/types'

interface ModelInfo {
  id: LifetimeModelType
  name: string
  nameEn: string
  description: string
  descriptionEn: string
  equation: string
  category: 'basic' | 'advanced' | 'recommended'
}

const MODEL_INFO: ModelInfo[] = [
  {
    id: 'coffin_manson',
    name: 'Coffin-Manson',
    nameEn: 'Coffin-Manson',
    description: '基础热疲劳模型',
    descriptionEn: 'Basic thermal fatigue model',
    equation: 'Nf = A × (ΔTj)^(-α)',
    category: 'basic',
  },
  {
    id: 'coffin_manson_arrhenius',
    name: 'Coffin-Manson-Arrhenius',
    nameEn: 'Coffin-Manson-Arrhenius',
    description: '含Arrhenius温度依赖',
    descriptionEn: 'With Arrhenius temperature dependence',
    equation: 'Nf = A × (ΔTj)^(-α) × exp(Ea/(kB×Tj_mean))',
    category: 'advanced',
  },
  {
    id: 'norris_landzberg',
    name: 'Norris-Landzberg',
    nameEn: 'Norris-Landzberg',
    description: '含频率因子',
    descriptionEn: 'With frequency factor',
    equation: 'Nf = A × (ΔTj)^(-α) × f^β × exp(Ea/(kB×Tj_max))',
    category: 'advanced',
  },
  {
    id: 'cips2008',
    name: 'CIPS 2008 (Bayerer)',
    nameEn: 'CIPS 2008 (Bayerer)',
    description: '综合模型（推荐）',
    descriptionEn: 'Comprehensive model (Recommended)',
    equation: 'Nf = K × (ΔTj)^β1 × exp(β2/Tj_max) × t_on^β3 × I^β4 × V^β5 × D^β6',
    category: 'recommended',
  },
  {
    id: 'lesit',
    name: 'LESIT',
    nameEn: 'LESIT',
    description: 'LESIT项目模型',
    descriptionEn: 'LESIT project model',
    equation: 'Nf = A × (ΔTj)^α × exp(Q/(R×Tj_min))',
    category: 'basic',
  },
]

interface ModelSelectorProps {
  value: LifetimeModelType
  onChange: (model: LifetimeModelType) => void
  disabled?: boolean
}

const getCategoryColor = (category: ModelInfo['category']) => {
  switch (category) {
    case 'recommended':
      return 'success'
    case 'advanced':
      return 'primary'
    case 'basic':
    default:
      return 'default'
  }
}

const getCategoryLabel = (category: ModelInfo['category']) => {
  switch (category) {
    case 'recommended':
      return '推荐'
    case 'advanced':
      return '高级'
    case 'basic':
    default:
      return '基础'
  }
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ value, onChange, disabled }) => {
  const handleChange = (event: SelectChangeEvent<LifetimeModelType>) => {
    onChange(event.target.value as LifetimeModelType)
  }

  const selectedModel = MODEL_INFO.find((m) => m.id === value)

  return (
    <Box>
      <FormControl fullWidth disabled={disabled}>
        <InputLabel id="model-selector-label">选择模型 / Select Model</InputLabel>
        <Select
          labelId="model-selector-label"
          value={value}
          label="选择模型 / Select Model"
          onChange={handleChange}
          startAdornment={<Functions sx={{ mr: 1, color: 'primary.main' }} />}
        >
          {MODEL_INFO.map((model) => (
            <MenuItem key={model.id} value={model.id}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                <Typography variant="body2" sx={{ flex: 1 }}>
                  {model.name}
                </Typography>
                <Chip
                  label={getCategoryLabel(model.category)}
                  size="small"
                  color={getCategoryColor(model.category) as any}
                  variant={model.category === 'recommended' ? 'filled' : 'outlined'}
                />
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {selectedModel && (
        <Card
          sx={{
            mt: 2,
            border: 1,
            borderColor: 'divider',
            backgroundColor: selectedModel.category === 'recommended' ? 'success.50' : 'background.paper',
          }}
        >
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <TrendingUp color="primary" fontSize="small" />
              <Typography variant="h6">{selectedModel.name}</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto' }}>
                {selectedModel.nameEn}
              </Typography>
            </Box>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                {selectedModel.description}
              </Typography>
              <Typography variant="caption" color="text.disabled">
                {selectedModel.descriptionEn}
              </Typography>
            </Box>

            <Box
              sx={{
                p: 1.5,
                backgroundColor: 'background.paper',
                borderRadius: 1,
                border: 1,
                borderColor: 'divider',
              }}
            >
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                方程 / Equation:
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  fontFamily: 'monospace',
                  fontWeight: 500,
                  color: 'primary.main',
                  wordBreak: 'break-all',
                }}
              >
                {selectedModel.equation}
              </Typography>
            </Box>

            {selectedModel.category === 'recommended' && (
              <Box sx={{ mt: 1.5 }}>
                <Chip
                  icon={<TrendingUp />}
                  label="推荐模型: 综合考虑多种因素，预测精度更高"
                  color="success"
                  size="small"
                  variant="outlined"
                />
              </Box>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  )
}

export default ModelSelector
