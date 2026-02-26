import React, { useState, useEffect } from 'react'
import {
  Box,
  Grid,
  TextField,
  Typography,
  InputAdornment,
  Tooltip,
  Paper,
  Collapse,
  Alert,
} from '@mui/material'
import {
  HelpOutline,
  ExpandMore,
  ExpandLess,
  Science,
  Thermostat,
  ElectricalServices,
} from '@mui/icons-material'
import type { LifetimeModelType } from '@/types'

interface ParameterField {
  name: string
  label: string
  labelEn: string
  unit: string
  type: 'number'
  defaultValue: number
  min?: number
  max?: number
  step?: number
  tooltip: string
  tooltipEn: string
}

// Model parameter definitions
const MODEL_PARAMETERS: Record<LifetimeModelType, ParameterField[]> = {
  coffin_manson: [
    {
      name: 'A',
      label: '模型常数',
      labelEn: 'Model Constant (A)',
      unit: '',
      type: 'number',
      defaultValue: 100000,
      min: 1,
      step: 1000,
      tooltip: 'Coffin-Manson模型常数，影响整体寿命预测值',
      tooltipEn: 'Coffin-Manson model constant affecting overall lifetime prediction',
    },
    {
      name: 'n',
      label: '温度摆动指数',
      labelEn: 'Temperature Swing Exponent (n)',
      unit: '',
      type: 'number',
      defaultValue: 1.5,
      min: 0.1,
      max: 5,
      step: 0.1,
      tooltip: '温度摆动对寿命的影响指数',
      tooltipEn: 'Exponent for temperature swing effect on lifetime',
    },
    {
      name: 'm',
      label: '时间指数',
      labelEn: 'Time Exponent (m)',
      unit: '',
      type: 'number',
      defaultValue: 0.5,
      min: 0,
      max: 2,
      step: 0.1,
      tooltip: '加热时间对寿命的影响指数',
      tooltipEn: 'Exponent for heating time effect on lifetime',
    },
  ],
  coffin_manson_arrhenius: [
    {
      name: 'A',
      label: '模型常数',
      labelEn: 'Model Constant (A)',
      unit: '',
      type: 'number',
      defaultValue: 100000,
      min: 1,
      step: 1000,
      tooltip: '模型常数',
      tooltipEn: 'Model constant',
    },
    {
      name: 'n',
      label: '温度摆动指数',
      labelEn: 'Temperature Swing Exponent (n)',
      unit: '',
      type: 'number',
      defaultValue: 1.5,
      min: 0.1,
      max: 5,
      step: 0.1,
      tooltip: '温度摆动指数',
      tooltipEn: 'Temperature swing exponent',
    },
    {
      name: 'm',
      label: '时间指数',
      labelEn: 'Time Exponent (m)',
      unit: '',
      type: 'number',
      defaultValue: 0.5,
      min: 0,
      max: 2,
      step: 0.1,
      tooltip: '时间指数',
      tooltipEn: 'Time exponent',
    },
    {
      name: 'Ea',
      label: '激活能',
      labelEn: 'Activation Energy (Ea)',
      unit: 'eV',
      type: 'number',
      defaultValue: 0.8,
      min: 0.1,
      max: 2,
      step: 0.05,
      tooltip: 'Arrhenius激活能，描述温度依赖性',
      tooltipEn: 'Arrhenius activation energy for temperature dependence',
    },
    {
      name: 'Tjmax',
      label: '最高结温',
      labelEn: 'Max Junction Temperature',
      unit: '°C',
      type: 'number',
      defaultValue: 125,
      min: -55,
      max: 200,
      step: 1,
      tooltip: '最高结温',
      tooltipEn: 'Maximum junction temperature',
    },
  ],
  norris_landzberg: [
    {
      name: 'A',
      label: '模型常数',
      labelEn: 'Model Constant (A)',
      unit: '',
      type: 'number',
      defaultValue: 100000,
      min: 1,
      step: 1000,
      tooltip: '模型常数',
      tooltipEn: 'Model constant',
    },
    {
      name: 'n',
      label: '温度摆动指数',
      labelEn: 'Temperature Swing Exponent (n)',
      unit: '',
      type: 'number',
      defaultValue: 2,
      min: 0.1,
      max: 5,
      step: 0.1,
      tooltip: '温度摆动指数',
      tooltipEn: 'Temperature swing exponent',
    },
    {
      name: 'k',
      label: '频率指数',
      labelEn: 'Frequency Exponent (k)',
      unit: '',
      type: 'number',
      defaultValue: 0.33,
      min: 0,
      max: 2,
      step: 0.05,
      tooltip: '频率对寿命的影响指数',
      tooltipEn: 'Exponent for frequency effect on lifetime',
    },
    {
      name: 'Ea',
      label: '激活能',
      labelEn: 'Activation Energy (Ea)',
      unit: 'eV',
      type: 'number',
      defaultValue: 0.8,
      min: 0.1,
      max: 2,
      step: 0.05,
      tooltip: '激活能',
      tooltipEn: 'Activation energy',
    },
    {
      name: 'Tjmax',
      label: '最高结温',
      labelEn: 'Max Junction Temperature',
      unit: '°C',
      type: 'number',
      defaultValue: 125,
      min: -55,
      max: 200,
      step: 1,
      tooltip: '最高结温',
      tooltipEn: 'Maximum junction temperature',
    },
    {
      name: 'f',
      label: '循环频率',
      labelEn: 'Cycle Frequency',
      unit: 'Hz',
      type: 'number',
      defaultValue: 0.001,
      min: 0.0001,
      max: 1,
      step: 0.0001,
      tooltip: '循环频率',
      tooltipEn: 'Cycle frequency',
    },
  ],
  cips2008: [
    {
      name: 'K',
      label: '模型常数 (K)',
      labelEn: 'Model Constant (K)',
      unit: '',
      type: 'number',
      defaultValue: 1e17,
      min: 1e10,
      step: 1e14,
      tooltip: 'CIPS 2008模型常数（需根据具体器件拟合）',
      tooltipEn: 'CIPS 2008 model constant (needs fitting for specific device)',
    },
    {
      name: 'beta1',
      label: 'ΔTj指数 (β1)',
      labelEn: 'ΔTj Exponent (β1)',
      unit: '',
      type: 'number',
      defaultValue: -4.423,
      min: -10,
      max: 0,
      step: 0.01,
      tooltip: '温度摆动指数（负值表示ΔTj越大寿命越短）',
      tooltipEn: 'Temperature swing exponent (negative means larger ΔTj reduces lifetime)',
    },
    {
      name: 'beta2',
      label: '温度系数 (β2)',
      labelEn: 'Temperature Coefficient (β2)',
      unit: 'K',
      type: 'number',
      defaultValue: 1285,
      min: 0,
      max: 5000,
      step: 10,
      tooltip: 'Arrhenius温度系数（典型值: 1285 K）',
      tooltipEn: 'Arrhenius temperature coefficient (typical: 1285 K)',
    },
    {
      name: 'beta3',
      label: '时间指数 (β3)',
      labelEn: 'Time Exponent (β3)',
      unit: '',
      type: 'number',
      defaultValue: -0.462,
      min: -2,
      max: 0,
      step: 0.01,
      tooltip: '加热时间指数（负值表示加热时间越长寿命越短，典型值: -0.462）',
      tooltipEn: 'Heating time exponent (negative, typical: -0.462)',
    },
    {
      name: 'beta4',
      label: '电流指数 (β4)',
      labelEn: 'Current Exponent (β4)',
      unit: '',
      type: 'number',
      defaultValue: -0.716,
      min: -2,
      max: 0,
      step: 0.01,
      tooltip: '电流影响指数（典型值: -0.716）',
      tooltipEn: 'Current effect exponent (typical: -0.716)',
    },
    {
      name: 'beta5',
      label: '电压指数 (β5)',
      labelEn: 'Voltage Exponent (β5)',
      unit: '',
      type: 'number',
      defaultValue: -0.761,
      min: -2,
      max: 0,
      step: 0.01,
      tooltip: '电压影响指数（典型值: -0.761）',
      tooltipEn: 'Voltage effect exponent (typical: -0.761)',
    },
    {
      name: 'beta6',
      label: '键合线直径指数 (β6)',
      labelEn: 'Bond Wire Diameter Exponent (β6)',
      unit: '',
      type: 'number',
      defaultValue: -0.5,
      min: -2,
      max: 0,
      step: 0.01,
      tooltip: '键合线直径影响指数（典型值: -0.5）',
      tooltipEn: 'Bond wire diameter effect exponent (typical: -0.5)',
    },
  ],
  lesit: [
    {
      name: 'A',
      label: '模型常数',
      labelEn: 'Model Constant (A)',
      unit: '',
      type: 'number',
      defaultValue: 100000,
      min: 1,
      step: 1000,
      tooltip: '模型常数',
      tooltipEn: 'Model constant',
    },
    {
      name: 'n',
      label: '温度摆动指数',
      labelEn: 'Temperature Swing Exponent (n)',
      unit: '',
      type: 'number',
      defaultValue: 1.3,
      min: 0.1,
      max: 5,
      step: 0.1,
      tooltip: '温度摆动指数',
      tooltipEn: 'Temperature swing exponent',
    },
    {
      name: 'm',
      label: '时间指数',
      labelEn: 'Time Exponent (m)',
      unit: '',
      type: 'number',
      defaultValue: 0.4,
      min: 0,
      max: 2,
      step: 0.1,
      tooltip: '时间指数',
      tooltipEn: 'Time exponent',
    },
    {
      name: 'Ea',
      label: '激活能 (Q)',
      labelEn: 'Activation Energy (Q)',
      unit: 'eV',
      type: 'number',
      defaultValue: 0.5,
      min: 0.1,
      max: 2,
      step: 0.05,
      tooltip: '激活能',
      tooltipEn: 'Activation energy',
    },
    {
      name: 'Tjmax',
      label: '最高结温',
      labelEn: 'Max Junction Temperature',
      unit: '°C',
      type: 'number',
      defaultValue: 125,
      min: -55,
      max: 200,
      step: 1,
      tooltip: '最高结温',
      tooltipEn: 'Maximum junction temperature',
    },
    {
      name: 'Tjmin',
      label: '最低结温',
      labelEn: 'Min Junction Temperature',
      unit: '°C',
      type: 'number',
      defaultValue: 40,
      min: -55,
      max: 200,
      step: 1,
      tooltip: '最低结温',
      tooltipEn: 'Minimum junction temperature',
    },
  ],
}

// Cycle parameters (common to all models)
const CYCLE_PARAMETERS: ParameterField[] = [
  {
    name: 'Tmax',
    label: '最高温度',
    labelEn: 'Max Temperature (Tmax)',
    unit: '°C',
    type: 'number',
    defaultValue: 125,
    min: -55,
    max: 250,
    step: 1,
    tooltip: '循环周期内的最高温度',
    tooltipEn: 'Maximum temperature during cycle',
  },
  {
    name: 'Tmin',
    label: '最低温度',
    labelEn: 'Min Temperature (Tmin)',
    unit: '°C',
    type: 'number',
    defaultValue: 40,
    min: -55,
    max: 250,
    step: 1,
    tooltip: '循环周期内的最低温度',
    tooltipEn: 'Minimum temperature during cycle',
  },
  {
    name: 'theating',
    label: '加热时间',
    labelEn: 'Heating Time',
    unit: 's',
    type: 'number',
    defaultValue: 2,
    min: 0.1,
    max: 60,
    step: 0.1,
    tooltip: '加热阶段持续时间（典型值: 1-10秒）',
    tooltipEn: 'Duration of heating phase (typical: 1-10s)',
  },
  {
    name: 'tcooling',
    label: '冷却时间',
    labelEn: 'Cooling Time',
    unit: 's',
    type: 'number',
    defaultValue: 2,
    min: 0.1,
    max: 60,
    step: 0.1,
    tooltip: '冷却阶段持续时间',
    tooltipEn: 'Duration of cooling phase',
  },
]

// CIPS 2008 additional operating parameters
const CIPS_OPERATING_PARAMETERS: ParameterField[] = [
  {
    name: 'I',
    label: '负载电流',
    labelEn: 'Load Current (I)',
    unit: 'A',
    type: 'number',
    defaultValue: 100,
    min: 1,
    max: 1000,
    step: 1,
    tooltip: '负载电流（范围: 1-1000A）',
    tooltipEn: 'Load current (range: 1-1000A)',
  },
  {
    name: 'V',
    label: '阻断电压',
    labelEn: 'Blocking Voltage (V)',
    unit: 'V',
    type: 'number',
    defaultValue: 1200,
    min: 1,
    max: 4000,
    step: 50,
    tooltip: '器件阻断电压等级（范围: 1-4000V）',
    tooltipEn: 'Device blocking voltage rating (range: 1-4000V)',
  },
  {
    name: 'D',
    label: '键合线直径',
    labelEn: 'Bond Wire Diameter (D)',
    unit: 'μm',
    type: 'number',
    defaultValue: 300,
    min: 100,
    max: 400,
    step: 10,
    tooltip: '键合线直径（典型值: 100-400μm）',
    tooltipEn: 'Bond wire diameter (typical: 100-400μm)',
  },
]

interface ParameterInputProps {
  modelType: LifetimeModelType
  values: Record<string, number>
  onChange: (values: Record<string, number>) => void
  disabled?: boolean
}

interface ParameterGroup {
  title: string
  titleEn: string
  icon: React.ElementType
  fields: ParameterField[]
}

export const ParameterInput: React.FC<ParameterInputProps> = ({
  modelType,
  values,
  onChange,
  disabled,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(true)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Initialize default values when model changes
  useEffect(() => {
    const modelParams = MODEL_PARAMETERS[modelType]
    const defaults: Record<string, number> = {}

    modelParams.forEach((field) => {
      defaults[field.name] = field.defaultValue
    })

    CYCLE_PARAMETERS.forEach((field) => {
      defaults[field.name] = field.defaultValue
    })

    if (modelType === 'cips2008') {
      CIPS_OPERATING_PARAMETERS.forEach((field) => {
        defaults[field.name] = field.defaultValue
      })
    }

    onChange(defaults)
  }, [modelType, onChange])

  const validateField = (field: ParameterField, value: number): string | null => {
    if (field.min !== undefined && value < field.min) {
      return `值不能小于 ${field.min}`
    }
    if (field.max !== undefined && value > field.max) {
      return `值不能大于 ${field.max}`
    }
    return null
  }

  const handleFieldChange = (field: ParameterField, value: string) => {
    const numValue = parseFloat(value)
    if (isNaN(numValue)) return

    const error = validateField(field, numValue)
    setErrors((prev) => ({
      ...prev,
      [field.name]: error || '',
    }))

    onChange({ ...values, [field.name]: numValue })
  }

  const renderField = (field: ParameterField) => {
    const value = values[field.name] ?? field.defaultValue
    const error = errors[field.name]

    return (
      <Grid item xs={12} sm={6} md={4} key={field.name}>
        <TextField
          fullWidth
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <span>{field.label}</span>
              <Tooltip
                title={
                  <Box sx={{ fontSize: '0.75rem' }}>
                    <Typography variant="caption" display="block">
                      {field.tooltip}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block">
                      {field.tooltipEn}
                    </Typography>
                  </Box>
                }
              >
                <HelpOutline sx={{ fontSize: '0.9rem' }} />
              </Tooltip>
            </Box>
          }
          value={value}
          onChange={(e) => handleFieldChange(field, e.target.value)}
          type={field.type}
          inputProps={{
            min: field.min,
            max: field.max,
            step: field.step,
          }}
          InputProps={{
            endAdornment: field.unit ? (
              <InputAdornment position="end">
                <Typography variant="caption" color="text.secondary">
                  {field.unit}
                </Typography>
              </InputAdornment>
            ) : undefined,
          }}
          disabled={disabled}
          error={!!error}
          helperText={error || `${field.labelEn}`}
          size="small"
        />
      </Grid>
    )
  }

  const parameterGroups: ParameterGroup[] = [
    {
      title: '模型参数',
      titleEn: 'Model Parameters',
      icon: Science,
      fields: MODEL_PARAMETERS[modelType],
    },
    {
      title: '循环条件',
      titleEn: 'Cycle Conditions',
      icon: Thermostat,
      fields: CYCLE_PARAMETERS,
    },
  ]

  if (modelType === 'cips2008') {
    parameterGroups.push({
      title: '工作参数',
      titleEn: 'Operating Parameters',
      icon: ElectricalServices,
      fields: CIPS_OPERATING_PARAMETERS,
    })
  }

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
        }}
      >
        <Typography variant="h6">参数设置 / Parameters</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" color="text.secondary">
            ΔT = {values.Tmax || 0} - {values.Tmin || 0} = {(values.Tmax || 0) - (values.Tmin || 0)}°C
          </Typography>
        </Box>
      </Box>

      {parameterGroups.map((group, groupIndex) => {
        const Icon = group.icon
        return (
          <Paper
            key={groupIndex}
            sx={{
              mb: 2,
              border: 1,
              borderColor: 'divider',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                p: 1.5,
                backgroundColor: 'action.hover',
                cursor: 'pointer',
                userSelect: 'none',
              }}
              onClick={() => {
                if (groupIndex === 0) setShowAdvanced(!showAdvanced)
              }}
            >
              <Icon color="primary" sx={{ mr: 1 }} />
              <Typography variant="subtitle1" sx={{ flex: 1 }}>
                {group.title}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                {group.titleEn}
              </Typography>
              {groupIndex === 0 && (showAdvanced ? <ExpandLess /> : <ExpandMore />)}
            </Box>

            <Collapse in={showAdvanced}>
              <Box sx={{ p: 2 }}>
                <Grid container spacing={2}>
                  {group.fields.map(renderField)}
                </Grid>
              </Box>
            </Collapse>
          </Paper>
        )
      })}

      <Alert severity="info" sx={{ mt: 2 }}>
        <Typography variant="body2">
          提示: 所有参数都会影响预测结果。使用"推荐"标签的模型可获得更准确的预测。
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Tip: All parameters affect prediction results. Use recommended models for better accuracy.
        </Typography>
      </Alert>
    </Box>
  )
}

export default ParameterInput
