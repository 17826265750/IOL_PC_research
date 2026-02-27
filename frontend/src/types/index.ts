/**
 * Lifetime Model Types
 * Defines all interfaces for lifetime prediction models and related data structures
 */

// ============================================
// Lifetime Model Parameters
// ============================================

/**
 * Coffin-Manson Model Parameters
 * N_f = A * (ΔT)^(-n) * (t_on)^(-m) * exp(E_a / (k * T_jmax))
 */
export interface CoffinMansonParams {
  /** Model constant */
  A: number
  /** Temperature swing exponent */
  n: number
  /** Time exponent */
  m: number
  /** Activation energy in eV */
  Ea?: number
  /** Maximum junction temperature in Kelvin */
  Tjmax?: number
  /** Boltzmann constant (8.617e-5 eV/K) - optional, uses default if not provided */
  k?: number
}

/**
 * Coffin-Manson-Arrhenius Model Parameters
 * Enhanced model combining Coffin-Manson with Arrhenius temperature dependence
 */
export interface CoffinMansonArrheniusParams {
  /** Model constant */
  A: number
  /** Temperature swing exponent */
  n: number
  /** Time exponent */
  m: number
  /** Activation energy in eV */
  Ea: number
  /** Maximum junction temperature in Kelvin */
  Tjmax: number
  /** Minimum junction temperature in Kelvin */
  Tjmin?: number
}

/**
 * Norris-Landzberg Model Parameters
 * N_f = A * (ΔT)^(-n) * (f)^(-k) * exp(E_a / (k * T_jmax))
 */
export interface NorrisLandzbergParams {
  /** Model constant */
  A: number
  /** Temperature swing exponent */
  n: number
  /** Frequency exponent */
  k: number
  /** Activation energy in eV */
  Ea: number
  /** Maximum junction temperature in Kelvin */
  Tjmax: number
  /** Minimum junction temperature in Kelvin */
  Tjmin?: number
  /** Frequency in Hz */
  f?: number
}

/**
 * CIPS 2008 Model Parameters
 * LESIT institute model with specific solder parameters
 */
export interface CIPS2008Params {
  /** Model constant */
  A: number
  /** Temperature swing exponent */
  n: number
  /** Time exponent */
  m: number
  /** Activation energy in eV */
  Ea: number
  /** Maximum junction temperature in Kelvin */
  Tjmax: number
  /** Heating time in seconds */
  theating?: number
  /** Reference temperature in Kelvin */
  Tref?: number
}

/**
 * LESIT Model Parameters
 * Alternative lifetime model from LESIT project
 */
export interface LESITParams {
  /** Model constant */
  A: number
  /** Temperature swing exponent */
  n: number
  /** Time exponent */
  m: number
  /** Activation energy in eV */
  Ea: number
  /** Maximum junction temperature in Kelvin */
  Tjmax: number
  /** Minimum junction temperature in Kelvin */
  Tjmin: number
  /** Cycle period in seconds */
  period?: number
}

// ============================================
// Model Type Union
// ============================================

export type LifetimeModelType =
  | 'coffin_manson'
  | 'coffin_manson_arrhenius'
  | 'norris_landzberg'
  | 'cips2008'
  | 'lesit'

export type LifetimeModelParams =
  | CoffinMansonParams
  | CoffinMansonArrheniusParams
  | NorrisLandzbergParams
  | CIPS2008Params
  | LESITParams

// ============================================
// Prediction Request/Response
// ============================================

/**
 * Temperature cycle data for prediction
 */
export interface TemperatureCycle {
  /** Maximum temperature in °C */
  Tmax: number
  /** Minimum temperature in °C */
  Tmin: number
  /** Heating time in seconds */
  theating: number
  /** Cooling time in seconds */
  tcooling?: number
  /** Cycle period in seconds */
  period?: number
  /** Frequency in Hz */
  frequency?: number
}

/**
 * Lifetime prediction request
 */
export interface PredictionRequest {
  /** Model type to use for prediction */
  modelType: LifetimeModelType
  /** Model parameters */
  params: LifetimeModelParams
  /** Temperature cycle data */
  cycles: TemperatureCycle[]
  /** Target number of cycles (optional) */
  targetCycles?: number
}

/**
 * Lifetime prediction result
 */
export interface PredictionResult {
  /** Model type used */
  modelType: LifetimeModelType
  /** Predicted number of cycles to failure */
  predictedCycles: number
  /** Lifetime in hours */
  lifetimeHours: number
  /** Confidence interval lower bound (optional) */
  confidenceLower?: number
  /** Confidence interval upper bound (optional) */
  confidenceUpper?: number
  /** Individual cycle predictions */
  cycleResults: CycleResult[]
  /** Calculation timestamp */
  timestamp: string
}

/**
 * Individual cycle prediction result
 */
export interface CycleResult {
  /** Cycle index */
  index: number
  /** Temperature swing in °C */
  deltaT: number
  /** Predicted cycles to failure for this cycle */
  cyclesToFailure: number
  /** Damage per cycle */
  damagePerCycle: number
}

// ============================================
// Experiment Data
// ============================================

/**
 * Power cycling experiment condition
 */
export interface ExperimentCondition {
  /** Test ID */
  testId: string
  /** Device type/part number */
  deviceType: string
  /** Maximum junction temperature in °C */
  Tjmax: number
  /** Minimum junction temperature in °C */
  Tjmin: number
  /** Heating time in seconds */
  theating: number
  /** Cooling time in seconds */
  tcooling: number
  /** Current during heating in Amperes */
  Iheating: number
  /** Voltage across device in Volts */
  Vce?: number
  /** Gate voltage in Volts */
  Vge?: number
  /** Number of cycles to failure */
  Nf: number
  /** Failure criterion (e.g., Vce increase %) */
  failureCriterion: string
  /** Test status */
  status: 'completed' | 'running' | 'failed' | 'pending'
  /** Additional notes */
  notes?: string
}

/**
 * Experiment data file
 */
export interface ExperimentData {
  /** Data file ID */
  id: string
  /** File name */
  fileName: string
  /** Upload timestamp */
  uploadDate: string
  /** Number of test conditions in file */
  conditionCount: number
  /** Test conditions */
  conditions: ExperimentCondition[]
  /** File metadata */
  metadata?: {
    /** Device manufacturer */
    manufacturer?: string
    /** Package type */
    packageType?: string
    /** Chip technology */
    technology?: string
  }
}

// ============================================
// Rainflow Cycle Data
// ============================================

/**
 * Rainflow counted cycle
 */
export interface RainflowCycle {
  /** Range of the cycle (peak to peak) */
  range: number
  /** Mean value of the cycle */
  mean: number
  /** Number of cycles at this range and mean */
  count: number
  /** Cycle type: full or half */
  cycleType?: 'full' | 'half'
}

/**
 * Rainflow counting result
 */
export interface RainflowResult {
  /** Original data points */
  originalData: number[]
  /** Extracted cycles */
  cycles: RainflowCycle[]
  /** Total number of cycles */
  totalCycles: number
  /** Maximum range */
  maxRange: number
  /** Minimum range */
  minRange: number
  /** Number of bins used for histogram */
  binCount?: number
  /** Backend summary statistics */
  summary?: Record<string, unknown>
  /** Aggregated matrix rows from pipeline */
  matrixRows?: Array<{ delta_tj: number; mean_tj: number; count: number }>
  /** Optional Miner damage output */
  damage?: Record<string, unknown> | null
  /** Thermal design summary (Tj_max, Tj_min, …) */
  thermalSummary?: {
    tj_max: number
    tj_min: number
    tj_mean: number
    tj_range: number
    delta_tj_max: number
  } | null
  /** From-To transition matrix */
  fromToMatrix?: {
    matrix: number[][]
    band_values: number[]
    n_band: number
    y_min: number
    y_max: number
  } | null
  /** Amplitude distribution histogram */
  amplitudeHistogram?: {
    bin_centers: number[]
    counts_full: number[]
    counts_half: number[]
    counts_total: number[]
    bin_edges: number[]
  } | null
  /** Residual reversal points */
  residual?: number[] | null
  /** Multi-source: Tj series per node  {'IGBT': [...], 'Diode': [...]} */
  allJunctionTemperatures?: Record<string, number[]> | null
  /** Model-based CDI result with per-cycle details */
  modelDamage?: {
    total_damage_per_block: number
    blocks_to_failure: number | null
    safety_factor: number
    model_used: string
    cycle_details: Array<{
      delta_tj: number
      mean_tj: number
      count: number
      nf: number
      damage: number
    }>
  } | null
}

// ============================================
// Damage Accumulation Data
// ============================================

/**
 * Damage accumulation entry
 */
export interface DamageEntry {
  /** Cycle range or index */
  cycleIndex: number
  /** Number of cycles at this condition */
  cycles: number
  /** Cycles to failure for this condition */
  cyclesToFailure: number
  /** Damage contribution (cycles / cyclesToFailure) */
  damage: number
  /** Cumulative damage */
  cumulativeDamage: number
}

/**
 * Damage accumulation result
 */
export interface DamageAccumulationResult {
  /** Model type used */
  modelType: LifetimeModelType
  /** Total damage (should be < 1 for survival) */
  totalDamage: number
  /** Remaining damage capacity (1 - totalDamage) */
  remainingDamage: number
  /** Estimated remaining life in percentage */
  remainingLifePercent: number
  /** Damage entries */
  entries: DamageEntry[]
  /** Prediction date */
  predictionDate: string
}

// ============================================
// Weibull Analysis Results
// ============================================

/**
 * Weibull distribution parameters
 */
export interface WeibullParams {
  /** Shape parameter (beta) */
  shape: number
  /** Scale parameter (eta) in cycles */
  scale: number
  /** Location parameter (gamma) - usually 0 for lifetime analysis */
  location?: number
}

/**
 * Weibull analysis result
 */
export interface WeibullAnalysisResult {
  /** Weibull parameters */
  params: WeibullParams
  /** Sample size */
  sampleSize: number
  /** Failure data points */
  failures: number[]
  /** Censored data points */
  censored?: number[]
  /** Confidence level (e.g., 0.9 for 90%) */
  confidenceLevel: number
  /** Mean time to failure (cycles) */
  MTTF: number
  /** Median life (cycles at 63.2% failure probability) */
  medianLife: number
  /** Characteristic life (eta) */
  characteristicLife: number
  /** Reliability at specified cycles */
  reliabilityAtCycles?: { cycles: number; reliability: number }[]
  /** Analysis timestamp */
  timestamp: string
}

// ============================================
// Sensitivity Analysis Results
// ============================================

/**
 * Sensitivity analysis parameter
 */
export interface SensitivityParameter {
  /** Parameter name */
  name: string
  /** Base value */
  baseValue: number
  /** Variation range */
  variation: { min: number; max: number }
  /** Step size */
  step: number
}

/**
 * Sensitivity analysis result
 */
export interface SensitivityAnalysisResult {
  /** Model type analyzed */
  modelType: LifetimeModelType
  /** Base prediction (cycles) */
  basePrediction: number
  /** Analyzed parameters */
  parameters: SensitivityParameter[]
  /** Sensitivity coefficients */
  sensitivities: SensitivityCoefficient[]
  /** Tornado chart data */
  tornadoData: TornadoData[]
  /** Analysis timestamp */
  timestamp: string
}

/**
 * Sensitivity coefficient for a parameter
 */
export interface SensitivityCoefficient {
  /** Parameter name */
  parameter: string
  /** Elasticity (percent change in output / percent change in input) */
  elasticity: number
  /** Partial derivative */
  partialDerivative: number
  /** Normalized sensitivity (0-1) */
  normalized: number
}

/**
 * Tornado chart data point
 */
export interface TornadoData {
  /** Parameter name */
  parameter: string
  /** Minimum value (at parameter min) */
  min: number
  /** Maximum value (at parameter max) */
  max: number
  /** Base value */
  base: number
}

// ============================================
// API Response Types
// ============================================

/**
 * API response envelope
 */
export interface ApiResponse<T = unknown> {
  /** Success flag */
  success: boolean
  /** Response data */
  data?: T
  /** Error message */
  error?: string
  /** Error code */
  errorCode?: string
  /** Pagination metadata */
  meta?: {
    total: number
    page: number
    limit: number
  }
}

// ============================================
// UI State Types
// ============================================

/**
 * Navigation menu item
 */
export interface MenuItem {
  /** Menu item ID */
  id: string
  /** Display name (Chinese) */
  label: string
  /** Icon name */
  icon: string
  /** Route path */
  path: string
  /** Child menu items */
  children?: MenuItem[]
}

/**
 * Form field error
 */
export interface FieldError {
  /** Field name */
  field: string
  /** Error message */
  message: string
}

// ============================================
// Chart Data Types
// ============================================

/**
 * Chart data series
 */
export interface ChartSeries {
  /** Series name */
  name: string
  /** Data points [x, y] or [x, y, z] for 3D */
  data: (number | [number, number] | [number, number, number])[]
  /** Chart type */
  type?: 'line' | 'bar' | 'scatter' | 'area' | 'surface'
  /** Series color */
  color?: string
}

/**
 * Chart configuration
 */
export interface ChartConfig {
  /** Chart title */
  title: string
  /** X-axis label */
  xAxisLabel: string
  /** Y-axis label */
  yAxisLabel: string
  /** Z-axis label (for 3D charts) */
  zAxisLabel?: string
  /** Data series */
  series: ChartSeries[]
  /** Show legend */
  showLegend?: boolean
  /** Show data points */
  showDataPoints?: boolean
}

// ============================================
// Export Settings
// ============================================

/**
 * Export format options
 */
export type ExportFormat = 'csv' | 'xlsx' | 'json' | 'pdf'

/**
 * Export options
 */
export interface ExportOptions {
  /** Export format */
  format: ExportFormat
  /** Include charts */
  includeCharts?: boolean
  /** Include metadata */
  includeMetadata?: boolean
  /** Page orientation for PDF */
  orientation?: 'portrait' | 'landscape'
  /** File name (without extension) */
  fileName?: string
}
